import numpy, mako.template
from pycuda import gpuarray
from pycuda.tools import dtype_to_ctype
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
import pycbc.scheme
import pycuda.driver as drv
from pycbc.types import zeros



kernel_sources = mako.template.Template("""
texture<float, 1> freq_tex;
texture<float, 1> amp_tex;
texture<float, 1> phase_tex;


__device__ int binary_search(float freq, int lower, int upper){
    /*
       Input parameters:
       =================
       freq:  The target frequency
       lower: The index into the frequency texture at which
              to start the search
       upper: The index into the frequency texture at which
              to end the search
       Return value:
       =============
       The largest index into the frequency texture for
       which the value of the texture at that index is less
       than or equal to the target frequency 'freq'.
     */
    int begin = lower;
    int end = upper;
    while (begin != end){
        int mid = (begin + end)/2;
        float fcomp = tex1Dfetch(freq_tex, (float) mid);
        if (fcomp <= freq){
          begin = mid+1;
        } else {
          end = mid;
        }
    }
    return begin-1;
}


__global__ void find_block_indices(int *lower, int *upper, int texlen,
                                   float df, float flow, float fmax){
    /*
      Input parameters:
      =================
      texlen: The length of the sample frequency texture
      df:     The difference between successive frequencies in the
              output array
      flow:   The minimum frequency at which to generate an interpolated
              waveform
      Global variable:
      ===================
      freq_tex: Texture of sample frequencies (its length is texlen)
      Output parameters:
      ==================
      lower: array of indices, one per thread block, of the lower
             limit for each block within the frequency arrays.
      upper: array of indices, one per thread block, of the upper
             limit for each block within the frequency arrays.
    */
    // This kernel is launched with only one block; the number of
    // threads will equal the number of blocks in the next kernel.
    int i = threadIdx.x;
    // We want to find the index of the smallest freqency in our
    // texture which is greater than the freqency fmatch below:
    float ffirst = i*df*${ntpb};
    float flast = (i+1)*df*${ntpb}-df;
    if (ffirst < flow){
       ffirst = flow;
    }
    lower[i] = binary_search(ffirst, 0, texlen);
    upper[i] = binary_search(flast, 0, texlen) + 1;
    return;
}


__global__ void linear_interp_correlate(float2 *sh, float df, int hlen,
                              float flow, float fmax, int texlen,
                              int *lower, int *upper, float2 *s){
    /*
      Input parameters:
      =================
      df:     The difference between successive frequencies in the
              output array
      hlen:   The length of the output array
      flow:   The minimum frequency at which to generate an interpolated
              waveform
      fmax:   The maximum frequency in the sample frequency texture; i.e.,
              freq_tex[texlen-1]
      texlen: The common length of the three sample textures
      lower:  Array that for each thread block stores the index into the
              sample frequency array of the largest sample frequency that
              is less than or equal to the smallest frequency considered
              by that thread block.
      upper:  Array that for each thread block stores the index into the
              sample frequency array of the smallest sample frequency that
              is greater than the next frequency considered *after* that
              thread block.
      s:      Waveform to be correlated with h.
      Global variables:
      ===================
      freq_tex:  Texture of sample frequencies (its length is texlen)
      amp_tex:   Texture of amplitudes corresponding to sample frequencies
      phase_tex: Texture of phases corresponding to sample frequencies
      Output parameters:
      ==================
      sh: array of complex 
    */
    __shared__ int low[1];
    __shared__ int high[1];
    int idx;
    float2 tmp;
    //float tmp;
    float amp, freq, phase, inv_df, x, y;
    float a0, a1, f0, f1, p0, p1;
    // Load values in global memory into shared memory that
    // all threads in this block will use:
    if (threadIdx.x == 0) {
        low[0] = lower[blockIdx.x];
        high[0] = upper[blockIdx.x];
    }
    __syncthreads();
    int i = ${ntpb}*blockIdx.x + threadIdx.x;
    if (i < hlen){
        freq = df*i;
        if ( (freq<flow) || (freq>fmax) ){
          //tmp = 0.0;
          tmp.x = 0.0;
          tmp.y = 0.0;
        } else {
          idx = binary_search(freq, low[0], high[0]);
          if (idx < texlen-1) {
              f0 = tex1Dfetch(freq_tex, idx);
              f1 = tex1Dfetch(freq_tex, idx+1);
              inv_df = 1.0/(f1-f0);
              a0 = tex1Dfetch(amp_tex, idx);
              a1 = tex1Dfetch(amp_tex, idx+1);
              p0 = tex1Dfetch(phase_tex, idx);
              p1 = tex1Dfetch(phase_tex, idx+1);
              amp = a0*inv_df*(f1-freq) + a1*inv_df*(freq-f0);
              phase = p0*inv_df*(f1-freq) + p1*inv_df*(freq-f0);
          } else {
             // We must have idx = texlen-1, so this frequency
             // exactly equals fmax
             amp = tex1Dfetch(amp_tex, idx);
             phase = tex1Dfetch(phase_tex, idx);
          }
          __sincosf(phase, &y, &x);
          tmp.x = amp*x*s[i].x + amp*y*s[i].y;
          tmp.y = amp*x*s[i].y - amp*y*s[i].x;
          //tmp = amp;
          //tmp.x = s[i].x;
          //tmp.y = s[i].y;

        }
       sh[i] = tmp;
    } 
    return;
}
""")




def decompress_correlate_test(amps, phases, freqs, s, output, df, flow=None, hlen=None):
    if flow is None:
        flow = freqs[0]
    else:
        flow = numpy.float32(flow)

    texlen = numpy.int32(len(freqs))
    fmax = numpy.float32(freqs[texlen-1])
    hlen = numpy.int32( len(output) )
    df = numpy.float32(df)

    nt = 1024
    nb = int(numpy.ceil(hlen / 1024.0))
    mod = SourceModule(kernel_sources.render(ntpb=nt,nblocks=nb))
    freq_tex = mod.get_texref("freq_tex")
    amp_tex = mod.get_texref("amp_tex")
    phase_tex = mod.get_texref("phase_tex")

    fn1 = mod.get_function("find_block_indices")
    fn1.prepare("PPifff", texrefs=[freq_tex])
    fn2 = mod.get_function("linear_interp_correlate")
    fn2.prepare("PfiffiPPP", texrefs=[freq_tex, amp_tex, phase_tex])
    
    freqs_gpu = gpuarray.to_gpu(freqs)
    freqs_gpu.bind_to_texref_ext(freq_tex, allow_offset=False)
    amps_gpu = gpuarray.to_gpu(amps)
    amps_gpu.bind_to_texref_ext(amp_tex, allow_offset=False)
    phases_gpu = gpuarray.to_gpu(phases)
    phases_gpu.bind_to_texref_ext(phase_tex, allow_offset=False)
    fn1 = fn1.prepared_call
    fn2 = fn2.prepared_call
    out_gpu = output.data.gpudata
    s_gpu = s.data.gpudata
    lower = zeros(nb, dtype=numpy.int32).data.gpudata
    upper = zeros(nb, dtype=numpy.int32).data.gpudata
    fn1((1, 1), (nb, 1, 1), lower, upper, texlen, df, flow, fmax)
    fn2((nb, 1), (nt, 1, 1), out_gpu, df, hlen, flow, fmax, texlen, lower, upper, s_gpu)
    pycbc.scheme.mgr.state.context.synchronize()
    return output




