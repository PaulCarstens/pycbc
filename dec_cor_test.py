from matplotlib import use
use('Agg')

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pycbc import scheme
from pycbc.types import Array
from pycbc.types import zeros
from dec_cor import CUDALinearInterpolateCorrelate
from dec_cor import inline_linear_interp_cor

df = 1

#fl = h5py.File("./O1_compressed_000001.hdf", "r")
#freq = np.array(fl["compressed_waveforms"]["-9223347044519888525"]["sample_points"], dtype=np.float32)
#amp = np.array(fl["compressed_waveforms"]["-9223347044519888525"]["amplitude"], dtype=np.float32)
#phase = np.array(fl["compressed_waveforms"]["-9223347044519888525"]["phase"], dtype=np.float32)
ctx = scheme.CUDAScheme(0)
freq = np.zeros(2500, dtype=np.float32)
amp = np.zeros(len(freq), dtype=np.float32)
phase = np.zeros(len(freq), dtype=np.float32)
for i in range(0,len(freq)):
    freq[i] = 20 + (.01*i)**3
    amp[i] = freq[i]/(i+1)
    phase[i] = np.cos(2*freq[i])**2
output1 = zeros(int(max(freq)/df), dtype=np.complex64)
output2 = np.zeros(int(max(freq)/df), dtype=np.complex64)
f = zeros(len(output1), dtype=np.float32)
s = zeros(len(output1), dtype=np.complex64)
s2 = np.zeros(len(output1), dtype=np.complex64)
for i in range(0, len(output1)):
    f[i] = i * df
    s[i] = np.cos(i) + np.sin(i)*1j
    s2[i] = np.cos(i) + np.sin(i)*1j


with ctx:
    intcor = CUDALinearInterpolateCorrelate(s, output1, df)
    intcor.interpolatecorrelate(20, freq, amp, phase)
    #inline_linear_interp_cor(amp, phase, freq, s, output2, df, 20)


A = np.interp(f, freq, amp)
P = np.interp(f, freq, phase)
for i in range(0, len(output2)):
    output2[i] = A[i]*((np.cos(P[i])*s2.real[i]+np.sin(P[i])*s2.imag[i])+(np.cos(P[i])*s2.imag[i]-np.sin(P[i])*s2.real[i])*1j)


#for i in range(0, len(output2)):
#    print A[i]*(np.sin(P[i])+np.cos(P[i])*1j)



for i in range(0, len(output1)):
    dif = 2*(output1[i]-output2[i])/(output1[i]+output2[i])
    magdif = np.sqrt(dif.real**2+dif.imag**2)
    print f[i], output1[i], output2[i], magdif

#plt.plot(f, output1.real, 'r--', f, output1.imag)
#plt.savefig('/home/paul.carstens/public_html/test.png")
