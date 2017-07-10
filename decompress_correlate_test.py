from matplotlib import use
use('Agg')

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pycbc import scheme
from pycbc.types import Array
from pycbc.types import zeros
from decompress_correlate_test import decompress_correlate_test

df = 1

ctx = scheme.CUDAScheme(0)
fl = h5py.File("./O1_compressed_000001.hdf", "r")
freq = np.array(fl["compressed_waveforms"]["-9223347044519888525"]["sample_points"], dtype=np.float32)
amp = np.zeros(len(freq), dtype=np.float32)
for i in range(0,len(amp)):
    amp[i] = 1
phase = np.zeros(len(freq), dtype=np.float32)
for i in range(0,len(phase)):
    phase[i] = np.pi/4
s = zeros(len(freq), dtype=np.complex64)
for i in range(0,len(s)):
    s[i] = 1 + 1j
output = zeros(int(max(freq)/df), dtype=np.complex64)

f = zeros(len(output), dtype=np.float32)
for i in range(0, len(output)):
    f[i] = i * df


with ctx:
    decompress_correlate_test(amp, phase, freq, s, output, df)


for i in range(0, len(output)):
    print f[i], output[i]

#for i in output:
#    print i


#plt.plot(f, output.real, 'r--', f, output.imag)
#plt.savefig('/home/paul.carstens/public_html/test.png')
