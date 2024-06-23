import numpy as np
import matplotlib.pyplot as plt
import cmath
from math import ceil, floor, pi

k = 8
N = 2*k+1
N_PTS = 10000

def hilbert(x):
    if x == 0:
        return 0
    elif 0 <= x <= 0.5: 
        return -1j
    elif 0.5 < x <= 1:
        return 1j

def main():
    samples = []
    for i in np.arange(0, 1, 1/N):
        samples.append(hilbert(i))

    samples[1]   = -0.9j
    samples[k]   = -0.7j
    samples[k+1] =  0.7j
    samples[2*k]  = 0.9j

    r_1 = np.fft.ifft(samples)
    r_n = np.concatenate((r_1[ceil(N/2):], r_1[:floor(N/2)+1]), axis=None)

    r_f = []
    f = np.arange(0.0, 1.0, 1/N_PTS)
    for F_i in f:
        s = 0
        for n in range(-8, 8+1):
            s += r_n[n+8]*cmath.exp(-1j*2*pi*F_i*n)
        r_f.append(s.imag)

    plt.figure(figsize=(10,5))
    plt.plot(f, r_f)
    plt.plot(f, [hilbert(i).imag for i in f])
    plt.title("Frequency Response")
    plt.legend(['R(F)', 'H_d(F)'])
    plt.show()

    plt.figure(figsize=(10,5))
    plt.stem(np.array(range(-8, 8+1)), r_n)
    plt.title("Impulse Response r[n]")
    plt.legend(['r[n]'])
    plt.show()

if __name__ == "__main__":
    main()
