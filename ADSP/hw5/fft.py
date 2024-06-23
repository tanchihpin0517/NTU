import cmath

def fft(signal):
    """
    Compute the FFT of a signal using the Cooley-Tukey FFT algorithm.

    Parameters:
    signal (list of complex): Input signal.

    Returns:
    list of complex: FFT of the input signal.
    """
    N = len(signal)
    if N <= 1:
        return signal

    even = fft(signal[0::2])
    odd = fft(signal[1::2])

    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k % len(odd)] for k in range(N)]
    return [even[k % len(even)] + T[k] for k in range(N // 2)] + \
           [even[k % len(even)] - T[k] for k in range(N // 2)]

def fftreal(x, y):
    """
    Compute the FFT of two N-point real signals x and y using only one N-point FFT.

    Parameters:
    x (list of float): First real signal of length N.
    y (list of float): Second real signal of length N.

    Returns:
    tuple: FFTs of x and y (Fx, Fy).
    """
    N = len(x)
    if len(x) != len(y):
        raise ValueError("Input signals x and y must have the same length")

    # Form a complex signal from x and y
    z = [x[i] + 1j * y[i] for i in range(N)]

    # Compute the FFT of the complex signal
    Z = fft(z)

    # Extract the FFTs of x and y
    Z_conj = fft([z_i.conjugate() for z_i in z])

    Fx = [0.5 * (Z[k] + Z_conj[k].conjugate()).real for k in range(N)]
    Fy = [0.5 * (Z[k] - Z_conj[k].conjugate()).imag for k in range(N)]

    return Fx, Fy

# Example usage
x = [1.0, 2.0, 3.0, 4.0]
y = [5.0, 6.0, 7.0, 8.0]

Fx, Fy = fftreal(x, y)

print("FFT of x:", Fx)
print("FFT of y:", Fy)
