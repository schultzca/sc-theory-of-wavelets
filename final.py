import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import subprocess
import sympy as sy
import utilities.image as im
import utilities.quantization as quant
import utilities.wavelet as wv


def problem_1():
    f = sy.symbols('f')
    symbol = wv.battle_lemarie_symbol()

    # plot symbol
    symbol_lambda = sy.lambdify(f, symbol, 'numpy')  # lambdify for plotting
    freq = np.linspace(-0.5, 0.5)
    plt.figure()
    plt.title('B-L Symbol')
    plt.xlabel('frequency')
    plt.plot(freq, [np.abs(symbol_lambda(v)) for v in freq])
    plt.show()


def problem_2():
    f = sy.symbols('f')
    symbol = wv.battle_lemarie_symbol()
    symbol_lambda = sy.lambdify(f, symbol)
    exp = lambda f: np.sqrt(2)*symbol_lambda(f)
    n_coef = 23
    dc, hp, hn = f_coef(exp=exp, period=1, n=n_coef)

    print 'dc:'
    print np.real(dc)
    print 'hp:'
    for v in hp:
        print np.real(v)
    print 'hn:'
    for v in hn:
        print np.real(v)


def problem_3():
    f = sy.symbols('f')
    psi_hat = wv.battle_lemarie_wavelet_transform()
    psi_hat_lambda = sy.lambdify(f, psi_hat, 'numpy')

    freq = np.linspace(-2, 2, 100)
    freq[0] = np.finfo(float).eps
    psi_hat_signal = np.array(np.abs([psi_hat_lambda(v) for v in freq]))

    plt.figure()
    plt.title('Fourier Transform of Mother Wavelet')
    plt.plot(freq, psi_hat_signal)
    plt.show()


def problem_4():
    # Convert scaling function to time domain
    phi_hat = wv.battle_lemarie_scaling_transform()
    Xs, ts = f_trans(phi_hat, 8.1, 128, 0)

    # Plot scaling function
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Scaling Function')
    plt.plot(ts, Xs)

    # Convert wavelet to time domain
    psi_hat = wv.battle_lemarie_wavelet_transform()
    Xs, ts = f_trans(psi_hat, 8.1, 128, 0)

    # Plot wavelet
    plt.subplot(1, 2, 2)
    plt.title('Mother Wavelet')
    plt.plot(ts, Xs)
    plt.show()


def problem_5():
    img = im.read_gecko_image()
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img, cm.Greys_r)

    # Forward Transform
    img_fwd = copy.deepcopy(img)
    dim = max(img.shape)
    while dim >= 8:
        P = wv.permutation_matrix(dim)
        T_a = wv.cdf_24_encoding_transform(dim)
        img_fwd[:dim, :dim] = P.dot(T_a).dot(
            img_fwd[:dim, :dim]).dot(T_a.T).dot(P.T)
        dim = dim / 2

    plt.subplot(1, 3, 2)
    plt.title('Transformed Image')
    plt.imshow(img_fwd, cm.Greys_r)

    # Threshold + Encode
    t, ltmax = quant.log_thresh(img_fwd, cutoff=0.98)
    img_encode = quant.encode(img_fwd, t, ltmax)

    # Store to file
    filename = 'encoded_image'
    img_encode.tofile(filename)
    file_size = os.stat(filename).st_size

    # Compress File
    subprocess.call(['gzip', filename])
    c_file_size = os.stat(filename + '.gz').st_size

    # Decompress File
    subprocess.call(['gunzip', filename + '.gz'])

    # # Read from file
    img_encode = np.fromfile(filename).reshape(img.shape)

    # Decode Image
    img_decode = quant.decode(img_encode, t, ltmax)

    # Inverse Transform
    img_inv = copy.deepcopy(img_decode)
    dim = 8
    while dim <= max(img.shape):
        P = wv.permutation_matrix(dim)
        T_b = wv.cdf_24_decoding_transform(dim)
        img_inv[:dim, :dim] = T_b.T.dot(P.T).dot(
            img_inv[:dim, :dim]).dot(P).dot(T_b)
        dim = dim * 2

    plt.subplot(1, 3, 3)
    plt.title('Recreated Image')
    plt.imshow(img_inv, cm.Greys_r)
    plt.show()

    print "Compression Level: %s" % (1 - float(c_file_size) / float(file_size))

# Utilities


def f_coef(exp, period, n):
    """
    Find fourier coefficients of expression.

    Args:
        exp (function): symbolic function
        period (float): period of symbolic function
        n (int): number of coefficients to compute

    Returns:
        dc (float): dc coefficient
        hp (float): positive coefficients
        hn (float): negative coefficients
    """
    period = float(period)

    # compute delta time
    dt = period/n

    x_sample = np.zeros([n])
    for k in range(n):
        if k == 0:
            x_sample[k] = exp(0 + np.finfo(float).eps)
        else:
            x_sample[k] = exp(k*dt)

    coef = np.fft.fft(x_sample)/n

    dc = coef[0]
    hp = coef[1:n/2+1]
    hn = coef[-n/2+1:]

    return dc, hp, hn


def f_trans(x, F, N, M):
    """
    Discrete Fourier Transform (frequency to time)

    x (symbolic): symbolic function of f
    F (float): frequency range
    N (int): number of points
    M (int): number of aliases
    """
    f = sy.symbols('f')

    dt = 1/F
    df = F/N
    T = N/F

    xp = copy.deepcopy(x)

    for k in range(1, M+1):
        xp = xp+x.subs(f, f-k*F)+x.subs(f, f+k*F)

    # lambdify symbolic function
    xp = sy.lambdify(f, xp, 'numpy')

    xps = np.zeros([N])
    ts = np.zeros([N])
    for n in range(N):
        if n == 0:
            xps[n] = xp(np.finfo(float).eps)
            ts[n] = -T/2
        else:
            xps[n] = xp(n*df)
            ts[n] = n*dt - T/2

    Xs = np.fft.fft(xps)*df
    Xs = np.fft.fftshift(Xs.T)

    return Xs, ts

if __name__ == "__main__":
    problem_1()
    problem_2()
    problem_3()
    problem_4()
    problem_5()
