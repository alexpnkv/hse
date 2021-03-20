import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from cmath import sqrt

from scipy.fftpack import fft
import math


def omega(p, q):
    return np.exp((-2j * np.pi * p) / q)


def my_fft0(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def my_fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = my_fft(x[0::2])
    odd = my_fft(x[1::2])
    combined = [0] * N
    for k in range(N//2):
         combined[k] = even[k] + omega(k, N) * odd[k]
         combined[k + N//2] = even[k] - omega(k, N) * odd[k]
    return combined


def my_fft2(x):
    n = len(x)

    j = sqrt(-1)

    y = [0] * n
    for k in range(0, n):
        for l in range(0, n):
            y[k] += x[l] * (np.exp((-np.pi * j * k * 2) / n))

    return y


def task(args):
    pid = args[0]
    num_threads = args[1]
    values = args[2]
    S = len(values)
    s = S // num_threads
    start = s * pid
    count = s if (pid != (num_threads - 1)) else s + S % num_threads
    end = start + count
    res = my_fft(values[start:end])
    return [pid, res]


if __name__ == '__main__':

    N = 2 ** 8
    T = 1.0 / 800.0
    x = np.linspace(0, N * T, N)
    # y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    y = np.sin(50.0 * 2.0 * np.pi * x)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    thr = 8
    with mp.Pool(thr) as p:
        mp_res = p.map(task, [(i, thr, y) for i in range(thr)])

    start_time_0 = time.time()
    mp_yf = []
    for i in mp_res:
        mp_yf += i[1]
    mp_yfa = 2.0/N * np.abs(mp_yf[0:N//2])
    plt.plot(xf, mp_yfa)
    print('mp time: '+str(time.time() - start_time_0))

    start_time_1 = time.time()
    yf = my_fft(y)
    print('time: '+str(time.time() - start_time_1))
    yfa = 2.0/N * np.abs(yf[0:N//2])
    plt.plot(xf, yfa)



    """
    start_time_2 = time.time()
    ref_yf = fft(y)
    print('ref time: '+str(time.time() - start_time_2))
    ref_yfa = 2.0/N * np.abs(ref_yf[0:N//2])
    plt.plot(xf, ref_yfa)
    """
    plt.show()





