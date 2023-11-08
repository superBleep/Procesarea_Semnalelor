'''
    Lab 4 Procesarea Semnalelor
    Florian Luca-Paul, CTI, Grupa 461
'''
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy
from os.path import exists

def ex1():
    def dft(n, x):
        start = time.time()
        F = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                F[i][j] = np.e ** (-2 * np.pi * i * j / n)
        matrix_time = time.time() - start

        start = time.perf_counter()
        np.dot(F, x)
        dft_time = time.perf_counter() - start

        return matrix_time, dft_time

    N = [2**i for i in range(7, 14)]

    if exists('exec_times.npy'):
        print('Loading exec times from disk...')
        times = np.load('exec_times.npy')
    else:
        print('Exec times not found. Computing...')
        domains = [np.linspace(0, 3, n) for n in N]
        xs = [np.sin(2 * np.pi * d) for d in domains]
        times = np.empty((len(N), 3))

        for i, n in enumerate(N):
            matrix_time, dft_time = dft(n, xs[i])

            start = time.perf_counter()
            np.fft.fft(xs[i], n)
            npfft_time = time.perf_counter() - start

            times[i] = [matrix_time, dft_time, npfft_time]
            print(f'Computed X for n = {n}...')

        print('Writing exec times to disk...', end='')
        np.save('exec_times.npy', times)
        print('done')

    plt.figure(1)

    plt.plot(np.arange(len(N)), [t[1] for t in times], 'r', label='DFT')
    plt.plot(np.arange(len(N)), [t[2] for t in times], 'b', label='FFT (Numpy)')

    plt.title('Timpi de execuție ai transformatelor Fourier discrete și rapide')
    plt.xlabel('Dimensiunile vectorilor')
    plt.xticks(np.arange(len(N)), N)
    plt.yscale('log')
    plt.ylabel('Timpul de executie (s)')
    plt.grid(linestyle='--')
    plt.legend()

    plt.figure(2)

    plt.plot(np.arange(len(N)), [t[0] for t in times], 'g')

    plt.title('Timpi de execuție ai calculului matricei Fourier')
    plt.xlabel('Dimensiunile vectorilor')
    plt.xticks(np.arange(len(N)), N)
    plt.yscale('log')
    plt.ylabel('Timpul de executie (s)')
    plt.grid(linestyle='--')

    plt.show()


def ex2():
    f = 20
    f_s = 15
    time = np.linspace(0, 0.3, 1000)
    time_s = np.linspace(0, 0.3, round(0.3 * f_s))
    
    _, axs = plt.subplots(3, figsize=(7, 7))
    plt.suptitle('Fenomenul de aliere (aliasing)')
    plt.subplots_adjust(hspace=0.405)

    colors = ['r', 'g', 'b']
    for k in range(3):
        axs[k].plot(time, np.sin(2 * np.pi * (f + k * f_s) * time), colors[k], zorder=2)
        axs[k].scatter(time_s, np.sin(2 * np.pi * (f + k * f_s) * time_s), color='black', zorder=3)

    for ax in axs.flat:
        ax.grid(linestyle='--', zorder=1)
        ax.set_xlabel('Timp (s)')
        ax.set_ylabel('Amplitudine')

    plt.show()


def ex6():
    rate, x = scipy.io.wavfile.read('vocale_legate.wav')   
    x = (x[:, 0] + x[:, 1]) / 2 # Conversie din stereo in mono

    duration = len(x) / rate
    time = np.linspace(0, duration, rate)

    # Padding pentru impartirea in grupuri
    # de dimensiuni egale
    while len(x) % 100 != 0:
        x = np.append(x, 0)

    x = np.array_split(x, 100)
    
    X = []
    for slice in x:
        X.append(np.abs(np.fft.fft(slice)))
    X = np.array(X).T


def main():
    ex1()
    ex2()
    ex6()


if __name__ == '__main__':
    main()
