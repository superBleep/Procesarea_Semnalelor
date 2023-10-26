'''
    Lab 3 Procesarea Semnalelor
    Florian Luca-Paul, CTI, Grupa 461
'''
import numpy as np
import matplotlib.pyplot as plt

def ex1():
    N = 8
    time = np.linspace(0, 3, N)
    x = np.sin(2 * np.pi * time)
    F = np.zeros((N,N), dtype=np.complex128)

    # 16 grafice, 8 cos si 8 sin
    # pune in git png si pdf
    # pt 1, odata cu X, Fx
    # la restul cum vrem

    for i in range(N):
        for j in range(N):
            F[i][j] = np.e ** (2 * np.pi * 1j * i * j / N)

    _, axs = plt.subplots(N, 2)
    plt.suptitle('Partea reala/imaginara a elementelor matricei Fourier')
    k = 0
    for i in range(N):
        axs[i][0].plot(np.arange(8), [e.real for e in F[k]], 'b')
        axs[i][1].plot(np.arange(8), [e.imag for e in F[k]], 'r')
        k += 1

    for ax in axs.flat:
        ax.grid(linestyle='--')

    ## Comparatia cu matricea unitate cu un ebsilon dat (np.all?)

    plt.show()

def main():
    ex1()


if __name__ == '__main__':
    main()