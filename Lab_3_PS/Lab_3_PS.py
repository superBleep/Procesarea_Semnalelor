'''
    Lab 3 Procesarea Semnalelor
    Florian Luca-Paul, CTI, Grupa 461
'''
import numpy as np
import matplotlib.pyplot as plt

def ex1():
    n = 8
    
    F = np.empty((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            F[i][j] = np.e ** (-2 * np.pi * 1j * i * j / n)

    _, axs = plt.subplots(n, 2, figsize=(7, 7))
    plt.suptitle('Părțile reale și imaginare ale matricei Fourier (n = 8)')
    plt.subplots_adjust(wspace=0.3, hspace=0.7)
    for i in range(n):
        axs[i][0].plot(np.arange(n), F[i][:].real, 'r')
        axs[i][1].plot(np.arange(n), F[i][:].imag, 'b')

    axs[0][0].set_title('Partea reală (cos)')
    axs[0][1].set_title('Partea imaginară (sin)')

    for ax in axs.flat:
        ax.grid(linestyle='--')

    plt.show()

    F = F / np.sqrt(n)
    F_H = F.conjugate().T
    prod = np.dot(F, F_H)
    if np.allclose(prod, np.eye(n, dtype=np.complex128)):
        print('Matricea Fourier scalată este unitară')
    else:
        print('Matricea Fourier scalată NU este unitară')


def ex2():
    def wrap(omega):
        z = np.empty(n, dtype=np.complex128)
        for i in range(n):
            z[i] = x[i] * np.e ** (-2 * np.pi * 1j * omega * time[i])

        return z
    
    def dist(x, y):
        return np.sqrt(x**2 + y**2)
    
    f = 7 # Frecventa sinusoidei
    f_s = 1000 # Frecventa de esantionare
    sec = 3 # Durata semnalului
    n = sec * f_s # Nr. de esatnioane
    
    time = np.linspace(0, sec, n)
    x = np.sin(2 * np.pi * f * time)
    
    y = np.empty(n, dtype=np.complex128)
    for i in range(n):
        y[i] = x[i] * np.e ** (-2 * np.pi * 1j * time[i])

    omegas = [1, 5, f, 9]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    _, axs = plt.subplots(2)
    plt.suptitle(r'Semnal sinusoidal și reprezentarea sa în planul complex (f = {} Hz)'.format(f))
    plt.subplots_adjust(hspace=0.5)
    axs[0].plot(time, x, 'g')
    axs[0].set_xlabel('Timp (secunde)')
    axs[0].set_ylabel('Amplitudine')
    axs[0].axhline(0, c='black')

    axs[1].scatter(y.real, y.imag, s=2, c=dist(y.real, y.imag), cmap='rainbow')
    axs[1].set_xlim([-1.2, 1.2])
    axs[1].set_ylim([-1.2, 1.2])
    axs[1].set_xlabel('Real')
    axs[1].set_ylabel('Imaginar')
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].axhline(0, c='black')
    axs[1].axvline(0, c='black')
    
    for ax in axs.flat:
        ax.grid(linestyle='--')

    _, axs2 = plt.subplots(2, 2, figsize=(8, 6))
    plt.suptitle('Frecvențe de înfășurare ale cercului unitate')
    plt.subplots_adjust(wspace=0.28, hspace=0.4)

    for i, omega in enumerate(omegas):
        xg = wrap(omega).real
        yg = wrap(omega).imag
        axs2[positions[i]].scatter(xg, yg, s=2, c=dist(xg,yg), cmap='rainbow')
        axs2[positions[i]].set_title(r'$\omega$ = {}'.format(omega))

    for ax in axs2.flat:
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.axhline(0, c='black')
        ax.axvline(0, c='black')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginar')
        ax.grid(linestyle='--')

    plt.show()


def ex3():
    n = 1000
    time = np.linspace(0, 1, n)
    omegas = [30, 15, 18]
    x = np.sin(2 * np.pi * omegas[0] * time) + 0.7 * np.cos(2 * np.pi * omegas[1] * time) + 3 * np.sin(2 * np.pi * omegas[2] * time)

    X = np.empty(n, dtype=np.complex128)
    for o in range(n):
        for i in range(n):
            X[o] += x[i] * np.e ** (-2 * np.pi * 1j * i * o / n)

    _, axs = plt.subplots(2)
    plt.suptitle('Modulul transformatei Fourier pentru un semnal')
    plt.subplots_adjust(hspace=0.4)
    axs[0].plot(time, x)
    axs[0].set_xlabel('Timp (s)')
    axs[0].set_ylabel('x(t)')

    axs[1].stem(np.arange(n), np.sqrt(X.real**2 + X.imag**2), markerfmt='o', linefmt='black', basefmt=' ')
    axs[1].set_xlabel('Frecvența (Hz)')
    axs[1].set_ylabel(r'$\vert X(\omega) \vert$')

    for ax in axs.flat:
        ax.grid(linestyle='--')

    plt.show()


def main():
    ex1()
    ex2()
    ex3()


if __name__ == '__main__':
    main()
