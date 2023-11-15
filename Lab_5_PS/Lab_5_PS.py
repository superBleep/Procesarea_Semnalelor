'''
    Lab 5 Procesarea Semnalelor
    Florian Luca-Paul, CTI, Grupa 461
'''
import numpy as np
import matplotlib.pyplot as plt


def ex1():
    # Subpunctul d
    x = np.genfromtxt('Train.csv', delimiter=',')
    x = x.T[2, 1:].T # Scoatem esantioanele din csv
    
    n = len(x)
    f_s = 1 / 3600

    f = f_s * np.arange(n/2) / n
    f = 1e3 * f # Scalarea domeniului de frecvente

    X = np.fft.fft(x)
    X = np.abs(X / n) # Modulul transformatei
    X = X[:int(n/2)] # Injumatatirea spectrului

    plt.figure(1)

    plt.plot(f, X, 'r')

    plt.title('Modulul transformatei Fourier')
    plt.grid(linestyle='--')
    plt.xlabel('Frecvența (mHz)')
    plt.ylabel('Amplitudine')

    plt.show()

    # Subpunctul e
    centered_x = x - np.mean(x)

    # Subpunctul f
    X_abs = np.fft.fft(centered_x)
    X_abs = np.abs(X_abs / n)
    X_abs = X_abs[:int(n/2)]

    top4_indices = np.argpartition(X_abs, -4)[-4:]
    top4_abs = X_abs[top4_indices]
    top4_freq = f[top4_indices]

    print('Primele 4 cele mai mari module:', top4_abs)
    print('Frecvențe asociate primelor 4 module (Hz)', top4_freq)

    # Subpunctul g
    start = 5592 # Esantionul de inceput
    stop = start + 24 * 30
    samples = np.linspace(start, stop, stop - start)

    plt.figure(1)

    plt.plot(samples, x[start:stop], 'g')

    plt.title('Nr. de mașini care trec într-o perioada de o lună')
    plt.grid(linestyle='--')
    plt.xlabel('Eșantioane')
    plt.ylabel('Nr. mașini')

    plt.show()


if __name__ == '__main__':
    ex1()
