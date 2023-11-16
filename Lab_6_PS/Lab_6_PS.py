'''
    Lab 6 Procesarea Semnalelor
    Florian Luca-Paul, CTI, Grupa 461
'''
import scipy
import numpy as np
import matplotlib.pyplot as plt


def ex1():
    n = 100
    x = np.random.normal(0, 1, n)

    _, axs = plt.subplots(4, figsize=(6, 7))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle('Convoluție recursivă asupra unui semnal generat aleator (n = 100)')

    axs[0].plot(np.arange(n), x, 'r')
    for i in range(3):
        x = np.convolve(x, np.convolve(x, x))
        axs[i+1].plot(np.arange(len(x)), x, 'r')
    
    for ax in axs.flat:
        ax.grid(linestyle='--')

    plt.show()


    '''
        Cu fiecare iterație, semnalul devine din ce în ce mai
        "zgomotos", iar semnalul converge către o distribuție Gaussiană.
        Numărul de elemente se dublează datorită
        operației de convoluție.
    '''


def ex2():
    n = 3
    a_max = 40
    p = [np.random.randint(-a_max, a_max) for i in range(n)]
    q = [np.random.randint(-a_max, a_max) for i in range(n)]

    r = np.convolve(np.fft.fft(p), np.fft.fft(q))
    r = np.fft.ifft(r)

    print(r)

def ex3():
    def square_w(x):
        w = np.repeat([1], len(x))
        return x * w
    
    def hanning_w(x):
        w = 0.5 * (1 - np.cos(2 * np.pi * x / len(x)))
        return x * w

    time = np.linspace(0, 0.1, 1000)
    x = np.sin(2 * np.pi * 100 * time)

    plt.figure(1)

    plt.plot(time, x)
    plt.plot(time, square_w(x))
    plt.plot(time, hanning_w(x))

    plt.grid(linestyle='--')

    plt.show()

if __name__ == '__main__':
    ex1()
    ex2()
    ex3()