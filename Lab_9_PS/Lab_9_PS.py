'''
    Lab 9 Procesarea Semnalelor
    Florian Luca-Paul, CTI, Grupa 461
'''
import numpy as np
import matplotlib.pyplot as plt

def ex1():
    n = 200
    x = np.arange(n)

    trend = 2 * x**2 - 5 * x + 7
    season = np.sin(2 * np.pi * 3 * x) + np.sin(2 * np.pi * 5 * x)
    res = np.random.rand(n)
    
    trend = trend * 1e-5
    season = season * 1e12

    series = trend + season + res

    return series


def ex2(x):
    n = np.size(x)

    s = np.empty(n) # Aplicarea medierei exponentiale
    s[0] = x[0]
    alfa_og = 0.2 # Alfa fixat

    for i in range(1, n):
        s[i] = alfa_og * x[i] + (1 - alfa_og) * s[i - 1]

    sTuned = np.empty(n) # Mediere exponentiala cu alfa dinamic
    sTuned[0] = x[0]

    for i in range(1, n):
        a = []
        b = []
        for j in range(i):
            a.append(x[j] - s[j - 1])
            b.append(x[j + 1] - s[j - 1])

        alfa = np.dot(a, b) / np.linalg.norm(a)

        sTuned[i] = alfa * x[i] + (1 - alfa) * sTuned[i - 1]

    _, axs = plt.subplots(3, figsize=(6, 7))
    plt.suptitle('Serii de timp nenormalizate')
    plt.subplots_adjust(hspace=0.655)

    axs[0].plot(np.arange(n), x, 'r')
    axs[0].set_title('Seria originală')

    axs[1].plot(np.arange(n), s, 'g')
    axs[1].set_title(f'Rez. medierei exponențiale (alfa={alfa_og})')

    axs[2].plot(np.arange(n), sTuned, 'purple')
    axs[2].set_title('Rez. medierei exponențiale (alfa dinamic)')

    for ax in axs.flat:
        ax.set_xlabel('Timp (t)')
        ax.set_ylabel('Valori')
        ax.grid(linestyle='--')

    plt.show()


def ex3(x):
    n = np.size(x)
    q = 50 # Orizontul modelului MA

    y = []
    for i in range(n):
        if i < q:
            y.append(0)
        else:
            # todo
            pass


def main():
    x = ex1() # Seria de timp

    ex2(x)
    ex3(x)


if __name__ == '__main__':
    main()