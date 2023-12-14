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


def ex2():
    x = ex1()
    n = np.size(x)

    s = np.empty(n)
    s[0] = x[0]
    alfa = 0.2

    for i in range(1, n):
        s[i] = alfa * x[i] + (1 - alfa) * s[i - 1]

    sTuned = np.empty(n)
    sTuned[0] = x[0]

    for i in range(1, n):
        a = np.empty(i)
        b = np.empty(i)

        for j in range(1, i):
            a[j - 1] = x[j] - s[j - 1]
        
        for j in range(2, i):
            b[j - 2] = x[j + 1] - s[j - 1]

        alfa = np.dot(a, b) / np.dot(a, a)

        sTuned[i] = alfa * x[i] + (1 - alfa) * sTuned[i - 1]


    plt.figure(1)

    plt.plot(np.arange(n), x, label='Seria originală')
    plt.plot(np.arange(n), s, label='Rez. medierii exponențiale')
    plt.plot(np.arange(n), sTuned, label='Rez. medierii exponențiale (alfa dinamic)')

    plt.title('Serii de timp nenormalizate')
    plt.xlabel('Timp (t)')
    plt.ylabel('Valori')
    plt.legend()
    plt.grid(linestyle='--')

    plt.show()


def main():
    ex2()


if __name__ == '__main__':
    main()