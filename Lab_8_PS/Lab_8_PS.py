'''
    Lab 8 Procesarea Semnalelor
    Florian Luca-Paul, CTI, Grupa 461
'''
import numpy as np
import matplotlib.pyplot as plt

def ex1():
    # Subpunctul (a)
    n = 1000
    x = np.arange(n)

    trend = 2 * x**2 - 5 * x + 7
    season = np.sin(2 * np.pi * 3 * x) + np.sin(2 * np.pi * 5 * x)
    res = np.random.rand(n)
    
    # Scalarea primelor doua componente
    trend = trend * 1e-5
    season = season * 1e12

    series = trend + season + res # Seria de timp

    _, axs = plt.subplots(4, figsize=(7, 7))
    plt.subplots_adjust(hspace=0.37)
    plt.suptitle('Serie de timp cu trei componente')

    axs[0].plot(x, series, 'black')
    axs[0].set_ylabel('Serie de timp')

    axs[1].plot(x, trend, 'r')
    axs[1].set_ylabel('Trend')

    axs[2].plot(x, season, 'g')
    axs[2].set_ylabel('Comp. sezonieră')

    axs[3].plot(x, res, 'b')
    axs[3].set_ylabel('Comp. reziduală')

    for ax in axs.flat:
        ax.grid(linestyle='--')

    # Subpunctul (b)
    def normalize(x):
        return (x - np.mean(x)) / np.std(x)

    nseries = normalize(series)
    autocorr = np.correlate(nseries, nseries, mode='full')
    halfSize = np.round(autocorr.size / 2).astype(np.int32) - 1
    autocorr = autocorr[halfSize:]

    plt.figure(2)

    plt.plot(x, autocorr, 'black')

    plt.title('Vectorul de autocorelație al seriei de timp')
    plt.grid(linestyle='--')

    # Subpunctul (c)
    p = 200
    ar = np.empty(n)

    for i in range(n):
        if i - p < 0:
            ar[i] = 0
        else:
            for j in range(p):
                ar[i] += x[i] * series[i - j]

    plt.figure(3)
    plt.title('Seria de timp (normalizată) și predicțiile sale')

    plt.plot(x, normalize(ar), 'red', label='Model AR', zorder=3)
    plt.plot(x, nseries, 'black', label='Seria de timp', zorder=2)

    plt.grid(linestyle='--', zorder=1)

    plt.show()

        
if __name__ == '__main__':
    ex1()