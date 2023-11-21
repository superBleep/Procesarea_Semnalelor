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
        x = np.convolve(x, x)
        axs[i+1].plot(np.arange(len(x)), x, 'r')
    
    for ax in axs.flat:
        ax.grid(linestyle='--')

    plt.show()


    '''
        Cu fiecare iterație, semnalul converge către o distribuție Gaussiană.
        Numărul de elemente din vectorul semnalului se dublează datorită
        operației de convoluție.
    '''


# nu merge
def ex2():
    n = 3
    a_max = 40
    p = np.array([np.random.randint(-a_max, a_max) for _ in range(n)])
    q = np.array([np.random.randint(-a_max, a_max) for _ in range(n)])

    print(p * q)

    p_fft = np.fft.fft(p)
    q_fft = np.fft.fft(q)

    r_fft = np.convolve(p_fft, q_fft, mode='same')
    r = np.fft.ifft(r_fft).real

    print(r)


def ex3():
    def square_w(x, N_w):
        w = np.ones(N_w)
        return np.convolve(x, w, mode='same')
    
    def hanning_w(x, N_w):
        w = 0.5 * (1 - np.cos((2 * np.pi * x) / N_w))
        return np.convolve(x, w, mode='same')

    time = np.linspace(0, 0.1, 1000)
    x = np.sin(2 * np.pi * 100 * time)
    N_w = 200

    plt.figure(1)

    plt.plot(time, x, 'r', label='Semnal original')
    plt.plot(time, square_w(x, N_w), 'g', label='Fereastră dreptunghiulară')
    plt.plot(time, hanning_w(x, N_w), 'b', label='Fereastră Hanning')

    plt.grid(linestyle='--')
    plt.title(r'Utilizarea ferestrelor asupra unui semnal (f=100, $N_w=200$)')
    plt.xlabel('Timp (s)')
    plt.ylabel('Amplitudine')
    plt.legend()

    plt.show()


def ex4():
    # Subpunctul a
    x = np.genfromtxt('Train.csv', delimiter=',')
    x = x.T[2, 1:].T

    period = 24 * 3
    time = np.arange(period)
    x = x[:period]

    # Subpunctul b
    def sliding_w(x, N_w):
        w = np.ones(N_w)
        return np.convolve(x, w, mode='same') / N_w

    plt.figure(1)

    plt.plot(time, x, zorder=2, alpha=0.4, label='Semnalul original')
    plt.plot(time, sliding_w(x, 5), zorder=3, label=r'$N_w=5$')
    plt.plot(time, sliding_w(x, 9), zorder=4, label=r'$N_w=9$')
    plt.plot(time, sliding_w(x, 13), zorder=5, label=r'$N_w=13$')
    plt.plot(time, sliding_w(x, 17), zorder=6, label=r'$N_w=17$')

    plt.grid(linestyle='--', zorder=1)
    plt.title('Filtrarea de tip medie alunecătoare')
    plt.xlabel('Timp (h)')
    plt.ylabel('Nr. mașini')
    plt.legend()

    # Subpunctul c
    # Cod preluat din laboratorul 5:
    # Extragem componenta continua si
    # afisam modulul transformatei Fourier
    centered_x = x - np.mean(x)
    n = len(centered_x)
    f_s = 1 / 3600

    f = f_s * np.arange(n/2) / n
    f = 1e3 * f

    X = np.fft.fft(centered_x)
    X = np.abs(X / n)
    X = X[:int(n/2)]

    plt.figure(2)

    plt.plot(f, X, 'r')

    plt.title('Modulul transformatei Fourier')
    plt.grid(linestyle='--')
    plt.xlabel('Frecvența (mHz)')
    plt.ylabel('Modul')

    '''
        Conform analizei transformatei Fourier aplicate
        asupra semnalului, se poate observa o scadere considerabila
        a modulului transformatei dupa frecventa de W_n = 0.105 mHz
    '''

    W_n = 0.105
    print('Frecventa de taiere:', W_n, 'mHz')
    Nyquist = (f_s / 2) * 1e3 # mHz
    print('Frecventa normalizata:', round(W_n / Nyquist, 3), 'mHz')

    # Subpunctul d
    N = 5
    rp = 5
    b_butter, a_butter = scipy.signal.butter(N, W_n, btype='low')
    b_cehby1, a_cheby1 = scipy.signal.cheby1(N, rp, W_n, btype='low')
    
    # Subpunctul e
    x_butter = scipy.signal.filtfilt(b_butter, a_butter, x)
    x_cheby1 = scipy.signal.filtfilt(b_cehby1, a_cheby1, x)

    plt.figure(3)

    plt.plot(time, x, label='Semnal original')
    plt.plot(time, x_butter, label='Filtrare Butterworth')
    plt.plot(time, x_cheby1, label='Filtrare Chebyshev')

    plt.grid(linestyle='--')
    plt.title('Filtrărirle Butterworth și Chebyshev (de ordin 5)')
    plt.xlabel('Timp (h)')
    plt.ylabel('Nr. mașini')
    plt.legend()

    '''
        Filtrarea Butterworth este mai potrivita, deoarece
        filtrarea Chebyshev supraestimeaza nr. de masini în
        intervalele orare când acesta este foarte mic. 
    '''

    # Subpunctul f
    N = 5
    rp = 2
    b_butter, a_butter = scipy.signal.butter(N, W_n, btype='low')
    b_cehby1, a_cheby1 = scipy.signal.cheby1(N, rp, W_n, btype='low')

    x_butter = scipy.signal.filtfilt(b_butter, a_butter, x)
    x_cheby1 = scipy.signal.filtfilt(b_cehby1, a_cheby1, x)

    plt.figure(4)

    plt.plot(time, x, label='Semnal original')
    plt.plot(time, x_butter, label='Filtrare Butterworth')
    plt.plot(time, x_cheby1, label='Filtrare Chebyshev')

    plt.grid(linestyle='--')
    plt.title('Filtrărirle Butterworth și Chebyshev ajustate')
    plt.xlabel('Timp (h)')
    plt.ylabel('Nr. mașini')
    plt.legend()

    '''
        La ordine mai mici sau mai mari decat 5, filtrarea Chebyshev
        se deplasează prea mult pe axa Oy a amplitudinii.
    '''

    plt.show()


if __name__ == '__main__':
    ex1()
    ex2()
    ex3()
    ex4()
