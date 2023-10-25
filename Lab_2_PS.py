'''
    Lab 2 Procesarea Semnalelor
    Florian Luca-Paul, CTI, Grupa 461
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice as sound


def ex1():
    time = np.linspace(0, 3, 1000)

    fig, axs = plt.subplots(2)
    fig.suptitle('Semnalele sin și cos (t=0:3)')
    fig.subplots_adjust(hspace=0.5)
    axs[0].plot(time, 2 * np.sin(2 * np.pi * 4 * time), 'r')
    axs[1].plot(time, 2 * np.cos(2 * np.pi * 4 * time - np.pi / 2), 'g')

    for ax in axs.flat:
        ax.set_xlabel('Timp (t)')
        ax.set_ylabel('Amplitudine (A)')
        ax.grid(linestyle='--')

    plt.show()


def ex2():
    def noisify(domain, snr, sample_nr):
        x = np.sin(2 * np.pi * domain)
        z = np.random.normal(0, 1, sample_nr)
        gamma = np.sqrt(np.linalg.norm(x)**2 / (snr * np.linalg.norm(z)**2))

        return x + gamma * z

    sample_nr = 1000
    domain = np.linspace(0, 3, sample_nr)
    
    plt.figure(1)

    plt.plot(domain, np.sin(2 * np.pi * domain), label=r"$\phi=0$")
    plt.plot(domain, np.sin(2 * np.pi * domain + np.pi / 2), label=r"$\phi=\frac{\pi}{2}$")
    plt.plot(domain, np.sin(2 * np.pi * domain + np.pi), label=r"$\phi=\pi$")
    plt.plot(domain, np.sin(2 * np.pi * domain + 3 * np.pi / 2), label=r"$\phi=\frac{3\pi}{2}$")
    
    plt.grid(linestyle='--')
    plt.title('Semnale sinusoidale de faze diferite (f = 1 Hz, t = 0:31)')
    plt.xlabel('TImp (t)')
    plt.ylabel('Amplitudine (A)')
    plt.legend(loc='upper right')

    plt.figure(2)

    plt.plot(domain, noisify(domain, 0.1, sample_nr), 'r', label='SNR = 0.1')
    plt.plot(domain, noisify(domain, 1, sample_nr), 'g', label='SNR = 1')
    plt.plot(domain, noisify(domain, 10, sample_nr), 'b', label='SNR = 10')
    plt.plot(domain, noisify(domain, 100, sample_nr), 'purple', label='SNR = 100')

    plt.grid(linestyle='--')
    plt.title(r'Semnale sinusoidale cu zgomot (f = 1 Hz, $\phi$ = 0, t = 0:3)')
    plt.xlabel('TImp (t)')
    plt.ylabel('Amplitudine (A)')
    plt.legend(loc='upper right')

    plt.show()


def ex3():
    rate = 44100
    seconds = 1
    time = np.linspace(0, seconds, seconds*rate)

    s1 = np.sin(2 * np.pi * 400 * time)
    s2 = np.sin(2 * np.pi * 800 * time)
    s3 = 2 * (time * 240 - np.floor(1 / 2 + time * 240))
    s4 = np.sign(np.sin(2 * np.pi * 300 * time))

    signals = [s1, s2, s3, s4]
    descr = [
        'sinusoidal, 400 Hz' ,
        'sinusoidal, 800 Hz',
        'sawtooth, 240 Hz',
        'square, 300 Hz'
    ]
    print('/// Redare de semnale ({}s) ///'.format(seconds))
    for i, signal in enumerate(signals):
        sound.play(signal)
        print("Se reda semnalul:", descr[i] + '...')
        sound.wait()
    print('/// Redare terminata ///\n')

    scipy.io.wavfile.write('sawtooth.wav', rate, s3)
    _, new_s3 = scipy.io.wavfile.read('sawtooth.wav')
    if (s3 == new_s3).all():
        print('Incarcare pe disc cu succes\n')


def ex4():
    time = np.linspace(0, 3, 1000)

    fig, axs = plt.subplots(3)
    fig.suptitle('Semnale însumate (t = 0:3)')
    fig.subplots_adjust(hspace=0.6)
    axs[0].plot(time, np.sin(2 * np.pi * 5 * time), 'r', label='sin, 5 Hz')
    axs[1].plot(time, np.sign(np.sin(2 * np.pi * 3 * time)), 'g', label='square, 3 Hz')
    axs[2].plot(time, np.sin(2 * np.pi * 5 * time) + np.sign(np.sin(2 * np.pi * 3 * time)), 'b', label='Semnal sumă')

    fig.legend()
    for ax in axs.flat:
        ax.grid(linestyle='--')
        ax.set_xlabel('Timp (t)')
        ax.set_ylabel('Amplitudine (A)')

    plt.show()


def ex5():
    seconds = 3
    rate = 44100
    time = np.linspace(0, seconds, seconds * rate)

    s1 = np.sin(2 * np.pi * 400 * time)
    s2 = np.sin(2 * np.pi * 600 * time)

    s = np.append(s1, s2)

    sound.play(s, rate)
    print('Se reda semnalul')
    sound.wait()

    '''
        Semnalul cu frecventa mai mare se aude
        mai ascutit decat cel cu frecventa mai joasa.
    '''


def ex6():
    freq = 200
    time = np.linspace(0, 1, freq)

    plt.figure(1)

    plt.plot(time, np.sin(2 * np.pi * (freq / 2) * time), 'b', label='f = 100 Hz')
    plt.plot(time, np.sin(2 * np.pi * (freq / 4) * time), 'y', label='f = 50 Hz')
    plt.plot(time, np.sin(2 * np.pi * 0 * time), 'purple', label = 'f = 0 Hz')
    
    plt.title(r'Frecvențe fundamentale ale semnalului sinusoidal ($f_s$ = 200 Hz, t = 0:1)')
    plt.xlabel('Timp (t)')
    plt.ylabel('Amplitudine (A)')
    plt.grid(linestyle='--')
    plt.legend()
    plt.show()

    '''
        Observatii:
        Amplitudinile semnalelor cu f = f_s / 2 si f = f_s / 4 descriu
        clopote Gaussiene pe graficul in timp
    '''


def ex7():
    freq = 1000
    time = np.linspace(0, 3, 3*freq)

    s1 = np.sin(2 * np.pi * time)
    s2 = s1[::3]
    s3 = s1[1::3]

    fig, axs = plt.subplots(3)
    fig.suptitle('Semnale sinusoidale decimate (t = 0:3)')
    fig.subplots_adjust(hspace=0.5)
    axs[0].plot(time, s1, 'r')
    axs[1].plot(time[::3], s2, 'g')
    axs[2].plot(time[1::3], s3, 'b')

    for ax in axs.flat:
        ax.set_xlabel('Timp (t)')
        ax.set_ylabel('Amplitudine (A)')
        ax.grid(linestyle='--')

    plt.show()

    '''
        Toate functiile sunt identice pe intervalul t = 0:3,
        deoarece frecventa de esantionare este foarte mare;
        la frecvente mai mici (ex: 20 Hz), diferentele dintre cele trei
        decimari se vad mult mai clar
    '''


def ex8a():
    domain = np.linspace(-np.pi/2, np.pi/2, 400)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle('Aproximările Taylor și Pade ale funcției sinus')
    fig.subplots_adjust(wspace=0.3, hspace=0.4)

    axs[0,0].plot(domain, np.sin(domain), 'r', label=r'$sin(\alpha)$')
    axs[0,0].plot(domain, domain, 'g', label='Aprox. Taylor')

    axs[0, 0].set_title('Aproximarea Taylor a funcției sinus')
    axs[0, 0].set_xlabel(r'$Alfa (\alpha)$')
    axs[0, 0].set_ylabel('Valori')
    axs[0, 0].legend()

    axs[0, 1].plot(domain, np.sin(domain), 'r', label=r'$sin(\alpha)$')
    axs[0, 1].plot(domain, (domain - 7/60 * domain**3) / (1 + domain**2 / 20), 'g', label='Aprox. Pade')

    axs[0, 1].set_title('Aproximarea Pade a funcției sinus')
    axs[0, 1].set_xlabel(r'$Alfa (\alpha)$')
    axs[0, 1].set_ylabel('Valori')
    axs[0, 1].legend()

    axs[1, 0].plot(domain, np.abs(np.sin(domain) - domain), 'b')

    axs[1, 0].set_title('Eroarea absolută (aprox. Taylor)')
    axs[1, 0].set_xlabel(r'$Alfa (\alpha)$')
    axs[1, 0].set_ylabel('Eroare')

    axs[1, 1].plot(domain, np.abs(np.sin(domain) - (domain - 7/60 * domain**3) / (1 + domain**2 / 20)), 'b')

    axs[1, 1].set_title('Eroarea absolută (aprox. Pade)')
    axs[1, 1].set_xlabel(r'$Alfa (\alpha)$')
    axs[1, 1].set_ylabel('Eroare')

    for ax in axs.flat:
        ax.grid(linestyle='--')

    plt.show()


def ex8b():
    domain = np.linspace(-np.pi/2, np.pi/2, 400)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle('Aproximările Taylor și Pade ale funcției sinus (oY log)')
    fig.subplots_adjust(wspace=0.3, hspace=0.4)

    axs[0,0].plot(domain, np.sin(domain), 'r', label=r'$sin(\alpha)$')
    axs[0,0].plot(domain, domain, 'g', label='Aprox. Taylor')

    axs[0, 0].set_title('Aproximarea Taylor a funcției sinus')
    axs[0, 0].set_xlabel(r'$Alfa (\alpha)$')
    axs[0, 0].set_ylabel('Valori')
    axs[0, 0].legend()

    axs[0, 1].plot(domain, np.sin(domain), 'r', label=r'$sin(\alpha)$')
    axs[0, 1].plot(domain, (domain - 7/60 * domain**3) / (1 + domain**2 / 20), 'g', label='Aprox. Pade')

    axs[0, 1].set_title('Aproximarea Pade a funcției sinus')
    axs[0, 1].set_xlabel(r'$Alfa (\alpha)$')
    axs[0, 1].set_ylabel('Valori')
    axs[0, 1].legend()

    axs[1, 0].plot(domain, np.abs(np.sin(domain) - domain), 'b')

    axs[1, 0].set_title('Eroarea absolută (aprox. Taylor)')
    axs[1, 0].set_xlabel(r'$Alfa (\alpha)$')
    axs[1, 0].set_ylabel('Eroare')

    axs[1, 1].plot(domain, np.abs(np.sin(domain) - (domain - 7/60 * domain**3) / (1 + domain**2 / 20)), 'b')

    axs[1, 1].set_title('Eroarea absolută (aprox. Pade)')
    axs[1, 1].set_xlabel(r'$Alfa (\alpha)$')
    axs[1, 1].set_ylabel('Eroare')

    for ax in axs.flat:
        ax.grid(linestyle='--')
        ax.set_yscale('log')

    plt.show()


def main():
    ex1()
    ex2()
    ex3()
    ex4()
    ex5()
    ex6()
    ex7()
    ex8a()
    ex8b()

if __name__ == '__main__':
    main()