import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice as sound

def ex1():
    time = np.linspace(0, 3, 1000)

    fig, axs = plt.subplots(2)
    fig.suptitle('Semnalele sin și cos (t=0:3)')
    axs[0].plot(time, 2 * np.sin(2 * np.pi * time), 'r')
    axs[1].plot(time, 2 * np.cos(2 * np.pi * time - np.pi / 2), 'g')

    for ax in axs.flat:
        ax.set_xlabel('Timp (t)')
        ax.set_ylabel('Amplitudine (A)')
        ax.grid(linestyle='--')

    plt.show()

def ex2():
    def noisify(domain, phi, snr, sample_nr):
        x = np.sin(2 * np.pi * domain + phi)
        z = np.random.normal(0, 1, sample_nr)
        gamma = np.sqrt(np.linalg.norm(x)**2 / (snr * np.linalg.norm(z)**2))

        return x + gamma * z

    sample_nr = 1000
    domain = np.linspace(0, 2*np.pi, sample_nr)

    plt.figure(1)

    plt.plot(domain, noisify(domain, 0, 0.1, sample_nr), 'r', label=r"$\phi=0$")
    plt.plot(domain, noisify(domain, np.pi/2, 1, sample_nr), 'g', label=r"$\phi=\frac{\pi}{2}$")
    plt.plot(domain, noisify(domain, np.pi, 10, sample_nr), 'b', label=r"$\phi=\pi$")
    plt.plot(domain, noisify(domain, 3 * np.pi / 2, 100, sample_nr), 'purple', label=r"$\phi=\frac{3\pi}{2}$")

    plt.grid(linestyle='--')
    plt.title('Semnale cu zgomot')
    plt.xlabel('TImp (t)')
    plt.ylabel('Amplitudine (A)')
    plt.legend()
    plt.show()

def ex3():
    rate = 44100
    time = np.linspace(0, 3, 3*rate)

    s1 = np.sin(2 * np.pi * 400 * time)
    s2 = np.sin(2 * np.pi * 800 * time)
    s3 = 2 * (time * 240 - np.floor(1 / 2 + time * 240))
    s4 = np.sign(np.sin(2 * np.pi * 300 * time))

    scipy.io.wavfile.write('s1.wav', 44100, s1)
    scipy.io.wavfile.write('s2.wav', 44100, s2)
    scipy.io.wavfile.write('s3.wav', 44100, s3)
    scipy.io.wavfile.write('s4.wav', 44100, s4)

def main():
    # ex1()
    # ex2()
    ex3()

if __name__ == '__main__':
    main()