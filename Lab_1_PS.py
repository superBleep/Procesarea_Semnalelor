'''
    Lab 1 Procesarea Semnalelor
    Florian Luca-Paul, CTI, Grupa 461
'''
import numpy as np
import matplotlib.pyplot as plt

def ex1():
    # Pct. (a)
    time = np.linspace(0, 0.03, 1000)

    # Pct. (b)
    fig, axs = plt.subplots(3)
    fig.suptitle('Semnale continue')
    axs[0].plot(time, np.cos(520 * np.pi * time + np.pi / 3), 'r')
    axs[1].plot(time, np.cos(280 * np.pi * time - np.pi / 3), 'g')
    axs[2].plot(time, np.cos(120 * np.pi * time + np.pi / 3), 'b')

    for ax in axs.flat:
        ax.grid(linestyle='--')
        ax.set_xlabel('Timp (t)')
        ax.set_ylabel('Amplitudine (A)')

    # Pct. (c)
    sample_nr = np.int8(0.03 * 200)
    samples = np.linspace(0, 0.03, sample_nr)

    fig2, axs2 = plt.subplots(3)
    fig2.suptitle('Semnale eșantionate ({} eșantioane, 200 Hz)'.format(sample_nr))

    axs2[0].stem(samples, np.cos(520 * np.pi * samples + np.pi / 3), 'r')
    axs2[1].stem(samples, np.cos(280 * np.pi * samples - np.pi / 3), 'g')
    axs2[2].stem(samples, np.cos(120 * np.pi * samples + np.pi / 3), 'b')

    axs2[0].plot(time, np.cos(520 * np.pi * time + np.pi / 3), 'r')
    axs2[1].plot(time, np.cos(280 * np.pi * time - np.pi / 3), 'g')
    axs2[2].plot(time, np.cos(120 * np.pi * time + np.pi / 3), 'b')

    for ax in axs2.flat:
        ax.grid(linestyle='--')
        ax.set_xlabel("Esantion")
        ax.set_ylabel("Amplitudine (A)")

    plt.show()

def ex2():
    # Pct. (a)
    samples = np.linspace(0, 0.03, 1600)

    plt.figure(1)
    plt.stem(samples, np.sin(2 * np.pi * 400 * samples), 'r')
    plt.title('Semnal sinusoidal (f = 400 Hz, t = 0:0.03, 1600 de eșantioane)')
    plt.xlabel('Esantion')
    plt.ylabel('Amplitudine (A)')
    plt.grid(linestyle='--')

    # Pct. (b)
    time = np.linspace(0, 3, 1000)

    plt.figure(2)
    plt.plot(time, np.sin(2 * np.pi * 800 * time), 'r')
    plt.title('Semnal sinusoidal (f = 800 Hz, t = 0:3)')
    plt.xlabel('Timp (t)')
    plt.ylabel('Amplitudine (A)')
    plt.grid(linestyle='--')

    # Pct. (c)
    time = np.linspace(0, 0.02, 1000)

    plt.figure(3)
    plt.plot(time, 2 * (time * 240 - np.floor(1 / 2 + time * 240)), color='purple')
    plt.title('Semnal sawtooth (f = 240 Hz, t = 0:0.02)')
    plt.xlabel('Timp (t)')
    plt.ylabel('Amplitudine (A)')
    plt.grid(linestyle='--')

    # Pct. (d)
    time = np.linspace(0, 0.02, 1000)

    plt.figure(4)
    plt.plot(time, np.sign(np.sin(2 * np.pi * 300 * time)), 'g')
    plt.title('Semnal square (f = 300 Hz, t = 0:0.02)')
    plt.xlabel('Timp (t)')
    plt.ylabel('Amplitudine (A)')
    plt.grid(linestyle='--')

    # Pct. (e)
    arr = np.random.rand(128, 128)

    plt.figure(5)
    plt.imshow(arr)
    plt.xlabel('Semnal 1')
    plt.ylabel('Semnal 2')
    plt.title('Semnal 2D aleator (128x128)')

    # Pct. (f)
    arr = np.ones((128, 128))
    for i in range(128):
        for j in range(128):
            arr[i][j] = (i + 1) % (j + 1)

    plt.figure(6)
    plt.imshow(arr)
    plt.xlabel('Semnal 1')
    plt.ylabel('Semnal 2')
    plt.title('Semnal 2D (128x128)')

    plt.show()

'''
    Exercitiul 3

    (a) T = 1 / 2000 = 0,0005 s
    (b) esantioane = 3600 / 0,0005 = 7.200.000
        biti = 7.200.000 * 4 = 28.800.000 b
        bytes = 28.800.000 / 8 = 3.600.000 = 3.6 Mb
    
'''

def main():
    ex1()
    ex2()

if __name__ == '__main__':
    main()