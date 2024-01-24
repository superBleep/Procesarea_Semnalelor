'''
    Lab 10 Procesarea Semnalelor
    Florian Luca-Paul, CTI, Grupa 461
'''
import numpy as np
import matplotlib.pyplot as plt

def ex1():
    n = 200

    med = 10
    var = 3
    x1 = np.array([np.random.normal(med, np.sqrt(var)) for _ in range(n)])

    plt.figure(1)

    plt.plot(np.arange(n), x1, 'r')

    plt.title(r'Distribuție Gaussiană 1D ($\mu = {}$, $\sigma^2 = {}$, n = {})'.format(med, var, n))
    plt.xlabel('Indice eșantion')
    plt.ylabel('Valoare')
    plt.grid(linestyle='--')

    Mu = np.array([0, 0])
    Sigma = np.array([
        [1, 3/5],
        [3/5, 2]
    ])

    U, s, _ = np.linalg.svd(Sigma) # Desc. val. proprii
    L = np.diag(s) # Matricea diagonala lambda

    x2 = np.array([U @ np.sqrt(L) @ np.random.multivariate_normal(np.array([0, 0]), np.eye(np.shape(Sigma)[0])) + Mu for _ in range(n)])

    plt.figure(2)

    plt.scatter(x2[:, 0], x2[:, 1], color='g')

    plt.title('Distribuție Gaussiană 2D (n = {})'.format(n))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(linestyle='--')

    plt.show()


def ex2():
    def linear(n, figNr):
        x = [np.random.normal() for _ in range(n)]
        y = [np.random.normal() for _ in range(n)]

        med = np.zeros(n)   
        C = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                C[i, j] = x[i] * y[j]

        z = [np.random.multivariate_normal(med, C) for _ in range(n)]

        plt.figure(figNr)

        plt.plot(x, z)

        plt.title('PG cu kernel liniar')
        plt.xlim(-1, 1)
        plt.grid(linestyle='--')

    linear(2, 1)
    plt.show()


def ex3():
    def covariance(x, y):
        # Aici kernel-ul simetric
        pass

    # Subpunctul a
    x = np.genfromtxt('co2_daily_mlo.csv', delimiter=',')
    x = x[:, [0,1,2,4]]

    y = []
    ref = x[0, 1]
    vals = [x[0, 3]]
    for i in range(1, len(x)):
        if x[i, 1] == ref:
            vals.append(x[i, 3])

            if i == len(x)-1:
                y.append([int(x[i, 0]), ref, np.round(np.average(vals), 2)])
        else:
            y.append([int(x[i-1, 0]), ref, np.round(np.average(vals), 2)])
            ref = x[i, 1]
            vals = [x[i, 3]]

    y = np.array(y)
    domain = np.arange(len(y))

    plt.figure(1)

    plt.plot(domain, y[:, 2], 'g')

    plt.title("Calitatea lunară a aerului din setul MLO")
    plt.grid(linestyle='--')
    plt.xlabel("Index lună")
    plt.ylabel(r"Conținut de CO$_2$")

    # Subpunctul b
    A = np.vstack([domain, np.ones(len(domain))]).T
    m, c = np.linalg.lstsq(A, y[:, 2], rcond=None)[0]
    trend = m * domain + c
    y2 = y[:, 2] - trend

    plt.figure(2)

    plt.plot(domain, y2, 'r')

    plt.title("Seria de timp aferentă setului MLO, cu trendul eliminat")
    plt.grid(linestyle='--')
    plt.xlabel("Index lună")

    plt.show()

    # Subpunctul c
    setA = y2[-12:] # Ultimele 12 luni
    setB = y2[:-12] # Restul de luni

    med = np.array([np.mean(setA), np.mean(setB)])
    C = np.array([
        [covariance(setA, setA), covariance(setA, setB)],
        [covariance(setB, setA), covariance(setB, setB)]
    ])

    #m = med[0] + C[0, 1] * np.linalg.inv(C[1, 1]) * (y - med[1])
    #D = C[0, 0] - C[0, 1] * np.linalg.inv(C[1, 1]) * np.linalg.inv(C[1, 0])


if __name__ == '__main__':
    #ex1()
    #ex2()
    ex3()