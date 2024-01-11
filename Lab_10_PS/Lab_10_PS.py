'''
    Lab 10 Procesarea Semnalelor
    Florian Luca-Paul, CTI, Grupa 461
'''
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def ex3():
    # Subpunctul a
    x = np.genfromtxt('co2_daily_mlo.csv', delimiter=',')
    x = x[:, [0,1,2,4]]

    y = []
    vals = []
    for i in range(len(x)):
        if i < len(x) - 1:
            if x[i+1, 2] > x[i, 2]:
                vals.append(x[i, 3])
            else:
                y.append([int(x[i, 0]), int(x[i, 1]), np.round(np.average(vals), 2)])
                vals = []

    y = np.array(y)
    print(y[-1])

if __name__ == '__main__':
    ex3()