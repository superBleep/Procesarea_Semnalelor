import numpy as np
from numpy.typing import NDArray


def MSE(A: NDArray, B: NDArray):
    """
    Determină eroarea pătrată medie dintre două matrici.

        Parametri:
            A (NDArray): Prima matrice
            B (NDArray): A doua matrice

        Returnează:
            err (float32): Eroarea pătrată medie
    """
    s = np.float64(0)
    nx, ny = A.shape[0], A.shape[1]

    for i in range(nx):
        for j in range(ny):
            s += (A[i, j] - B[i, j]) ** 2

    err = (s / (nx * ny))

    return err


def SSIM(A: NDArray, B: NDArray):
    """
    Calculează indicele de similaritate structurala (SSIM - Structural Similarity Index) pentru două matrici A și B.

        Parametri:
            A (NDArray): Prima matrice
            B (NDArray): A doua matrice

        Returnează:
            ssim (float32): Indicele de similaritate structurală
    """
    muA, muB = np.mean(A), np.mean(B)
    varA, varB = np.var(A), np.var(B)
    varAB = np.cov(A.flatten(), B.flatten())[0, 1]
    k1, k2 = 0.01, 0.03
    L = 255
    c1, c2 = (k1 * L)**2, (k2 * L)**2

    ssim = ((2 * muA * muB + c1) * (2 * varAB + c2)) / ((muA**2 + muB**2 + c1) * (varA + varB + c2))
    ssim = np.float32(ssim)

    return ssim


def PSNR(A: NDArray, B: NDArray):
    """
    Calculează rata dintre puterea maximă și puterea zgomotului (PSNR - Peak Signal-to-Noise Ratio) dintre două matrici A și B.

        Parametri:
            A (NDArray): Prima matrice
            B (NDArray): A doua matrice

        Returnează:
            psnr (float32): Valoarea ratei
    """
    mse = MSE(A, B)
    max = 255

    psnr = np.float32(20 * np.log10(max) - 10 * np.log10(mse))

    return psnr