import numpy as np
import skimage.data as dataset
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from jpegalg import jpeg
from webpalg import webp
from stats import *


def rgbStat(A: NDArray, B: NDArray, statFunc):
    """
    Aplică o funcție de statistică peste cele 3 canale ale unei imagini RGB și face media dintre rezultate.

        Parametri:
            A (NDArray): Matricea aferentă primei imagini
            B (NDArray): Matricea aferentă celei de-a doua imagini
            statFunc (function): Funcția de aplicat (MSE, SSIM, PSNR)

        Returnează:
            statRGB: Media rezultatelor calculate
    """
    statRGB = np.float32(0)

    for i in range(3):
        statRGB += statFunc(A[:, :, i], B[:, :, i])

    statRGB = statRGB / 3

    return statRGB


def main():
    images = [
        dataset.coffee(),
        dataset.rocket(),
        dataset.cat(),
        dataset.stereo_motorcycle()[0]
    ]

    imagesWebP, imagesJPEG = [], []
    statsWebP, statsJPEG = [], []

    for i, img in enumerate(images):
        imgWebP = webp(img)
        imagesWebP.append(imgWebP)
        print("Imaginea {} convertită în WebP".format(i + 1))

        stats = []
        stats.append(rgbStat(img, imgWebP, SSIM))
        stats.append(rgbStat(img, imgWebP, MSE))
        stats.append(rgbStat(img, imgWebP, PSNR))
        statsWebP.append(stats)
        print("Calculat statistici WebP pt. imaginea {}".format(i + 1))

        imgJPEG = jpeg(img)
        imagesJPEG.append(img)
        print("Imaginea {} convertită în JPEG".format(i + 1))

        stats = []
        stats.append(rgbStat(img, imgJPEG, SSIM))
        stats.append(rgbStat(img, imgJPEG, MSE))
        stats.append(rgbStat(img, imgJPEG, PSNR))
        statsJPEG.append(stats)
        print("Calculat statistici JPEG pt. imaginea {}".format(i + 1))

        plt.imsave("pics/image{}.webp".format(i+1), imgWebP)
        plt.imsave("pics/image{}.jpeg".format(i+1), imgJPEG)

    statsWebP, statsJPEG = np.array(statsWebP), np.array(statsJPEG)

    x = ["Coffe", "Rocket", "Cat", "Moto"]
    xAxis = np.arange(len(x))
    w = 0.2

    plt.figure(1)

    plt.bar(xAxis - w, statsWebP[:, 0], 2*w, label="WebP", color="purple")
    plt.bar(xAxis + w, statsJPEG[:, 0], 2*w, label="JPEG", color="orange")

    plt.xticks(xAxis, x)
    plt.xlabel("Imagini")
    plt.ylabel("SSIM")
    plt.title("Indexul SSIM pentru patru imagini din setul skimage.data, comprimate cu WebP/JPEG")
    plt.legend()

    plt.figure(2)

    plt.bar(xAxis - w, statsWebP[:, 1], 2*w, label="WebP", color="purple")
    plt.bar(xAxis + w, statsJPEG[:, 1], 2*w, label="JPEG", color="orange")

    plt.xticks(xAxis, x)
    plt.xlabel("Imagini")
    plt.ylabel("MSE")
    plt.title("MSE pentru patru imagini din setul skimage.data, comprimate cu WebP/JPEG")
    plt.legend()

    plt.show()

    plt.figure(3)

    plt.bar(xAxis - w, statsWebP[:, 2], 2*w, label="WebP", color="purple")
    plt.bar(xAxis + w, statsJPEG[:, 2], 2*w, label="JPEG", color="orange")

    plt.xticks(xAxis, x)
    plt.xlabel("Imagini")
    plt.ylabel("PSNR")
    plt.title("PSNR pentru patru imagini din setul skimage.data, comprimate cu WebP/JPEG")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()