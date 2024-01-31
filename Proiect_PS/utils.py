import numpy as np
from numpy.typing import NDArray
from scipy.fft import dctn, idctn


def splitIntoBlocks(arr: NDArray, blockSize: int):
    """
    Partiționează o matrice de n dimensiuni în blocuri de (blockSize x blockSize) dimensiuni.

        Parametri:
            arr (NDArray): Matricea de partiționat
            blockSize (int): Dimensiunea unui bloc

        Returnează:
            blocks (NDArray): Vector cu blocurile rezultate
    """
    nrBlocksX = np.int32(arr.shape[0] / blockSize)
    partitionsX = [(i * blockSize, (i + 1) * blockSize) for i in range(nrBlocksX)]
    nrBlocksY = np.int32(arr.shape[1] / blockSize)
    partitionsY = [(i * blockSize, (i + 1) * blockSize) for i in range(nrBlocksY)]

    positions = []
    for i in partitionsX:
        for j in partitionsY:
            positions.append((i, j))

    blocks = []
    for i in range(len(positions)):
        blocks.append(arr[positions[i][0][0]:positions[i][0][1], positions[i][1][0]:positions[i][1][1]])

    return np.array(blocks)


def combineBlocks(subBlocks: NDArray, x: int, y: int):
    """
    Combină segmente înapoi în blocuri.

        Parametri:
            subBlocks (NDArray): Segmentele de combinat
            x (int): Nr. de segmente de pus pe o linie
            y (int): Nr. de segmente de pus pe o coloană
        Returnează
            block (NDArray): Blocul aferent combinării
    """
    block = np.concatenate(subBlocks[0:y], axis=1)
    for i in range(1, x):
        row = np.concatenate(subBlocks[(i * y):(i+1) * y], axis=1)
        block = np.concatenate((block, row), axis=0)
    
    return block


def convertColorspace(img: NDArray, toWhat: str, direction = True):
    """
    Convertește o imagine RGB într-un alt spațiu de culori, și invers.

        Parametri:
            img (NDArray): Matricea imaginii
            toWhat (str): Spațiul de culori destinație
            direction (bool): implicit True pentru RGB->spațiu, False pentru spațiu->RGB

        Returnează:
            dest (NDArray): Matricea imaginii convertite
    """
    h, w = img.shape[0], img.shape[1]
    dest = np.empty((h, w, 3))
    YCbCrOffset = np.array([0, 128, 128])

    RGBtoYUV = np.array([
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001]
    ])
    RGBtoYCbCr = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.500],
        [0.500, -0.418688, -0.081312]
    ])
    YUVtoRGB = np.array([
        [1, 0, 1.13983],
        [1, -0.39465, -0.58060],
        [1, 2.03211, 0]
    ])
    YCbCrtoRGB = np.array([
        [1.0, 0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0]
    ])

    for i in range(h):
        for j in range(w):
            src = np.array([
                img[i, j, 0],
                img[i, j, 1],
                img[i, j, 2]
            ])

            if toWhat == "YUV":
                if direction:
                    dest[i, j] = np.dot(src, RGBtoYUV.T)
                else:
                    dest[i, j] = np.dot(src, YUVtoRGB.T)
            if toWhat == "YCbCr":
                if direction:
                    dest[i, j] = np.dot(src, RGBtoYCbCr.T) + YCbCrOffset
                else:
                    dest[i, j] = np.dot(src - YCbCrOffset, YCbCrtoRGB.T)

    if direction:
        return dest
    else:
        return dest.astype(np.int16)


def jpegify(x: NDArray):
    """
    Compresia JPEG pentru o anumită matrice.

        Parametri:
            x (NDArray): Matricea peste care se aplică compresia

        Returnează:
            xJPEG (NDArray): Rezultatul compresiei
    """
    QPj = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 28, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ]) # Matricea de cuantizare JPEG (4x4)

    if x.shape[1] == 4:
        QPj = QPj[:4, :4]

    y = dctn(x)
    yJPEG = QPj * np.round(y / QPj)
    xJPEG = idctn(yJPEG)

    return np.array(xJPEG)


def padImage(img: NDArray, padSize: int):
    """
    Adaugă pixeli în lungimea/lățimea imaginii dacă acestea nu sunt divizibile cu padSize.

        Parametri:
            img (NDArray): Imaginea de modificat
            padSize (int): Până la cât trebuie adăugați pixeli

        Returnează:
            paddedImg (NDArray): Imaginea modificată
    """
    h, w = img.shape[0], img.shape[1]

    padHeight = -h % padSize
    padWidth = -w % padSize

    paddedImg = np.pad(img, ((0, padHeight), (0, padWidth), (0, 0)), mode='constant')

    return np.array(paddedImg)