from scipy.ndimage import zoom
import numpy as np
from numpy.typing import NDArray
from utils import *
from stats import *


def gatherPixels(subBlocks: NDArray):
    """
    Generează, pentru o serie de blocuri 4x4 de pixeli, vectorii de pixeli necesari
    aplicării predictorilor.

        Parametri
            subBlocks (NDArray): Setul de blocuri de 4x4 pixeli

        Returnează
            A, L, C (tuple[NDArray, NDArray, NDArray]): Vectorii de pixeli
    """
    A, L, C = [], [], []

    for i in range(16):
        match i:
            case 0:
                A.append(subBlocks[4, 0, :])
                L.append(subBlocks[1, :, 0])
                C.append(subBlocks[5, 0, 0])
            case 1, 2, 3:
                A.append(subBlocks[i + 4, 0, :])
                L.append(subBlocks[i - 1, :, -1])
                C.append(subBlocks[i + 3, 0, -1])
            case 4, 8, 12:
                A.append(subBlocks[i - 4, -1, :])
                L.append(subBlocks[i + 1, :, 0])
                C.append(subBlocks[i - 3, -1, 0])
            case _:
                A.append(subBlocks[i - 4, -1, :])
                L.append(subBlocks[i - 1, :, -1])
                C.append(subBlocks[i - 5, -1, -1])

    A, L, C = np.array(A), np.array(L), np.array(C)

    return A, L, C


def predictValues(subBlocks: NDArray):
    """
    Alege și aplică cel mai bun mod de predicție peste pixelii unui set de blocuri.

        Parametri:
            subBlocks (NDArray): Blocurile peste care se aplică predicția

        Returnează:
            predicted (NDArray): Blocurile "prezise"
    """
    A, L, C = gatherPixels(subBlocks)
    predicted = []

    for i, subBlock in enumerate(subBlocks):
        predictors = [
            np.array([np.repeat(p, 4) for p in L[i]]),                                          # H_PRED
            np.array([np.repeat(p, 4) for p in A[i]]).T,                                        # V_PRED
            np.array([np.repeat((np.average(A) + np.average(L)) / 2, 4) for _ in range(4)]),    # DC_PRED
            np.array([[L[i, x] + A[i, y] - C[i] for y in range(4)] for x in range(4)])          # TM_PRED
        ]

        bestMatch = predictors[0]
        comp = MSE(subBlock, predictors[0])
        for i in range(1, len(predictors)):
            newComp = MSE(subBlock, predictors[i])

            if newComp < comp:
                bestMatch = predictors[i]
                comp = newComp

        predicted.append(bestMatch)

    return np.array(predicted)


def webp(img: NDArray):
    """
    Convertește o imagine RGB, transmisa ca o matrice (N x M x 3), într-o imagine în formatul WebP (compresie lossy).

        Parametri:
            img (NDArray): Matricea imaginii originale
        
        Returnează
            webp (NDArray): Matricea imaginii convertite
    """
    h, w = img.shape[0], img.shape[1] # Dimensiunile imaginii
    webp = padImage(img, 16) # Aplică padding dacă este cazul

    macroBlocks = splitIntoBlocks(webp, 16) # Imparte imaginea in blocuri de 16x16
    macroBlocksWebP = []

    for block in macroBlocks:
        blockYUV = convertColorspace(block, "YUV") # Aplică o conversie RGB -> YUV

        # ---- Componenta Y ----
        blockY = blockYUV[:, :, 0] # Componenta Y (16x16) din blocul YUV
        subBlocksY = splitIntoBlocks(blockY, 4) # Imparte blocul Y 16x16 in segmente 4x4
        predSubBlocksY = predictValues(subBlocksY) # Segmentele Y "prezise"
        subBlocksYwebp = jpegify(predSubBlocksY) # Aplica DCT, cuantizare si IDCT pt. fiecare segment 4x4 "prezis"
        blockYwebp = combineBlocks(subBlocksYwebp, 4, 4)

        # ---- Componentele U și V ----
        blockU, blockV = blockYUV[:, :, 1], blockYUV[:, :, 2] # Componentele 16x16 din blocul YUV
        blockU, blockV = blockU[::2, ::2], blockV[::2, ::2] # Componentele 8x8, subesantionate la rata 4:2:0
        subBlocksU, subBlocksV = splitIntoBlocks(blockU, 4), splitIntoBlocks(blockV, 4) # Imparte componentele in segmente 4x4
        subBlocksUjpeg, subBlockVjpeg = jpegify(subBlocksU), jpegify(subBlocksV) # Aplica DCT, cuantizare si IDCT pt. fiecare segment 4x4
        blockUjpeg, blockVjpeg = combineBlocks(subBlocksUjpeg, 2, 2), combineBlocks(subBlockVjpeg, 2, 2)
        upscaledUjpeg, upscaledVjpeg = zoom(blockUjpeg, 2, order=1), zoom(blockVjpeg, 2, order=1)

        # ---- Combinarea componentelor ----
        blockYUVwebp = np.empty(blockYUV.shape)
        blockYUVwebp[:, :, 0] = blockYwebp
        blockYUVwebp[:, :, 1] = upscaledUjpeg
        blockYUVwebp[:, :, 2] = upscaledVjpeg

        # ---- Conversia înapoi în RGB ----
        blockRGBwebp = convertColorspace(blockYUVwebp, "YUV", False)

        macroBlocksWebP.append(blockRGBwebp)

    macroBlocksWebP = np.array(macroBlocksWebP)

    x = np.int16(webp.shape[0] / 16)
    y = np.int16(webp.shape[1] / 16)
    
    webp = combineBlocks(macroBlocksWebP, x, y)
    webp = np.clip(webp, 0, 255).astype(np.uint8)

    webp = webp[:h, :w, :] # Scoate padding-ul

    return webp