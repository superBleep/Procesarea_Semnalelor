from scipy.ndimage import zoom
from numpy.typing import NDArray
from utils import *


def jpeg(img: NDArray):
    """
    Convertește o imagine RGB, transmisa ca o matrice (N x M x 3), într-o imagine în formatul JPEG.

        Parametri:
            img (NDArray): Matricea imaginii originale
        
        Returnează
            jpeg (NDArray): Matricea imaginii convertite
    """
    h, w = img.shape[0], img.shape[1] # Dimensiunile imaginii
    jpeg = padImage(img, 8) # Aplică padding dacă este cazul

    blocks = splitIntoBlocks(jpeg, 8) # Imparte imaginea in blocui de 8x8
    blocksJPEG = []

    for block in blocks:
        blockYCbCr = convertColorspace(block, "YCbCr") # Aplică o conversie RGB -> YCbCr

        # ---- Componenta Y ----
        blockY = blockYCbCr[:, :, 0] # Componenta Y din blocul YCbCr
        blockYjpeg = jpegify(blockY) # Aplică DCT, cuantizare si IDCT peste bloc

        # ---- Componentele Cb și Cr ----
        blockCb, blockCr = blockYCbCr[:, :, 1], blockYCbCr[:, :, 2] # Componentele Cb si Cr din blocul YCbCr
        blockCb, blockCr = blockCb[::2, ::2], blockCr[::2, ::2] # Componentele subesantionate la rata 4:2:0
        blockCbjpeg, blockCrjpeg = jpegify(blockCb), jpegify(blockCr) # Aplică DCT, cuantizare si IDCT peste blocuri
        upscaledCbjpeg, upscaledCrjpeg = zoom(blockCbjpeg, 2), zoom(blockCrjpeg, 2)

        # ---- Combinarea componentelor ----
        blockYCbCrjpeg = np.empty(blockYCbCr.shape)
        blockYCbCrjpeg[:, :, 0] = blockYjpeg
        blockYCbCrjpeg[:, :, 1] = upscaledCbjpeg
        blockYCbCrjpeg[:, :, 2] = upscaledCrjpeg

        # ---- Conversia înapoi în RGB ----
        blockRGBjpeg = convertColorspace(blockYCbCrjpeg, "YCbCr", False)

        blocksJPEG.append(blockRGBjpeg)

    blocksJPEG = np.array(blocksJPEG)

    x = np.int16(jpeg.shape[0] / 8)
    y = np.int16(jpeg.shape[1] / 8)

    jpeg = combineBlocks(blocksJPEG, x, y)
    jpeg = np.clip(jpeg, 0, 255).astype(np.uint8)

    jpeg = jpeg[:h, :w, :] # Scoate padding-ul

    return jpeg
