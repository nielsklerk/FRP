import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def gaussian_weight(height, width, xc=0, yc=0, a=0, b=0):
    """
    Compute a Gaussian weight map centered at (xc, yc).
    """
    y, x = np.indices((height, width), dtype=float)
    weight = np.exp(-0.5 * (((x - xc) / a) ** 2 + ((y - yc) / b) ** 2))
    return weight / weight.sum()

def wiener_deconvolution(img, psf, K):
    psf_fft = fft2(ifftshift(psf))
    img_fft = fft2(img)

    # Wiener filter
    result_fft = (np.conj(psf_fft) * img_fft) / (psf_fft * np.conj(psf_fft) + K)
    result = np.real(ifft2(result_fft))

    # Crop back to original size
    # result_cropped = result[:img.shape[0], :img.shape[1]]
    return result