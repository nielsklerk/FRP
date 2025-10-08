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
    # Normalize PSF
    psf = psf / psf.sum()

    # Pad both image and PSF to avoid wrap-around artifacts
    pad_shape = [s * 2 for s in img.shape]
    psf_padded = np.zeros(pad_shape)
    img_padded = np.zeros(pad_shape)

    psf_center = tuple(s // 2 for s in psf.shape)
    img_center = tuple(s // 2 for s in img.shape)

    # place psf in center of padded array
    psf_padded[
        pad_shape[0]//2 - psf_center[0] : pad_shape[0]//2 + psf.shape[0] - psf_center[0],
        pad_shape[1]//2 - psf_center[1] : pad_shape[1]//2 + psf.shape[1] - psf_center[1]
    ] = psf

    img_padded[:img.shape[0], :img.shape[1]] = img

    # Compute FFTs
    psf_fft = fft2(ifftshift(psf_padded))
    img_fft = fft2(img_padded)

    # Wiener filter
    result_fft = (np.conj(psf_fft) * img_fft) / (psf_fft * np.conj(psf_fft) + K)
    result = np.real(ifft2(result_fft))

    # Crop back to original size
    result_cropped = result[:img.shape[0], :img.shape[1]]
    return result_cropped