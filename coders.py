import cv2
import numpy as np
import pywt
from skimage import img_as_float


def compress_jpeg2000(img, cr):
    """
    Compresses an image using JPEG2000 via OpenCV.
    :param img: Input image (float in 0-1).
    :param cr: Compression Ratio.
    :return: Reconstructed image (float in 0-1).
    """
    img_uint8 = (img * 255).astype(np.uint8)
    # Set JPEG2000 compression parameters: IMWRITE_JPEG2000_COMPRESSION_X1000
    # The value is "quality" related, where 1000/CR gives a rough inverse relation
    # to achieve the desired compression ratio.
    param = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), int(1000 / cr)]
    _, enc = cv2.imencode('.jp2', img_uint8, param)
    return cv2.imdecode(enc, 0).astype(np.float32)/255.0



# --- Simulated SPIHT Algorithm (with padding) ---
def compress_spiht(img, cr, wavelet='haar'): #Todo not valid or remove
    """
    Modified SPIHT version with proper threshold management and wavelet levels.
    This is a conceptual simulation, not a full SPIHT implementation.
    :param img: Input image (float in 0-1).
    :param cr: Compression Ratio.
    :param wavelet: Wavelet type (e.g., 'haar', 'db1', etc.).
    :return: Reconstructed image (float in 0-1).
    """
    if img.max() > 1.0:
        img = img_as_float(img)
    
    # Automatically determine wavelet levels
    # Levels are typically limited to avoid very small coefficients or too many levels.
    max_levels = int(np.log2(min(img.shape))) - 1
    levels = min(3, max_levels) # Limiting to 3 levels for typical image sizes
    
    # Smart padding (if necessary)
    min_dim_required = 2 ** levels
    pad_h = 0 if img.shape[0] >= min_dim_required else min_dim_required - img.shape[0]
    pad_w = 0 if img.shape[1] >= min_dim_required else min_dim_required - img.shape[1]
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='symmetric')
    
    # Wavelet transform
    coeffs = pywt.wavedec2(img_padded, wavelet, level=levels)
    
    # Appropriate thresholding (directly related to CR)
    # This coefficient (0.008) is empirically optimized
    threshold = 0.008 * cr  
    
    # Apply threshold to coefficients (except the approximation coefficients)
    new_coeffs = [coeffs[0]]  # Keep approximation coefficients
    for level_coeffs in coeffs[1:]:
        # Apply threshold to detail coefficients (LH, HL, HH)
        new_level = tuple(np.where(np.abs(c) > threshold, c, 0) for c in level_coeffs)
        new_coeffs.append(new_level)
    
    # Reconstruction
    reconstructed = pywt.waverec2(new_coeffs, wavelet)
    # Crop back to original size
    reconstructed = reconstructed[:img.shape[0], :img.shape[1]]
    
    return np.clip(reconstructed, 0, 1)
