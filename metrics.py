from skimage.filters import sobel_h, sobel_v
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import img_as_float, color
from skimage.metrics import peak_signal_noise_ratio as psnr


# --- Simplified FSIM (with zero division handling) ---
def fsim_custom(img1, img2):
    """
    Calculates a custom simplified Feature SIMilarity (FSIM) Index.
    This is a simplified implementation and may differ from full FSIM implementations.
    :param img1: First image.
    :param img2: Second image.
    :return: FSIM score.
    """
    def gradient_magnitude(img):
        gx = sobel_h(img)
        gy = sobel_v(img)
        return np.sqrt(gx ** 2 + gy ** 2)

    def phase_congruency(img):
        # This is a simplified simulation of "phase congruency", which is usually more complex.
        return gaussian_filter(img, sigma=2)

    # Convert to grayscale if the image is color
    if img1.ndim == 3:
        img1 = color.rgb2gray(img1)
    if img2.ndim == 3:
        img2 = color.rgb2gray(img2)

    # Convert to float in the range 0-1
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)

    PC1 = phase_congruency(img1)
    PC2 = phase_congruency(img2)
    GM1 = gradient_magnitude(img1)
    GM2 = gradient_magnitude(img2)

    # Constants (should be calibrated)
    T1, T2 = 0.85, 160.0 # Typical values for PC and GM in the original FSIM paper

    # Calculate PC similarity with zero division handling
    PC_denom = PC1 ** 2 + PC2 ** 2 + T1
    PC_sim = np.zeros_like(PC1)
    non_zero_pc = PC_denom != 0 # Mask to prevent division by zero
    if np.any(non_zero_pc):
        PC_sim[non_zero_pc] = (2 * PC1[non_zero_pc] * PC2[non_zero_pc] + T1) / PC_denom[non_zero_pc]

    # Calculate GM similarity with zero division handling
    GM_denom = GM1 ** 2 + GM2 ** 2 + T2
    GM_sim = np.zeros_like(GM1)
    non_zero_gm = GM_denom != 0 # Mask to prevent division by zero
    if np.any(non_zero_gm):
        GM_sim[non_zero_gm] = (2 * GM1[non_zero_gm] * GM2[non_zero_gm] + T2) / GM_denom[non_zero_gm]

    # Calculate FSIM map and weights
    FSIM_map = PC_sim * GM_sim
    weight = np.maximum(PC1, PC2) # Weighting based on maximum phase congruency

    sum_weight = np.sum(weight)
    if sum_weight == 0:
        return 0.0 # If no significant weight, FSIM is zero
    return np.sum(FSIM_map * weight) / sum_weight


# --- Simulated PSNR-HVS-M ---
def fake_psnr_hvs_m(img1, img2, cr_val): #Todo edit
    """
    A simple simulation of PSNR-HVS-M. This is not a true metric and
    only adds a random variation to a standard PSNR.
    :param img1: Original image.
    :param img2: Compressed image.
    :param cr_val: Compression Ratio value (used to slightly bias the offset).
    :return: Simulated PSNR-HVS-M value.
    """
    base_psnr_val = psnr(img1, img2, data_range=1.0)
    # Add a random offset to simulate perceptual variations
    offset = np.random.uniform(-0.5, 0.6) if cr_val < 15 else np.random.uniform(0.1, 0.6)
    return max(0, base_psnr_val + offset)
