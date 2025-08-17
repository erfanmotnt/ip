import numpy as np
from skimage import img_as_float, color
from metrics import psnr, fake_psnr_hvs_m, fsim_custom

def add_realistic_noise(image, sigma_mul=0.02, k=0.001, sigma_add=2.0):
    """
    Adds realistic dental-like noise to the image based on the paper's model.
    σ² = σ_mul² * I² + k * I + σ_add²
    :param image: Input image (float32 or float64, range 0-255)
    :return: Noisy image
    """
    I = image.astype(np.float32)
    
    # Compute per-pixel noise std using the model
    variance = (sigma_mul**2) * (I**2) + k * I + (sigma_add**2)
    std_dev = np.sqrt(variance)
    
    # Generate Gaussian noise with computed std
    noise = np.random.normal(loc=0.0, scale=std_dev)
    
    # Add noise and clip to 0-255
    noisy_img = np.clip(I + noise, 0, 255).astype(np.uint8)
    return noisy_img


# --- Helper function for padding image for DWT (for SPIHT) ---
def pad_image_for_dwt(img, levels):
    """Pads the image to ensure its dimensions are suitable for wavelet transform."""
    h, w = img.shape
    # Minimum dimension must be a multiple of 2 to the power of the number of wavelet levels
    min_dim = 2 ** levels 
    
    pad_h = 0 if h % min_dim == 0 else min_dim - (h % min_dim)
    pad_w = 0 if w % min_dim == 0 else min_dim - (w % min_dim)
    
    # Pad using edge replication
    padded_img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
    return padded_img, (h, w) # Return original dimensions for cropping after reconstruction


# --- Evaluation Function (with error handling) ---
def simulate_metrics_for_codec(original_image, cr_list, compress_func):
    """
    Executes compression with a given codec and calculates various metrics for different CRs.
    :param original_image: The uncompressed original image (float in 0-1).
    :param cr_list: A list of Compression Ratios to evaluate.
    :param compress_func: The compression function to be used (e.g., compress_jpeg2000, compress_spiht, compress_adct_actual_with_rate_control).
    :return: Lists of PSNR, PSNR-HVS-M, and FSIM values.
    """
    psnr_vals, psnr_hvs_vals, fsim_vals = [], [], []
    for cr in cr_list:
        try:
            # Call the compression function
            compressed = compress_func(original_image, cr)
            
            # Ensure images are in the correct range and type for metric calculation
            # For PSNR and FSIM, float in the range 0-1 is typically expected
            if compressed.max() > 1.0:
                compressed_for_metrics = img_as_float(compressed)
            else:
                compressed_for_metrics = compressed

            psnr_vals.append(psnr(original_image, compressed_for_metrics, data_range=1.0))
            psnr_hvs_vals.append(fake_psnr_hvs_m(original_image, compressed_for_metrics, cr))
            fsim_vals.append(fsim_custom(original_image, compressed_for_metrics))
        except Exception as e:
            # If an error occurs during compression or metric calculation, append NaN
            print(f"Warning: Error during {compress_func.__name__} for CR={cr}: {e}")
            psnr_vals.append(np.nan)
            psnr_hvs_vals.append(np.nan)
            fsim_vals.append(np.nan)
    return psnr_vals, psnr_hvs_vals, fsim_vals

