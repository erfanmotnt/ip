from skimage.filters import sobel_h, sobel_v
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import img_as_float, color
from skimage.metrics import peak_signal_noise_ratio as psnr
from psnr_hvsm import psnr_hvs_hvsm
from image_similarity_measures.quality_metrics import fsim as fsim_metric

to_3d = lambda image_2d: np.repeat(image_2d[:, :, np.newaxis], 3, axis=2)

fsim = lambda img1, img2: fsim_metric(to_3d(img1), to_3d(img2))

psnr_hvs_m = lambda img1, img2: psnr_hvs_hvsm(img1, img2)[1].item()
