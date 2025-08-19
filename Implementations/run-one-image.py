import os
import numpy as np
import subprocess
import metrics
import pandas as pd
from skimage import io, img_as_float, color
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
from skimage import img_as_float

# ----------------------------
# Read raw pixel data directly
# ----------------------------

def read_raw(raw_image):
    with open(raw_image, 'rb') as f:
        # Specify dimensions and data type
        image = np.fromfile(f, dtype=np.uint8)
    image = image.reshape((512, 512))
    
    if image.ndim == 3: 
        image = color.rgb2gray(image) # Convert to grayscale if it's a color image
    image = img_as_float(image) # Convert to float for processing (0-1 range)
    
    return image


# ----------------------------
# Function for one quality setting
# ----------------------------
def process_quality_setting_adct(qs, input_path, original_image, destination_folder):
    destination_path = os.path.join(destination_folder, f'input-{qs}.agu')
    compressed_path = os.path.join(destination_folder, f'decompressed-{qs}.raw')

    # Compress
    compress_cmd = f'wine ADCT.EXE e {input_path} {destination_path} {qs}'
    subprocess.run(compress_cmd, shell=True, check=True)

    # Decompress
    decompress_cmd = f'wine ADCT.EXE d {destination_path} {compressed_path}'
    subprocess.run(decompress_cmd, shell=True, check=True)

    # Load compressed image
    compressed_image = read_raw(compressed_path)

    # Compute metrics
    psnr_hvs_m_val = metrics.psnr_hvs_m(original_image, compressed_image)
    fsim_val = metrics.fsim(original_image, compressed_image)
    psnr_val = metrics.psnr(original_image, compressed_image)

    # Compute Compression Ratio
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(destination_path)
    cr = original_size / compressed_size if compressed_size > 0 else np.inf

    return {
        "quality_setting": qs,
        "psnr_hvs_m": psnr_hvs_m_val,
        "fsim": fsim_val,
        "psnr": psnr_val,
        "compression_ratio": cr
    }


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
    original_size = img_uint8.nbytes
    compressed_size = len(enc)
    actual_cr = original_size / compressed_size
    reconstructed = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
    return reconstructed, actual_cr


# ----------------------------
# Function for one quality setting
# ----------------------------
def process_quality_setting_jpeg20000(qs, input_path, original_image, destination_folder):
    
    compressed_image, cr_act = compress_jpeg2000(original_image, qs)
    if compressed_image.max() > 1.0:
        compressed_image = img_as_float(compressed_image)
    else:
        pass

    # Compute metrics
    psnr_hvs_m_val = metrics.psnr_hvs_m(original_image, compressed_image)
    fsim_val = metrics.fsim(original_image, compressed_image)
    psnr_val = metrics.psnr(original_image, compressed_image)

    cr = cr_act
    
    return {
        "quality_setting": qs,
        "psnr_hvs_m": psnr_hvs_m_val,
        "fsim": fsim_val,
        "psnr": psnr_val,
        "compression_ratio": cr
    }


# ----------------------------
# Main Multi-thread Function
# ----------------------------
def compress_and_analyze_image(input_path, coder, qs_range=(6, 41), max_workers=4, output_file="metrics_results.csv"):
    destination_folder = 'datasets/' + input_path.split('/')[-1].split('.')[0]
    print(destination_folder)
    os.makedirs(destination_folder, exist_ok=True)

    # Load original image
    original_image = read_raw(input_path)
    results = []

    # Run multi-threaded processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(coder, qs, input_path, original_image, destination_folder): qs
            for qs in range(qs_range[0], qs_range[1] + 1)
        }

        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            print(f"Completed QS={res['quality_setting']}")
            # try:
            #     res = future.result()
            #     results.append(res)
            #     print(f"Completed QS={res['quality_setting']}")
            # except Exception as e:
            #     print(f"Error with QS={futures[future]}: {e}")

    # Convert results to DataFrame
    df = pd.DataFrame(results)#.sort_values("quality_setting")

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Metrics saved to {output_file}")

    return df


# ----------------------------
# Example run
# ----------------------------
if __name__ == "__main__":
    output_folder = 'datasets/dental-images-raw-512'
    for fname in reversed(os.listdir(output_folder)):
        print(fname)
        input_image = os.path.join(output_folder, fname)
        df = compress_and_analyze_image(
            input_image,
            process_quality_setting_jpeg20000,
            qs_range=(6, 42),
            max_workers=6,
            output_file=f"csvs/{fname}_metrics_results.csv"
        )
