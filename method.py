import numpy as np
from scipy.fftpack import dct, idct
from numba import njit

# --- 2D DCT and IDCT Functions ---
def dct2(block):
    """Performs a 2D Discrete Cosine Transform."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """Performs a 2D Inverse Discrete Cosine Transform."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

@njit
def compute_nosb_array(arr):
    h, w = arr.shape
    result = np.zeros((h, w), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            val = abs(int(arr[i, j]))
            if val == 0:
                result[i, j] = 0
            else:
                result[i, j] = int(np.floor(np.log2(val)) + 1)
    return result

@njit
def estimate_bits_njit(quant_coeffs, nosb_map, QS, Tr, k1, k2, k3):
    total_bits = 0.0
    h, w = quant_coeffs.shape
    for i in range(h):
        for j in range(w):
            coeff = quant_coeffs[i, j]
            nosb = nosb_map[i, j]
            abs_coeff = abs(coeff)
            if abs_coeff * QS < Tr:
                abs_coeff = 0
            if abs_coeff == 0:
                total_bits += k3
            else:
                total_bits += k1 * nosb + k2
    return max(1, int(total_bits))


# --- Advanced ADCTCCompressor Class ---
class ADCTCCompressor:
    def __init__(self, min_block_size=8, max_block_size=64, QS=12):
        """
        Initializes the ADCTC Compressor.
        :param min_block_size: Minimum allowed block size for partitioning.
        :param max_block_size: Maximum (initial) block size for partitioning.
        :param QS: Quantization Step.
        """
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.QS = QS
        # Tr: Additional quantization threshold for zeroing out small coefficients.
        # This value is based on the ADCTC paper.
        self.Tr = 0.4 * QS  # According to the ADCTC paper

        # Parameters for the simplified rate model (can be calibrated)
        # These weights empirically estimate the bit cost for different types of coefficients
        # if self.QS < 10:  # یعنی CR پایین
        #     self.k1 = 0.3
        #     self.k2 = 0.2
        #     self.k3 = 0.01
        # else:
        self.k1 = 1.2  # Weight for each bit of NOSB
        self.k2 = 1.0  # Weight for the sign bit (for non-zero coefficients)
        self.k3 = 0.05  # Weight for zero coefficients (background bits)

    def cost_function(self, quant_block):
        """
        Cost function E = ∑ ln(1 + |Xi| / QS).
        Used to guide partition optimization, though the final decision is based on estimated bits.
        """
        abs_coeff = np.abs(quant_block)
        E = np.sum(np.log(1 + abs_coeff / self.QS))
        return E

    # def estimate_block_bits_arithmetic(self, quant_coeffs, nosb_map):
    #     """
    #     Estimates the bits required for a block considering additional quantization
    #     and a simplified rate model. This function uses empirical weights instead of
    #     a true arithmetic encoder.
    #     :param quant_coeffs: Quantized DCT coefficients of the block.
    #     :param nosb_map: NOSB values for each coefficient.
    #     :return: Estimated total bits for the block.
    #     """
    #     total_bits = 0
    #     flat_coeffs = quant_coeffs.flatten()
    #     flat_nosb = nosb_map.flatten()

    #     for coeff, nosb in zip(flat_coeffs, flat_nosb):
    #         abs_coeff = abs(coeff)

    #         # Apply additional quantization: if the original coefficient value (before quantization)
    #         # was less than Tr, consider it as zero for bit counting purposes.
    #         # We check the magnitude of the de-quantized coefficient against Tr.
    #         if abs_coeff * self.QS < self.Tr: 
    #             abs_coeff = 0 # Zero out the coefficient for bit counting

    #         if abs_coeff == 0:
    #             total_bits += self.k3  # Cost for encoding a zero coefficient
    #         else:
    #             bits_nosb = nosb * self.k1 # Cost for NOSB
    #             bits_sign = self.k2      # Cost for the sign bit
    #             total_bits += bits_nosb + bits_sign

    #     return max(1, int(total_bits)) # Return at least 1 bit to prevent division by zero in CR calculation
    

    def estimate_block_bits_arithmetic(self, quant_coeffs, nosb_map):
        return estimate_bits_njit(quant_coeffs, nosb_map,
                                self.QS, self.Tr,
                                self.k1, self.k2, self.k3)

    def partition_optimization(self, block, r_offset=0, c_offset=0, level=0):
        """
        Implements the Partition Scheme (PS) Optimization recursively.
        This function selects the best partition scheme (no split, vertical, horizontal, Quadtree)
        based on minimizing the estimated number of bits.
        :param block: The current image block to process.
        :param r_offset: Row offset of the current block within the original image.
        :param c_offset: Column offset of the current block within the original image.
        :param level: Recursion depth (for potential future use or debugging).
        :return: A list of tuples, each representing a final (unsplit) block:
                 (block_data, (row_offset, col_offset), height, width, level),
                 and the total estimated bits for this partition.
        """
        h, w = block.shape
        QS = self.QS

        # Base case for recursion: If the block reaches or falls below the minimum size
        if h < self.min_block_size or w < self.min_block_size:
            # Even if the block is smaller than min_block_size (e.g., due to odd dimensions),
            # it is processed as a final block.
            dct_block = dct2(block)
            quant_block = np.round(dct_block / QS)
            nosb_map = compute_nosb_array(quant_block)
            bits = self.estimate_block_bits_arithmetic(quant_block, nosb_map)
            # Return a list containing this single final block
            return [(block, (r_offset, c_offset), h, w, level)], bits

        # --- No Split Option ---
        dct_block_no_split = dct2(block)
        quant_block_no_split = np.round(dct_block_no_split / QS)
        nosb_map_no_split = compute_nosb_array(quant_block_no_split)
        bits_no_split = self.estimate_block_bits_arithmetic(quant_block_no_split, nosb_map_no_split)
        
        # Initialize best case with "no split"
        best_partition = [(block, (r_offset, c_offset), h, w, level)]
        best_bits = bits_no_split

        # --- Vertical Split Option (2 blocks) ---
        # Only if the block is large enough for a vertical split (at least twice min_block_size in width)
        if w >= 2 * self.min_block_size:
            left_block = block[:, :w//2]
            right_block = block[:, w//2:]

            # Recursive calls for sub-blocks
            left_part, left_bits = self.partition_optimization(left_block, r_offset, c_offset, level + 1)
            right_part, right_bits = self.partition_optimization(right_block, r_offset, c_offset + w//2, level + 1)

            # Calculate total bits for this splitting scheme (including partition signaling overhead)
            total_split_bits_v = left_bits + right_bits + 3 # Overhead: e.g., 3 bits for signaling split type (vertical)
            
            # Compare and update best case if this split results in fewer bits
            if total_split_bits_v < best_bits:
                best_bits = total_split_bits_v
                best_partition = left_part + right_part

        # --- Horizontal Split Option (2 blocks) ---
        # Only if the block is large enough for a horizontal split (at least twice min_block_size in height)
        if h >= 2 * self.min_block_size:
            top_block = block[:h//2, :]
            bottom_block = block[h//2:, :]

            # Recursive calls for sub-blocks
            top_part, top_bits = self.partition_optimization(top_block, r_offset, c_offset, level + 1)
            bottom_part, bottom_bits = self.partition_optimization(bottom_block, r_offset + h//2, c_offset, level + 1)

            # Calculate total bits for this splitting scheme
            total_split_bits_h = top_bits + bottom_bits + 3 # Overhead
            
            # Compare and update best case
            if total_split_bits_h < best_bits:
                best_bits = total_split_bits_h
                best_partition = top_part + bottom_part

        # --- Quadtree Split Option (4 blocks) ---
        # Only if the block is large enough for a Quadtree split (even dimensions and at least twice min_block_size)
        if h >= 2 * self.min_block_size and w >= 2 * self.min_block_size:
            half_h, half_w = h // 2, w // 2
            q1 = block[:half_h, :half_w]
            q2 = block[:half_h, half_w:]
            q3 = block[half_h:, :half_w]
            q4 = block[half_h:, half_w:]

            # Recursive calls for sub-blocks
            q1_part, q1_bits = self.partition_optimization(q1, r_offset, c_offset, level + 1)
            q2_part, q2_bits = self.partition_optimization(q2, r_offset, c_offset + half_w, level + 1)
            q3_part, q3_bits = self.partition_optimization(q3, r_offset + half_h, c_offset, level + 1)
            q4_part, q4_bits = self.partition_optimization(q4, r_offset + half_h, c_offset + half_w, level + 1)

            # Calculate total bits for this splitting scheme
            total_split_bits_q = q1_bits + q2_bits + q3_bits + q4_bits + 5 # Overhead (e.g., 5 bits for signaling Quadtree)
            
            # Compare and update best case
            if total_split_bits_q < best_bits:
                best_bits = total_split_bits_q
                best_partition = q1_part + q2_part + q3_part + q4_part
        
        return best_partition, best_bits

    def compress(self, img):
        """
        Main ADCTC compression function.
        Divides the image into initial (super) blocks and runs partition optimization for each.
        :param img: The input grayscale image (NumPy array).
        :return: Reconstructed image, total compressed bits, and compression ratio.
        """
        h, w = img.shape
        reconstructed_img = np.zeros_like(img, dtype=np.float32)
        block_boundaries = []  # to store (row, col, height, width) for each block
        total_compressed_bits = 0

        # Pad the image to ensure dimensions are multiples of max_block_size
        pad_h = 0 if h % self.max_block_size == 0 else self.max_block_size - (h % self.max_block_size)
        pad_w = 0 if w % self.max_block_size == 0 else self.max_block_size - (w % self.max_block_size)
        padded_img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
        padded_h, padded_w = padded_img.shape

        # Process the image in large blocks (max_block_size)
        for i in range(0, padded_h, self.max_block_size):
            for j in range(0, padded_w, self.max_block_size):
                # Extract the initial block (Super-block) to start partitioning
                block_to_partition = padded_img[i:i+self.max_block_size, j:j+self.max_block_size]

                # Execute PS (Partition Scheme) Optimization recursively
                partition_blocks, bits_for_this_super_block = self.partition_optimization(
                    block_to_partition, i, j, 0 # r_offset and c_offset here indicate global block coordinates
                )
                total_compressed_bits += bits_for_this_super_block

                # Reconstruct the image from the optimized final blocks
                for b, (r_off, c_off), bh, bw, lvl in partition_blocks:
                    # Apply DCT to the block
                    dct_b = dct2(b)
                    # Quantize DCT coefficients
                    quant_b = np.round(dct_b / self.QS)
                    
                    # IMPORTANT: For reconstruction, coefficients that were zeroed out
                    # by additional quantization must also be zeroed here.
                    # This ensures that the reconstruction matches the encoded bits.
                    quant_b_for_recon = quant_b.copy()
                    # Check if the de-quantized value would have been less than Tr
                    # quant_b_for_recon[np.abs(quant_b_for_recon * self.QS) < self.Tr] = 0
                    
                    # De-quantize coefficients by multiplying by QS
                    dequant_b = quant_b_for_recon * self.QS
                    # Apply IDCT to reconstruct the block
                    recon_b = idct2(dequant_b)
                    
                    # Ensure pixel values are within the 0-255 range
                    recon_b = np.clip(recon_b, 0, 255)
                    
                    # Place the reconstructed block into the overall reconstructed image
                    reconstructed_img[r_off:r_off+bh, c_off:c_off+bw] = recon_b
                    block_boundaries.append((r_off, c_off, bh, bw))


        # Crop the reconstructed image to the original size (remove padding)
        reconstructed_img = reconstructed_img[:h, :w].astype(np.uint8)
        
        # Calculate compression ratio
        original_bits = h * w * 8 # 8-bit grayscale image
        compression_ratio = original_bits / total_compressed_bits if total_compressed_bits > 0 else float('inf')
        # Draw block boundaries for visualization
        vis_img = reconstructed_img.copy().astype(np.uint8)
        for r, c, h, w in block_boundaries:
            vis_img[r:r+1, c:c+w] = 255     # Top border
            vis_img[r+h-1:r+h, c:c+w] = 255 # Bottom border
            vis_img[r:r+h, c:c+1] = 255     # Left border
            vis_img[r:r+h, c+w-1:c+w] = 255 # Right border

        return reconstructed_img, total_compressed_bits, compression_ratio, vis_img
    
# --- Actual ADCTC Algorithm (using ADCTCCompressor) with Rate Control ---
def compress_adct_actual_with_rate_control(img, cr_target, min_block_size=8, max_block_size=64, max_iterations=20, tolerance=0.1):
    """
    Compresses an image using ADCTC with an iterative search for the optimal QS
    to achieve a target Compression Ratio (CR).
    :param img: The input image (float in 0-1 or uint8 in 0-255).
    :param cr_target: The desired Compression Ratio.
    :param min_block_size: Minimum block size for ADCTC.
    :param max_block_size: Maximum block size for ADCTC.
    :param max_iterations: Maximum iterations for QS search.
    :param tolerance: Acceptable difference from the target CR.
    :return: Reconstructed image (float in 0-1).
    """
    # Ensure the image is in the 0-255 range and uint8 type for ADCTCCompressor
    if img.max() <= 1.0:
        img_processed = (img * 255).astype(np.uint8)
    else:
        img_processed = img.astype(np.uint8)

    low_qs = 1
    high_qs = 255  # A reasonable range for QS, may need adjustment
    best_qs = None
    best_compression_ratio_diff = float('inf')
    reconstructed_img_uint8 = None # To store the best reconstructed image

    for iteration in range(max_iterations):
        current_qs = int((low_qs + high_qs) / 2)
        if low_qs > high_qs: # Prevent infinite loop if range collapses
            break
        
        # If current_qs becomes zero, it can be problematic (division by zero). Use a minimum value.
        if current_qs == 0: 
            current_qs = 1 

        compressor = ADCTCCompressor(min_block_size=min_block_size,
                                     max_block_size=max_block_size,
                                     QS=current_qs)
        
        # Perform compression for the current QS
        recon_img, total_bits, compression_ratio_actual, _ = compressor.compress(img_processed)
        
        # Store results of this iteration if it's the closest to the target CR
        current_diff = abs(compression_ratio_actual - cr_target)
        if current_diff < best_compression_ratio_diff:
            best_compression_ratio_diff = current_diff
            best_qs = current_qs
            reconstructed_img_uint8 = recon_img

        # Check if the target CR is reached within tolerance
        if abs(compression_ratio_actual - cr_target) < tolerance:
            print(f"ADCTC: Target CR {cr_target:.2f} reached with QS={current_qs} (Actual CR: {compression_ratio_actual:.2f})")
            break
        elif compression_ratio_actual < cr_target:
            # Actual CR is too low (less compression), need higher QS (harsher quantization)
            low_qs = current_qs + 1
        else:
            # Actual CR is too high (more compression), need lower QS (milder quantization)
            high_qs = current_qs - 1
        
        print(f"Iteration {iteration+1}: QS={current_qs}, Actual CR={compression_ratio_actual:.2f}, Diff={current_diff:.4f}")
            
    # If target CR was not reached after all iterations, use the best QS found
    if reconstructed_img_uint8 is None: 
        # If no valid QS was found during iterations, use a default QS (e.g., 20)
        final_qs = best_qs if best_qs is not None else 20
        print(f"ADCTC: Did not reach target CR. Using best QS found: {final_qs}")
        compressor = ADCTCCompressor(min_block_size=min_block_size,
                                     max_block_size=max_block_size,
                                     QS=final_qs)
        reconstructed_img_uint8, _, _ = compressor.compress(img_processed)

    # Convert the reconstructed image back to 0-1 range (if original input was like that) and float32 type
    if img.max() <= 1.0:
        return reconstructed_img_uint8.astype(np.float32) / 255.0
    else:
        return reconstructed_img_uint8.astype(np.float32)
