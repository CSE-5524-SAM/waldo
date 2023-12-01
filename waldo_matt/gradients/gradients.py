import math
from skimage import io
import skimage
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import multiprocessing


def process_gradients(start_row, end_row, template_gradients, template_rows, template_cols, field_grad_histograms, field_rows, field_cols):
    gradients_list = []
    for r in range(start_row, min(end_row, field_rows - template_rows)):
        for c in range(field_cols - template_cols):
            gradients = np.sum(field_grad_histograms[r:r+template_rows, c:c+template_cols], axis=(0,1))
            gradients = gradients / np.linalg.norm(gradients)
            dif = np.sum((template_gradients - gradients)**2)**.5
            gradients_list.append((dif, r, c))
    return gradients_list

def calculate_template_gradients(gxIm, gyIm, rows, cols):
    template_gradients = [0, 0, 0, 0, 0, 0, 0, 0]
    for r in range(rows):
        for c in range(cols):
            dx = gxIm[r, c]
            dy = gyIm[r, c]
            mag = math.sqrt(dx ** 2 + dy ** 2)
            if mag > 0:
                dir = np.arctan2(dy, dx)
                deg = math.degrees(dir)
                deg -= (45 + 45 / 2)
                deg += 360
                deg = deg % 360
                template_gradients[math.floor(deg / 45)] += mag
    return np.array(template_gradients) / np.linalg.norm(template_gradients)


def create_field_gradient_histograms(bin_index, mag, rows, cols):
    field_grad_histograms = np.zeros((rows, cols, 8))
    for r in range(rows):
        for c in range(cols):
            bin_idx = bin_index[r, c]
            field_grad_histograms[r, c, bin_idx] = mag[r, c]
    return field_grad_histograms

def run_gradient(test_image_path, ind):
    # Load and process the template image
    template = io.imread("sample_gradient/waldoW129noise.png")[:,:,:3]
    Im = skimage.color.rgb2gray(template)
    gx_mask = np.array([-1,0,1]).reshape((1,3))
    gy_mask = np.transpose(gx_mask)
    gxIm = scipy.ndimage.convolve(Im, gx_mask, mode='nearest')
    gyIm = scipy.ndimage.convolve(Im, gy_mask, mode='nearest')
    template_gradients = calculate_template_gradients(gxIm, gyIm, template.shape[0], template.shape[1])

    # Load and process the field image
    field = io.imread(test_image_path)[:,:,:3]
    field_grey = skimage.color.rgb2gray(field)
    field_gx = scipy.ndimage.convolve(field_grey, gx_mask, mode='nearest')
    field_gy = scipy.ndimage.convolve(field_grey, gy_mask, mode='nearest')
    field_mag = (field_gx**2 + field_gy**2)**.5
    field_dir = np.degrees(np.arctan2(field_gx, field_gy))
    field_deg = (field_dir - (45 + 45/2) + 360) % 360
    field_gradient_histogram_bin = np.floor(field_deg/45).astype(int)
    field_grad_histograms = create_field_gradient_histograms(field_gradient_histogram_bin, field_mag, field.shape[0], field.shape[1])

    # Parallel execution
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    chunk_size = field.shape[0] // num_processes
    tasks = [(i, i + chunk_size) for i in range(0, field.shape[0], chunk_size)]
    results = [pool.apply_async(process_gradients, args=(start, end, template_gradients, template.shape[0], template.shape[1], field_grad_histograms, field.shape[0], field.shape[1])) for start, end in tasks]

    # Collect results
    all_gradients = []
    for result in results:
        all_gradients.extend(result.get())

    pool.close()
    pool.join()

    # Find the best match
    all_gradients.sort(key=lambda x: x[0])
    best_dif, bestr, bestc = all_gradients[0]

    mask = np.zeros_like(field, dtype=np.uint8)
    cv2.rectangle(mask, (bestc, bestr), (bestc + template.shape[1], bestr + template.shape[0]), (255, 255, 255), -1)
    blurred_field = cv2.GaussianBlur(field, (21, 21), 0)
    blurred_field = np.clip(blurred_field * 0.5, 0, 255).astype(np.uint8)
    final_image = np.where(mask == np.array([255, 255, 255]), field, blurred_field)
    cv2.rectangle(final_image, (bestc, bestr), (bestc + template.shape[1], bestr + template.shape[0]), (255, 0, 0), 2)
    plt.imsave(f"result/gradient/gradient_result_{ind}.jpg", final_image)

    return (bestc, bestr, bestc + template.shape[1], bestr + template.shape[0])