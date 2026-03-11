import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

def compute_metrics(gt, pred):
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim_val = structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)
    delta_e_val = compute_delta_e(gt, pred)
    return psnr_val, ssim_val, delta_e_val

def compute_delta_e(img1, img2, step=10):
    h, w, _ = img1.shape
    total = count = 0
    for i in range(0, h, step):
        for j in range(0, w, step):
            lab1 = convert_color(sRGBColor(*img1[i, j], is_upscaled=False), LabColor)
            lab2 = convert_color(sRGBColor(*img2[i, j], is_upscaled=False), LabColor)
            total += delta_e_cie2000(lab1, lab2)
            count += 1
    return total / count if count else 0
