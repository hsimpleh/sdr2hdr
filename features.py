import numpy as np
import cv2

def linearize_sdr(rgb):
    return np.power(rgb, 2.2)

def linearize_hdr(rgb):
    return rgb

def expand_features(rgb_lin):
    r, g, b = rgb_lin[:, 0], rgb_lin[:, 1], rgb_lin[:, 2]
    rgb_sq = rgb_lin ** 2
    rgb_sqrt = np.sqrt(rgb_lin)
    Y = (0.2126 * r + 0.7152 * g + 0.0722 * b).reshape(-1, 1)
    rgb_img = rgb_lin.reshape(-1, 1, 3).astype(np.float32)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV).reshape(-1, 3) / 255.0
    features = np.concatenate([rgb_lin, rgb_sq, rgb_sqrt, Y, hsv_img], axis=1)
    return features

def tonemap(hdr, gamma=1.0 / 2.2, exposure=1.0):
    mapped = hdr * exposure
    mapped = mapped / (1.0 + mapped)
    mapped = np.power(mapped, gamma)
    return np.clip(mapped, 0.0, 1.0)
