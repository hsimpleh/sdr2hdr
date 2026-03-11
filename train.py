import os, glob
from tqdm import tqdm
import cv2
import numpy as np
from .features import linearize_sdr, linearize_hdr, expand_features
from .model import train_model

def load_pixels(sdr_path, hdr_path, step=5):
    sdr = cv2.cvtColor(cv2.imread(sdr_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hdr = cv2.cvtColor(cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
    if sdr.shape != hdr.shape:
        hdr = cv2.resize(hdr, (sdr.shape[1], sdr.shape[0]))
    sdr_lin = linearize_sdr(sdr)[::step, ::step].reshape(-1, 3)
    hdr_lin = linearize_hdr(hdr)[::step, ::step].reshape(-1, 3)
    return sdr_lin, hdr_lin

def train_from_dir(sdr_dir, hdr_dir, model_dir='models', max_pairs=50):
    sdr_files = sorted(glob.glob(os.path.join(sdr_dir, '*')))[:max_pairs]
    hdr_files = sorted(glob.glob(os.path.join(hdr_dir, '*')))[:max_pairs]
    X_all, Y_all = [], []
    for sdr_path, hdr_path in tqdm(zip(sdr_files, hdr_files), total=len(sdr_files)):
        if os.path.basename(sdr_path) != os.path.basename(hdr_path):
            continue
        X_raw, Y = load_pixels(sdr_path, hdr_path)
        X = expand_features(X_raw)
        X_all.append(X)
        Y_all.append(Y)
    if X_all:
        train_model(np.vstack(X_all), np.vstack(Y_all), save_dir=os.path.join(model_dir, 'model1'))
