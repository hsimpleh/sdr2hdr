import cv2
import numpy as np
import os
from .features import linearize_sdr, expand_features
from .model import load_models, predict_by_luminance
from .metric import compute_metrics

def infer_sdr_image(sdr_path, model_dir, alpha=0.7, gt_hdr_path=None, output_path=None):
    models_split, B = load_models(model_dir)
    sdr = cv2.imread(sdr_path)
    sdr_rgb = cv2.cvtColor(sdr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    sdr_lin = linearize_sdr(sdr_rgb)
    h, w, _ = sdr_lin.shape
    flat = sdr_lin.reshape(-1, 3)
    feat = expand_features(flat)
    pred_rf, pred_gbdt = predict_by_luminance(feat, models_split)
    pred_model = 0.5 * pred_rf + 0.5 * pred_gbdt
    pred_linear = flat @ B
    final = (alpha * pred_model + (1 - alpha) * pred_linear).reshape(h, w, 3)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor((final * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR))

    if gt_hdr_path:
        gt_img = cv2.imread(gt_hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
        gt_rgb = cv2.resize(gt_rgb, (w, h)) if gt_rgb.shape != final.shape else gt_rgb
        return compute_metrics(gt_rgb, final)
    return None
