import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

def split_by_luminance(X_features, Y_targets):
    Y_vals = X_features[:, 9]
    masks = {
        "low": Y_vals < 0.33,
        "mid": (Y_vals >= 0.33) & (Y_vals <= 0.66),
        "high": Y_vals > 0.66
    }
    return {k: (X_features[m], Y_targets[m]) for k, m in masks.items()}

def predict_by_luminance(X_features, models_split):
    Y_vals = X_features[:, 9]
    preds_rf = np.zeros((len(X_features), 3), dtype=np.float32)
    preds_gbdt = np.zeros_like(preds_rf)
    for region, mask in {
        "low": Y_vals < 0.33,
        "mid": (Y_vals >= 0.33) & (Y_vals <= 0.66),
        "high": Y_vals > 0.66
    }.items():
        if not np.any(mask):
            continue
        rf_model = models_split[region]['rf']
        gbdt_model = models_split[region]['gbdt']
        preds_rf[mask] = rf_model.predict(X_features[mask])
        preds_gbdt[mask] = gbdt_model.predict(X_features[mask])
    return preds_rf, preds_gbdt

def train_model(X_all, Y_all, save_dir):
    subsets = split_by_luminance(X_all, Y_all)
    models_split = {}
    os.makedirs(save_dir, exist_ok=True)
    for region in ['low', 'mid', 'high']:
        X, Y = subsets[region]
        models_split[region] = {
            'rf': MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=6)),
            'gbdt': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=6)),
        }
        for name, model in models_split[region].items():
            model.fit(X, Y)
            joblib.dump(model, os.path.join(save_dir, f'{region}_{name}.pkl'))

    Yp_rf, Yp_gbdt = predict_by_luminance(X_all, models_split)
    B = np.linalg.lstsq(X_all[:, :3], 0.5 * (Yp_rf + Yp_gbdt), rcond=None)[0]
    np.save(os.path.join(save_dir, 'linear_matrix.npy'), B)

def load_models(model_dir):
    models_split = {
        region: {
            'rf': joblib.load(os.path.join(model_dir, f'{region}_rf.pkl')),
            'gbdt': joblib.load(os.path.join(model_dir, f'{region}_gbdt.pkl'))
        } for region in ['low', 'mid', 'high']
    }
    B = np.load(os.path.join(model_dir, 'linear_matrix.npy'))
    return models_split, B
