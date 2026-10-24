import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

FEATURE_DIR = PROJECT_ROOT / "data"
FEATURE_PATH = FEATURE_DIR / 'train_features.parquet'
FEATURE_COLS_PATH = FEATURE_DIR / 'feature_cols.json'
CAT_COLS_PATH = FEATURE_DIR / 'cat_cols.json'

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_5B" / "lightgbm"
MODEL_DIR = OUTPUT_DIR / 'models'
OOF_DIR = OUTPUT_DIR / 'oof'
LOG_DIR = OUTPUT_DIR / 'logs'
for p in [MODEL_DIR, OOF_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

COLUMNS_TO_DROP = [
    'tbp_lv_B','tbp_lv_C','tbp_lv_H','tbp_lv_L','tbp_lv_radial_color_std_max',
    'tbp_lv_y','tbp_lv_z','luminance_contrast','lesion_color_difference',
    'normalized_lesion_size','tbp_lv_norm_border_patient_norm',
    'lesion_color_difference_patient_norm','age_normalized_nevi_confidence_2_patient_norm',
    'tbp_lv_deltaA','size_age_interaction','tbp_lv_deltaB','oof_eva_score__cluster',
    'hue_contrast_patient_norm__cluster','color_contrast_index_patient_norm',
    'tbp_lv_Lext__cluster','tbp_lv_Cext_patient_norm','oof_eva_score_m',
    'age_approx_patient_norm','volume_approximation_3d_patient_norm'
]

NOISE_STD = 0.1
NOISE_FEATURES = ['oof_edgenext_score', 'oof_eva_score']

LGB_PARAMS = {
    'objective': 'binary',
    'verbosity': -1,
    'n_estimators': 200,
    'boosting_type': 'gbdt',
    'device': 'gpu',
    'lambda_l1': 0.08758718919397321,
    'lambda_l2': 0.0039689175176025465,
    'learning_rate': 0.03231007103195577,
    'max_depth': 4,
    'num_leaves': 103,
    'colsample_bytree': 0.8329551585827726,
    'colsample_bynode': 0.4025961355653304,
    'bagging_fraction': 0.7738954452473223,
    'bagging_freq': 4,
    'min_data_in_leaf': 85,
    'scale_pos_weight': 2.7984184778875543,
}

def custom_metric_raw(y_hat, y_true):
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    return 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

parser = argparse.ArgumentParser()
parser.add_argument('--seed-index', type=int, default=-1, help='-1 = all 9 seeds; otherwise 1..9')
parser.add_argument('--fold', type=int, default=-1, help='-1 = all 5 folds; otherwise 0..4')
args = parser.parse_args()

df = pd.read_parquet(FEATURE_PATH)
feature_cols = json.loads(FEATURE_COLS_PATH.read_text())
cat_cols = json.loads(CAT_COLS_PATH.read_text())
model_features = [c for c in feature_cols if c not in COLUMNS_TO_DROP]

if len(model_features) != 242:
    raise RuntimeError(f'Expected 242 model features, got {len(model_features)}')
for c in NOISE_FEATURES:
    if c not in model_features:
        raise RuntimeError(f'Missing noise feature: {c}')

for c in cat_cols:
    if c in model_features:
        if df[c].isna().any():
            df[c] = df[c].astype(object).where(df[c].notna(), '__MISSING__')
        df[c] = df[c].astype('category')

seed_indices = list(range(1, 10)) if args.seed_index == -1 else [args.seed_index]
wanted_folds = list(range(5)) if args.fold == -1 else [args.fold]

print('=' * 90)
print('BASELINE 5 - ORIGINAL ARTIFACTS - LIGHTGBM x45')
print('=' * 90)
print('Rows:', len(df))
print('Model features:', len(model_features))
print('Gaussian noise: N(0, 0.1) TRAIN only ->', NOISE_FEATURES)

all_scores = []
all_oof = []

for seed_index in seed_indices:
    random_seed = seed_index * 10 + 17
    print(f'\n### SEED INDEX {seed_index} / LGB SEED {random_seed} ###')
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=random_seed)
    for fold_n, (train_index, val_index) in enumerate(cv.split(df, y=df.target, groups=df.patient_id)):
        if fold_n not in wanted_folds:
            continue
        print(f'\nSEED {random_seed} / FOLD {fold_n}')
        Xtr = df.iloc[train_index][model_features].reset_index(drop=True)
        Xva = df.iloc[val_index][model_features].reset_index(drop=True)
        ytr = df.iloc[train_index].target.reset_index(drop=True)
        yva = df.iloc[val_index].target.reset_index(drop=True)

        # Match original old-LGB path: noise on direct CNN scores before ROS/RUS.
        for col in NOISE_FEATURES:
            Xtr[col] = Xtr[col] + np.random.normal(0.0, NOISE_STD, size=len(Xtr))

        sampler = Pipeline([
            ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=random_seed)),
            ('sampler_2', RandomUnderSampler(sampling_strategy=0.01, random_state=random_seed)),
        ])
        Xtr, ytr = sampler.fit_resample(Xtr, ytr)

        params = dict(LGB_PARAMS)
        params['random_state'] = random_seed
        clf = lgb.LGBMClassifier(**params)
        clf.fit(Xtr, ytr, categorical_feature=cat_cols)
        preds = clf.predict_proba(Xva)[:, 1]

        pauc = custom_metric_raw(preds, yva.values)
        auroc = roc_auc_score(yva.values, preds)
        print(f'Validation AUROC: {auroc:.8f}')
        print(f'Validation pAUC : {pauc:.8f}')

        joblib.dump(clf, MODEL_DIR / f'lightgbm_seed{random_seed}_fold{fold_n}.joblib')
        oof_tmp = pd.DataFrame({
            'isic_id': df.iloc[val_index].isic_id.values,
            'patient_id': df.iloc[val_index].patient_id.values,
            'target': yva.values,
            'prediction': preds,
            'seed_index': seed_index,
            'random_seed': random_seed,
            'fold_n': fold_n,
        })
        oof_tmp.to_parquet(OOF_DIR / f'oof_seed{random_seed}_fold{fold_n}.parquet', index=False)
        all_oof.append(oof_tmp)
        all_scores.append({'seed_index': seed_index, 'random_seed': random_seed, 'fold_n': fold_n, 'pauc': pauc, 'auroc': auroc})
        pd.DataFrame(all_scores).to_csv(LOG_DIR / 'scores_progress.csv', index=False)

scores_df = pd.DataFrame(all_scores)
oof_df = pd.concat(all_oof, ignore_index=True)
scores_df.to_csv(LOG_DIR / 'scores.csv', index=False)
oof_df.to_parquet(OOF_DIR / 'oof_all_models.parquet', index=False)

print('\n' + '=' * 90)
print('LIGHTGBM TRAINING COMPLETE')
print('=' * 90)
print('Models trained:', len(scores_df))
print('Mean validation pAUC:', scores_df.pauc.mean())
print('Mean validation AUROC:', scores_df.auroc.mean())

if len(seed_indices) == 9 and len(wanted_folds) == 5:
    avg = oof_df.groupby(['isic_id','patient_id','target'], as_index=False).prediction.mean()
    auroc = roc_auc_score(avg.target, avg.prediction)
    pauc = custom_metric_raw(avg.prediction.values, avg.target.values)
    avg.to_parquet(OOF_DIR / 'oof_mean_across_9_seeds.parquet', index=False)
    print('Seed-averaged OOF AUROC:', auroc)
    print('Seed-averaged OOF pAUC :', pauc)
