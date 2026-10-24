import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from catboost import CatBoostClassifier, Pool


# ============================================================
# Paths
# ============================================================

FEATURE_DIR = PROJECT_ROOT / "data"

FEATURE_PATH = (
    FEATURE_DIR /
    "train_features.parquet"
)

FEATURE_COLS_PATH = (
    FEATURE_DIR /
    "feature_cols.json"
)

CAT_COLS_PATH = (
    FEATURE_DIR /
    "cat_cols.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_5B" / "catboost"

MODEL_DIR = OUTPUT_DIR / "models"
OOF_DIR = OUTPUT_DIR / "oof"
LOG_DIR = OUTPUT_DIR / "logs"

for p in [MODEL_DIR, OOF_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Original notebook configuration
# ============================================================

sampling_ratio = 0.01

# Baseline 4B:
# Match the winner-style CatBoost training path by perturbing only the
# two direct competition CNN prediction features in the TRAINING slice.
# Validation/inference stay noise-free.
NOISE_STD = 0.1
NOISE_FEATURES = [
    "oof_edgenext_score",
    "oof_eva_score",
]

CB_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "task_type": "GPU",

    "learning_rate": 0.02606161517843435,
    "depth": 6,
    "l2_leaf_reg": 18.04422276698195,
    "random_strength": 4.7069580783889995,
    "min_data_in_leaf": 38,
    "bagging_temperature": 0.8735940473548339,
    "border_count": 256,
    "grow_policy": "Lossguide",

    "iterations": 2000,

    # Same as notebook
    "od_wait": 100,

    # Original uses devices='0'
    "devices": "0",
}


# ============================================================
# Exact columns_to_drop from Cell 24
# ============================================================

columns_to_drop = [
    "tbp_lv_B",
    "tbp_lv_C",
    "tbp_lv_H",
    "tbp_lv_L",
    "tbp_lv_radial_color_std_max",
    "tbp_lv_y",
    "tbp_lv_z",
    "luminance_contrast",
    "lesion_color_difference",
    "normalized_lesion_size",
    "tbp_lv_norm_border_patient_norm",
    "lesion_color_difference_patient_norm",
    "age_normalized_nevi_confidence_2_patient_norm",
    "tbp_lv_deltaA",

    "size_age_interaction",
    "tbp_lv_deltaB",
    "oof_eva_score__cluster",
    "hue_contrast_patient_norm__cluster",
    "color_contrast_index_patient_norm",
    "tbp_lv_Lext__cluster",
    "tbp_lv_Cext_patient_norm",
    "oof_eva_score_m",
    "age_approx_patient_norm",
    "volume_approximation_3d_patient_norm",
]


# ============================================================
# Competition metric
# ============================================================

def custom_metric_raw(y_hat, y_true):

    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)

    v_gt = abs(y_true - 1)
    v_pred = np.array(
        [1.0 - x for x in y_hat]
    )

    partial_auc_scaled = roc_auc_score(
        v_gt,
        v_pred,
        max_fpr=max_fpr,
    )

    partial_auc = (
        0.5 * max_fpr**2
        + (
            max_fpr
            - 0.5 * max_fpr**2
        )
        / (1.0 - 0.5)
        * (
            partial_auc_scaled
            - 0.5
        )
    )

    return partial_auc


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed",
    type=int,
    default=-1,
    help="-1 = seeds 1..9"
)

parser.add_argument(
    "--fold",
    type=int,
    default=-1,
    help="-1 = all 5 folds"
)

args = parser.parse_args()


# ============================================================
# Load features
# ============================================================

print("=" * 90)
print("WINNER ORIGINAL ARTIFACTS CATBOOST - BASELINE 4D (GAUSSIAN NOISE + RANDOM STRENGTH)")
print("=" * 90)

df = pd.read_parquet(
    FEATURE_PATH
)

with open(
    FEATURE_COLS_PATH,
    "r",
    encoding="utf-8"
) as f:
    feature_cols = json.load(f)

with open(
    CAT_COLS_PATH,
    "r",
    encoding="utf-8"
) as f:
    cat_cols = json.load(f)


model_features = [
    c for c in feature_cols
    if c not in columns_to_drop
]

missing_noise_features = [
    c for c in NOISE_FEATURES
    if c not in model_features
]

if missing_noise_features:
    raise RuntimeError(
        f"Noise features absent from model feature list: "
        f"{missing_noise_features}"
    )


print("Rows:", len(df))
print("Positive:", int(df["target"].sum()))
print(
    "Negative:",
    int(len(df) - df["target"].sum())
)

print(
    "Original feature cols:",
    len(feature_cols)
)

print(
    "Dropped features:",
    len(columns_to_drop)
)

print(
    "Model features:",
    len(model_features)
)

print(
    "Categorical features:",
    len(cat_cols)
)

print(
    "Gaussian noise:",
    f"N(0, {NOISE_STD}) on TRAIN only -> "
    f"{NOISE_FEATURES}"
)


# ============================================================
# Integrity checks
# ============================================================

missing = [
    c for c in model_features
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing model features: {missing}"
    )


missing_cat = [
    c for c in cat_cols
    if c not in model_features
]

if missing_cat:
    raise RuntimeError(
        f"Categorical features absent "
        f"from model feature list: {missing_cat}"
    )


# Ensure CatBoost categorical columns
for c in cat_cols:

    # Original notebook has one-hot columns stored
    # as pandas category.
    #
    # CatBoost categorical values should not be NaN.
    if df[c].isna().any():
        df[c] = (
            df[c]
            .astype(object)
            .where(
                df[c].notna(),
                "__MISSING__"
            )
        )

    df[c] = df[c].astype(str)


# ============================================================
# Seeds / folds
# ============================================================

if args.seed == -1:
    seeds = list(range(1, 10))
else:
    if args.seed < 1 or args.seed > 9:
        raise ValueError(
            "--seed must be 1..9 or -1"
        )

    seeds = [args.seed]


if args.fold == -1:
    wanted_folds = list(range(5))
else:
    if args.fold < 0 or args.fold > 4:
        raise ValueError(
            "--fold must be 0..4 or -1"
        )

    wanted_folds = [args.fold]


# ============================================================
# Training
# ============================================================

all_scores = []
all_oof = []


for random_seed in seeds:

    print("\n" + "#" * 90)
    print(f"RANDOM SEED {random_seed}")
    print("#" * 90)

    tsp = StratifiedGroupKFold(
        5,
        shuffle=True,
        random_state=random_seed,
    )

    split_iterator = tsp.split(
        df,
        y=df["target"],
        groups=df["patient_id"],
    )


    for fold_n, (
        train_index,
        val_index,
    ) in enumerate(split_iterator):

        if fold_n not in wanted_folds:
            continue

        print("\n" + "=" * 90)
        print(
            f"SEED {random_seed} / "
            f"FOLD {fold_n}"
        )
        print("=" * 90)


        train_slice_x = (
            df.iloc[train_index][model_features]
            .reset_index(drop=True)
        )

        val_slice_x = (
            df.iloc[val_index][model_features]
            .reset_index(drop=True)
        )

        train_slice_y = (
            df.iloc[train_index]["target"]
            .reset_index(drop=True)
        )

        val_slice_y = (
            df.iloc[val_index]["target"]
            .reset_index(drop=True)
        )


        print(
            "Train:",
            len(train_slice_y),
            "Pos:",
            int(train_slice_y.sum()),
            "Neg:",
            int(
                len(train_slice_y)
                - train_slice_y.sum()
            )
        )

        print(
            "Valid:",
            len(val_slice_y),
            "Pos:",
            int(val_slice_y.sum()),
            "Neg:",
            int(
                len(val_slice_y)
                - val_slice_y.sum()
            )
        )


        # ====================================================
        # Original sampling
        #
        # RandomOverSampler  -> ratio 0.003
        # RandomUnderSampler -> ratio 0.01
        # ====================================================

        sampler = Pipeline([
            (
                "sampler_1",
                RandomOverSampler(
                    sampling_strategy=0.003,
                    random_state=random_seed,
                ),
            ),
            (
                "sampler_2",
                RandomUnderSampler(
                    sampling_strategy=0.01,
                    random_state=random_seed,
                ),
            ),
        ])

        train_slice_x, train_slice_y = (
            sampler.fit_resample(
                train_slice_x,
                train_slice_y,
            )
        )


        # ====================================================
        # Baseline 4B: winner-style Gaussian noise
        #
        # IMPORTANT:
        # - TRAIN only
        # - AFTER ROS/RUS
        # - Validation remains unchanged
        # - Inference remains unchanged
        # ====================================================

        for col in NOISE_FEATURES:
            train_slice_x[col] = (
                train_slice_x[col]
                + np.random.normal(
                    loc=0.0,
                    scale=NOISE_STD,
                    size=train_slice_x.shape[0],
                )
            )


        print(
            "After sampling:",
            len(train_slice_y),
            "Pos:",
            int(train_slice_y.sum()),
            "Neg:",
            int(
                len(train_slice_y)
                - train_slice_y.sum()
            )
        )


        # ====================================================
        # CatBoost
        # ====================================================

        params = dict(CB_PARAMS)

        params["random_state"] = (
            random_seed
        )


        clf = CatBoostClassifier(
            **params
        )


        train_pool = Pool(
            train_slice_x,
            train_slice_y.values,
            cat_features=cat_cols,
        )

        val_pool = Pool(
            val_slice_x,
            val_slice_y.values,
            cat_features=cat_cols,
        )


        clf.fit(
            train_pool,
            eval_set=val_pool,
            verbose=False,
        )


        preds = clf.predict_proba(
            val_slice_x
        )[:, 1]


        pauc = custom_metric_raw(
            preds,
            val_slice_y.values,
        )

        auroc = roc_auc_score(
            val_slice_y.values,
            preds,
        )


        print(
            f"Best iteration: "
            f"{clf.get_best_iteration()}"
        )

        print(
            f"Validation AUROC: "
            f"{auroc:.8f}"
        )

        print(
            f"Validation pAUC : "
            f"{pauc:.8f}"
        )


        # ====================================================
        # Save model
        # ====================================================

        model_path = (
            MODEL_DIR /
            f"catboost_seed{random_seed}"
            f"_fold{fold_n}.cbm"
        )

        clf.save_model(
            str(model_path)
        )


        # ====================================================
        # Save OOF
        # ====================================================

        oof_tmp = pd.DataFrame({
            "isic_id":
                df.iloc[val_index][
                    "isic_id"
                ].values,

            "patient_id":
                df.iloc[val_index][
                    "patient_id"
                ].values,

            "target":
                val_slice_y.values,

            "prediction":
                preds,

            "random_seed":
                random_seed,

            "fold_n":
                fold_n,
        })


        oof_path = (
            OOF_DIR /
            f"oof_seed{random_seed}"
            f"_fold{fold_n}.parquet"
        )

        oof_tmp.to_parquet(
            oof_path,
            index=False,
        )


        all_oof.append(
            oof_tmp
        )


        all_scores.append({
            "random_seed":
                random_seed,

            "fold_n":
                fold_n,

            "n_train":
                len(train_index),

            "n_valid":
                len(val_index),

            "valid_pos":
                int(
                    val_slice_y.sum()
                ),

            "auroc":
                auroc,

            "pauc":
                pauc,

            "best_iteration":
                clf.get_best_iteration(),
        })


        # Save progress after every model
        score_df = pd.DataFrame(
            all_scores
        )

        score_df.to_csv(
            LOG_DIR /
            "scores_progress.csv",
            index=False,
        )


# ============================================================
# Results
# ============================================================

scores_df = pd.DataFrame(
    all_scores
)

oof_df = pd.concat(
    all_oof,
    ignore_index=True,
)


scores_path = (
    LOG_DIR /
    "scores.csv"
)

scores_df.to_csv(
    scores_path,
    index=False,
)


oof_all_path = (
    OOF_DIR /
    "oof_all_models.parquet"
)

oof_df.to_parquet(
    oof_all_path,
    index=False,
)


print("\n" + "=" * 90)
print("TRAINING COMPLETE")
print("=" * 90)

print(
    "Models trained:",
    len(scores_df)
)

print(
    "Mean validation pAUC:",
    scores_df["pauc"].mean()
)

print(
    "Mean validation AUROC:",
    scores_df["auroc"].mean()
)


# ============================================================
# Additional diagnostic:
# average predictions across seeds for each lesion
#
# This is NOT Cell 26's reported metric.
# ============================================================

if len(seeds) == 9 and len(wanted_folds) == 5:

    averaged_oof = (
        oof_df
        .groupby(
            [
                "isic_id",
                "patient_id",
                "target",
            ],
            as_index=False,
        )
        ["prediction"]
        .mean()
    )


    pooled_auroc = roc_auc_score(
        averaged_oof["target"],
        averaged_oof["prediction"],
    )

    pooled_pauc = custom_metric_raw(
        averaged_oof["prediction"].values,
        averaged_oof["target"].values,
    )


    averaged_oof.to_parquet(
        OOF_DIR /
        "oof_mean_across_9_seeds.parquet",
        index=False,
    )


    print("\nAdditional diagnostic")
    print(
        "Rows after seed averaging:",
        len(averaged_oof)
    )

    print(
        "Seed-averaged OOF AUROC:",
        pooled_auroc
    )

    print(
        "Seed-averaged OOF pAUC :",
        pooled_pauc
    )


print("\nSaved:")
print(scores_path)
print(oof_all_path)
