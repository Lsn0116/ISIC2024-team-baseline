# ISIC 2024 Baseline Reproduction

本 repository 提供本專題使用的 **ISIC 2024 Skin Cancer Detection baseline reproduction**。

目前保留兩組主要 baseline：

| Baseline | Description | Private pAUC | Public pAUC |
|---|---|---:|---:|
| **Baseline 4D** | Best-performing reproduced CatBoost variant | **0.17190** | 0.18547 |
| **Baseline 5B** | Paper-style CatBoost + LightGBM + XGBoost rank ensemble | 0.17069 | **0.18585** |

原始競賽 winner 的 Private pAUC 約為 **0.1726**。

本專題後續將以這些 baseline 為基礎，研究 **class imbalance handling strategies**。

---

# 1. Repository Structure

```text
ISIC2024_team_baseline/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── train_features.parquet
│   ├── feature_cols.json
│   ├── final_feature_cols.json
│   ├── cat_cols.json
│   └── columns_to_drop.json
│
├── baseline_4D/
│   ├── train_catboost_4d.py
│   └── kaggle_submission_4d.ipynb
│
├── baseline_5B/
│   ├── train_catboost.py
│   ├── train_lightgbm.py
│   ├── train_xgboost.py
│   └── kaggle_submission_5b.ipynb
│
└── experiments/
```

---

# 2. Training Data

所有 downstream GBT experiments 共用：

```text
data/train_features.parquet
```

目前資料規格：

- 401,059 training samples
- 269 total columns
- 266 candidate feature columns
- 242 final model features
- 47 categorical features

其中已包含固定的 CNN-derived prediction features，例如：

- External EVA predictions
- Competition EVA predictions
- EdgeNeXt predictions

因此如果研究的是 **downstream class imbalance strategy**，不需要重新訓練 CNN，也不需要重新產生 CNN predictions。

所有組員必須使用同一份：

```text
data/train_features.parquet
```

以確保不同實驗之間可以公平比較。

---

# 3. Clone Repository

此 repository 使用 **Git LFS** 儲存約 654 MB 的：

```text
data/train_features.parquet
```

因此 clone 前請先確認電腦已安裝 Git LFS。

## 3.1 Check Git LFS

```bash
git lfs version
```

如果沒有安裝，可以使用：

```bash
conda install -c conda-forge git-lfs -y
```

接著：

```bash
git lfs install
```

---

## 3.2 Clone

```bash
git clone <REPOSITORY_URL>
cd ISIC2024_team_baseline
```

如果大型檔案沒有自動下載：

```bash
git lfs pull
```

確認 training features：

```bash
ls -lh data/train_features.parquet
```

檔案大小約為：

```text
654 MB
```

---

# 4. Environment Setup

建議使用 Python 3.11。

建立 conda environment：

```bash
conda create -n isic2024 python=3.11 -y
conda activate isic2024
```

安裝套件：

```bash
pip install -r requirements.txt
```

如果使用 GPU training，請另外確認：

- NVIDIA Driver
- CUDA
- CatBoost GPU support
- LightGBM GPU support
- XGBoost CUDA support

皆可正常使用。

---

# 5. Quick Environment Test

正式訓練 45 個 models 前，建議先確認程式可以正常執行。

---

# 6. Baseline 4D

## 6.1 Description

Baseline 4D 是目前 reproduction 中 **Private leaderboard 表現最佳的版本**。

主要流程：

```text
Fixed 242 features
        ↓
Random Oversampling
        ↓
Random Undersampling
        ↓
Gaussian noise on CNN prediction features
        ↓
CatBoost
        ↓
9 seeds × 5 folds
        ↓
45 CatBoost models
        ↓
Raw probability averaging
```

Leaderboard result：

```text
Private pAUC = 0.17190
Public pAUC  = 0.18547
```

---

## 6.2 Test One Fold First

第一次執行時，建議先測試一個 seed / fold：

```bash
python baseline_4D/train_catboost_4d.py \
    --seed 1 \
    --fold 0
```

確認：

- parquet 可正常讀取
- GPU 可正常使用
- CatBoost 可正常 training
- model 可正常儲存

之後再跑完整 baseline。

---

## 6.3 Full Training

```bash
python baseline_4D/train_catboost_4d.py
```

預設：

```text
9 seeds × 5 folds = 45 CatBoost models
```

---

## 6.4 Output

Models：

```text
outputs/baseline_4D/catboost/models/
```

OOF：

```text
outputs/baseline_4D/catboost/oof/
```

Logs：

```text
outputs/baseline_4D/catboost/logs/
```

---

# 7. Baseline 5B — Paper-style 3-GBT Ensemble

Baseline 5B 用來作為較接近 paper architecture 的 reproduction baseline。

架構：

```text
                     ┌── CatBoost ×45
242 fixed features ──┼── LightGBM ×45
                     └── XGBoost ×45
                              ↓
                     Model-wise ranking
                              ↓
                     Family-wise averaging
                              ↓
                     3-family rank ensemble
```

總共：

```text
45 × 3 = 135 GBT models
```

Leaderboard result：

```text
Private pAUC = 0.17069
Public pAUC  = 0.18585
```

---

# 8. Train Baseline 5B

三個 model family 分開訓練。

---

## 8.1 CatBoost

先測一個 fold：

```bash
python baseline_5B/train_catboost.py \
    --seed 1 \
    --fold 0
```

完整訓練：

```bash
python baseline_5B/train_catboost.py
```

輸出：

```text
outputs/baseline_5B/catboost/models/
```

---

## 8.2 LightGBM

先測一個 fold：

```bash
python baseline_5B/train_lightgbm.py \
    --seed-index 1 \
    --fold 0
```

完整訓練：

```bash
python baseline_5B/train_lightgbm.py
```

輸出：

```text
outputs/baseline_5B/lightgbm/models/
```

---

## 8.3 XGBoost

先測一個 fold：

```bash
python baseline_5B/train_xgboost.py \
    --seed-index 1 \
    --fold 0
```

完整訓練：

```bash
python baseline_5B/train_xgboost.py
```

輸出：

```text
outputs/baseline_5B/xgboost/models/
```

---

# 9. Which Baseline Should Be Used?

本專題保留兩個 baseline，各有不同用途。

## Baseline 5B

作為：

```text
Paper-style reproduction baseline
```

因為它包含：

```text
CatBoost
+
LightGBM
+
XGBoost
+
rank ensemble
```

比較忠實對應 paper 中的 3-GBT architecture。

---

## Baseline 4D

作為：

```text
Best-performing reproduced variant
```

目前在 Private leaderboard 上表現最好：

```text
Private pAUC = 0.17190
```

因此進行新的 class imbalance ablation 時，建議先從 **Baseline 4D / CatBoost branch** 開始。

確認新方法有效後，再放回完整 5B 3-GBT ensemble 做 final validation。

---

# 10. Class Imbalance Experiments

不要直接修改原始 baseline scripts。

例如要建立 Patient-Aware Hard Negative Sampling：

```bash
cp baseline_4D/train_catboost_4d.py \
   experiments/train_catboost_PA_HNS.py
```

之後修改：

```text
experiments/train_catboost_PA_HNS.py
```

---

## 10.1 Controlled Experiment Rule

比較 class imbalance methods 時，原則上應固定：

- training dataset
- CNN-derived representations
- 242 final features
- patient-level split
- number of folds
- random seeds
- model hyperparameters
- evaluation metric

主要修改：

```text
class imbalance / sampling strategy
```

例如：

```text
Original ROS/RUS
vs
Hard Negative Sampling
vs
Patient-Aware Hard Negative Sampling
vs
Diversity-aware Negative Sampling
vs
Class Weighting
```

這樣才能較合理地將 performance difference 歸因於 imbalance handling strategy。

---

# 11. Why CNN Is Fixed

本專題目前先固定原始 CNN-derived representation。

原因是：

```text
Fixed CNN representation
        ↓
Same 242 features
        ↓
Only change imbalance strategy
```

可以降低額外變因。

如果同時重新訓練 CNN、修改 augmentation、修改 sampling 和修改 GBT，就很難判斷 performance improvement 到底來自哪個部分。

因此主要研究問題為：

> 在固定 image representation 的情況下，不同 downstream class imbalance handling strategies 是否可以改善極度不平衡的 skin cancer detection？

後續若時間足夠，可以再延伸至 CNN-level imbalance-aware training。

---

# 12. Kaggle Hidden-test Inference

Local GBT training 不需要重新跑 CNN。

但是 Kaggle hidden test 沒有事先計算好的 CNN predictions，因此 hidden-test inference 還是需要：

- original winner CNN weights
- External EVA
- Competition EVA folds
- EdgeNeXt folds
- preprocessing artifacts
- CNN scaling statistics

目前共用 Kaggle Dataset：

```text
lsn0000/isic2024-winner-original-baseline4a
```

所有組員在 Kaggle submission notebook 都需要加入這份 Dataset 作為 Input。

如果該 Dataset 為 private Dataset，請先確認自己的 Kaggle account 有存取權限。

---

# 13. Baseline 4D Kaggle Submission

使用：

```text
baseline_4D/kaggle_submission_4d.ipynb
```

Notebook 內分成兩個重要路徑。

---

## 13.1 Shared Assets

```python
BASELINE_DIR
```

負責提供：

- CNN weights
- preprocessing
- metadata
- scaling statistics

這部分通常不需要修改。

---

## 13.2 GBT Model Directory

```python
USER_MODEL_DIR
```

預設：

```python
USER_MODEL_DIR = BASELINE_DIR
```

因此如果只是要重現目前 baseline leaderboard result，不需要修改。

---

# 14. Baseline 5B Kaggle Submission

使用：

```text
baseline_5B/kaggle_submission_5b.ipynb
```

同樣：

```python
BASELINE_DIR
```

負責 shared CNN / preprocessing assets。

```python
USER_MODEL_DIR
```

負責目前 experiment 所使用的 GBT models。

預設：

```python
USER_MODEL_DIR = BASELINE_DIR
```

即可重現目前 baseline。

---

# 15. Upload Your Own Models to Kaggle

當新的 imbalance experiment 訓練完成後，需要將 models 上傳為自己的 Kaggle Dataset。

注意：

**local training output 的資料夾名稱與 Kaggle inference notebook 所期待的資料夾名稱不完全相同。**

因此上傳前需要重新整理一次資料夾。

---

# 16. Package Baseline 4D Models for Kaggle

Local models：

```text
outputs/baseline_4D/catboost/models/
```

Kaggle notebook 預期：

```text
<your-kaggle-dataset>/
└── catboost_4d/
```

因此可以建立：

```bash
rm -rf kaggle_models_4d
mkdir -p kaggle_models_4d/catboost_4d
```

複製 models：

```bash
cp outputs/baseline_4D/catboost/models/*.cbm \
   kaggle_models_4d/catboost_4d/
```

確認：

```bash
find kaggle_models_4d/catboost_4d \
  -name '*.cbm' \
  | wc -l
```

完整 baseline 應為：

```text
45
```

---

# 17. Package Baseline 5B Models for Kaggle

建立：

```bash
rm -rf kaggle_models_5b

mkdir -p kaggle_models_5b/catboost_4d
mkdir -p kaggle_models_5b/lightgbm_5
mkdir -p kaggle_models_5b/xgboost_5
```

CatBoost：

```bash
cp outputs/baseline_5B/catboost/models/*.cbm \
   kaggle_models_5b/catboost_4d/
```

LightGBM：

```bash
cp outputs/baseline_5B/lightgbm/models/*.joblib \
   kaggle_models_5b/lightgbm_5/
```

XGBoost：

```bash
cp outputs/baseline_5B/xgboost/models/*.joblib \
   kaggle_models_5b/xgboost_5/
```

確認：

```bash
find kaggle_models_5b/catboost_4d \
  -name '*.cbm' \
  | wc -l
```

應為：

```text
45
```

確認 LightGBM：

```bash
find kaggle_models_5b/lightgbm_5 \
  -name '*.joblib' \
  | wc -l
```

應為：

```text
45
```

確認 XGBoost：

```bash
find kaggle_models_5b/xgboost_5 \
  -name '*.joblib' \
  | wc -l
```

應為：

```text
45
```

---

# 18. Create Kaggle Dataset

先確認已安裝並登入 Kaggle CLI。

例如：

```bash
kaggle --version
```

---

## Example Dataset Metadata

在要上傳的資料夾建立：

```text
dataset-metadata.json
```

例如：

```json
{
  "title": "ISIC 2024 My Experiment Models",
  "id": "<YOUR_KAGGLE_USERNAME>/isic2024-my-experiment-models",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
```

請將：

```text
<YOUR_KAGGLE_USERNAME>
```

及 dataset name 改成自己的。

---

## Upload

例如 4D：

```bash
kaggle datasets create \
  -p kaggle_models_4d \
  --dir-mode zip
```

或 5B：

```bash
kaggle datasets create \
  -p kaggle_models_5b \
  --dir-mode zip
```

建立完成後，到 Kaggle Dataset 頁面確認 models 已成功存在。

---

# 19. Use Your Own Models in Submission Notebook

先在 Kaggle notebook：

```text
Add Input
```

加入：

1. Shared baseline asset Dataset
2. 自己新建立的 model Dataset
3. ISIC 2024 competition data

接著修改：

```python
USER_MODEL_DIR = Path(
    "/kaggle/input/datasets/<username>/<dataset-name>"
)
```

例如：

```python
USER_MODEL_DIR = Path(
    "/kaggle/input/datasets/example-user/isic2024-pahns-models"
)
```

不要修改：

```python
BASELINE_DIR
```

因為它仍負責共用 CNN weights / preprocessing。

---

# 20. Expected 4D Kaggle Model Structure

```text
<your-kaggle-dataset>/
└── catboost_4d/
    ├── catboost_seed1_fold0.cbm
    ├── catboost_seed1_fold1.cbm
    ├── ...
    └── 45 models
```

---

# 21. Expected 5B Kaggle Model Structure

```text
<your-kaggle-dataset>/
├── catboost_4d/
│   └── 45 CatBoost models
│
├── lightgbm_5/
│   └── 45 LightGBM models
│
└── xgboost_5/
    └── 45 XGBoost models
```

如果 model 數量不完整，submission notebook 會停止並顯示 error。

---

# 22. Run Kaggle Submission Notebook

在 Kaggle：

```text
Run All
```

正常完成後，應產生：

```text
/kaggle/working/submission.csv
```

確認：

- no NaN
- no duplicated `isic_id`
- test row count 正確

再選擇：

```text
Submit to Competition
```

取得 leaderboard score。

---

# 23. Baseline Reference Results

目前 reproduction results：

| Version | Main Difference | Private pAUC | Public pAUC |
|---|---|---:|---:|
| 4A | Original CNN + CatBoost | 0.17168 | 0.18517 |
| 4C | + random_strength | 0.17166 | 0.18517 |
| **4D** | + Gaussian noise + random_strength | **0.17190** | 0.18547 |
| 5A | CatBoost model-wise rank | 0.17183 | 0.18545 |
| **5B** | CatBoost + LightGBM + XGBoost rank ensemble | 0.17069 | **0.18585** |

本專題主要保留：

```text
5B = Paper-style reproduction baseline
4D = Best-performing reproduced variant
```

---

# 24. Recommended Experimental Workflow

建議每個新的 class imbalance 方法遵循：

```text
Step 1
Clone repository

        ↓

Step 2
Run one fold of Baseline 4D

        ↓

Step 3
Confirm environment is correct

        ↓

Step 4
Copy Baseline 4D training script

        ↓

Step 5
Modify imbalance strategy only

        ↓

Step 6
Compare OOF pAUC / AUROC

        ↓

Step 7
Train full 45 CatBoost models

        ↓

Step 8
Upload models to Kaggle

        ↓

Step 9
Run hidden-test submission

        ↓

Step 10
If the method is promising,
apply it to the full 5B 3-GBT ensemble
```

---

# 25. Experimental Naming Recommendation

例如 Patient-Aware Hard Negative Sampling：

```text
experiments/
└── train_catboost_PA_HNS.py
```

Output 建議另外存：

```text
outputs/
└── experiments/
    └── PA_HNS/
```

避免覆蓋 baseline models。

---

# 26. Important Notes

## Do not overwrite baseline scripts

請將新的 experiment 放在：

```text
experiments/
```

不要直接修改：

```text
baseline_4D/
baseline_5B/
```

這兩個資料夾應視為 locked reference baseline。

---

## Keep patient grouping unchanged

ISIC 2024 中同一 patient 可能有多張 lesion images。

因此 train / validation split 應維持原本的：

```text
StratifiedGroupKFold
```

並以：

```text
patient_id
```

作為 grouping unit。

避免同一 patient 同時出現在 training 與 validation。

---

## Do not tune on Private leaderboard

Private leaderboard 應作為最終驗證，而不是反覆調參依據。

新的 imbalance strategy 應先根據：

- OOF pAUC
- OOF AUROC
- sensitivity
- specificity
- patient-level metrics

進行比較。

確認方法後，再做 Kaggle hidden-test submission。

---

# 27. Project Goal

目前 baseline reconstruction 已經接近原始 winner performance。

接下來的核心研究問題不再只是追求 reproduction score，而是：

> Can class-imbalance handling improve skin cancer detection under extreme imbalance while preserving clinically important benign diversity and patient-level generalization?

後續主要研究方向包含：

- Patient-Aware Hard Negative Sampling
- Benign Diversity-Aware Sampling
- Strong / Weak Label Weighting
- Class Weighting
- Other imbalance-aware strategies

---

# 28. Summary

如果只是要確認 baseline：

### 4D

```bash
python baseline_4D/train_catboost_4d.py \
    --seed 1 \
    --fold 0
```

完整：

```bash
python baseline_4D/train_catboost_4d.py
```

---

### 5B

CatBoost：

```bash
python baseline_5B/train_catboost.py
```

LightGBM：

```bash
python baseline_5B/train_lightgbm.py
```

XGBoost：

```bash
python baseline_5B/train_xgboost.py
```

---

如果要做新 experiment：

```bash
cp baseline_4D/train_catboost_4d.py \
   experiments/train_catboost_MY_METHOD.py
```

修改 imbalance strategy，完成 training 後打包 models，上傳 Kaggle Dataset，再使用：

```text
baseline_4D/kaggle_submission_4d.ipynb
```

或：

```text
baseline_5B/kaggle_submission_5b.ipynb
```

進行 hidden-test evaluation。
