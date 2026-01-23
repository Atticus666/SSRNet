# MiniRec: Multi-Model CTR Prediction Framework

TensorFlow implementation of **state-of-the-art recommendation models** for Click-Through Rate (CTR) prediction with unified training pipeline.

## Features

- **7 SOTA Models**: AutoInt, DCN-v2, RankMixer, Wukong, AutoFIS, AFN, Ours:SSRNet
- **Multi-Dataset Support**: Avazu, Criteo, Alibaba
- **Optimized Performance**: Efficient tf.data pipelines and batch processing
- **Production Ready**: Checkpointing, early stopping, comprehensive metrics

## Supported Models

### 1. **AutoInt** - Automatic Feature Interaction Learning via Self-Attentive Neural Networks
> CIKM 2019 | [Paper](https://arxiv.org/abs/1810.11921)

Multi-head self-attention for automatic feature interaction learning with residual connections.

### 2. **DCN-v2** - Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems
> WWW 2021 | [Paper](https://arxiv.org/abs/2008.13535)

Low-rank mixture of expert cross layers for explicit feature crossing.

### 3. **RankMixer** - Scaling Up Ranking Models in Industrial Recommenders
> 2025 | [Paper](https://arxiv.org/abs/2507.15551)

Advanced features mixer method for scaling Up ranking models.

### 4. **Wukong** - Towards a Scaling Law for Large-Scale Recommendation
> 2024 | [Paper](https://arxiv.org/pdf/2403.02545)

Factorization Machine Block (FMB) with synergistic upscaling strategy.

### 5. **AutoFIS** - Automatic Feature Interaction Selection
> KDD 2020 | [Paper](https://arxiv.org/abs/2003.11235)

Automatic feature interaction selection via two-stage architecture search.

### 6. **AFN** - Learning Adaptive-Order Feature Interactions
> AAAI 2020 | [Paper](https://arxiv.org/abs/1909.03276)

Logarithmic transformation network for adaptive feature interaction learning.

### 7. **SSRNet** - Explicit Sparsity for Scalable Recommendation
Ours: Explicit Sparsity for efficient scalable recommendation.


## Requirements

```bash
Python >= 3.10
TensorFlow == 2.11.*
scikit-learn >= 1.7.2
pandas >= 2.3.3
numpy >= 1.26.4
```

## Simple Environment

The project uses conda for dependency management. Install using the provided environment file:

```bash
# Create conda environment
conda env create -f environment_gpu.yml

# Activate environment  
conda activate tf211gpu

# Verify installation
python -c "import tensorflow as tf; print('TF version:', tf.__version__)"
```

## Manually Installation

If Simple Environment can not work, you can manually install the dependencies:

1. **Clone the repository**:
```bash
git clone <repository-url>
cd MiniRec
```

2. **Install dependencies**:
```bash
pip install tensorflow==2.11.* scikit-learn pandas numpy
```

3. **Verify installation**:
```bash
python -c "import tensorflow as tf; print('TF version:', tf.__version__)"
```


## Project Structure

```
MiniRec/
├── models/                     # Model implementations
│   ├── ssrnet/                # SSRNet module components
│   │   ├── block_t*.py        # Various SSRNet block implementations (t18/t18a/t21)
│   │   └── monitoring_callback.py  # Training monitoring
│   ├── model_autoint.py       # AutoInt
│   ├── model_dcn_v2.py        # DCN-v2
│   ├── model_rankmixer.py     # RankMixer
│   ├── model_ssrnet.py        # SSRNet
│   ├── model_ssrnet_t.py      # SSRNet Transformer
│   ├── model_wukong.py        # Wukong
│   ├── model_autofis.py       # AutoFIS
│   ├── model_afn.py           # AFN
│   ├── model_deepfm.py        # DeepFM
│   └── model_ffn.py           # FFN
├── runners/                    # Training scripts
│   ├── train_autoint.py       # AutoInt training
│   ├── train_dcn_v2.py        # DCN-v2 training
│   ├── train_rankmixer.py     # RankMixer training
│   ├── train_ssrnet.py        # SSRNet training
│   ├── train_ssrnet_t.py      # SSRNet Transformer training
│   ├── train_wukong.py        # Wukong training
│   ├── train_autofis.py       # AutoFIS training
│   ├── train_afn.py           # AFN training
│   ├── train_deepfm.py        # DeepFM training
│   └── train_ffn.py           # FFN training
├── dataprocess/               # Data processing
│   ├── avazu_optimized.py     # Avazu processor
│   ├── criteo_optimized.py    # Criteo processor
│   ├── aliccp_optimized.py    # Ali-CCP processor
│   ├── kdd2012_optimized.py   # KDD2012 processor
│   ├── kfold_split.py         # K-fold splitting
│   ├── base.py                # Base data processor
│   └── config.py              # Data processing config
├── utils/                     # Utility functions
│   ├── callbacks.py           # Training callbacks
│   ├── data_loader.py         # Data loading utilities
│   ├── metrics.py             # Metrics calculation
│   └── profiler.py            # Model profiling
├── data/                      # Dataset storage
│   ├── Avazu/                 # Avazu dataset
│   └── Criteo/                # Criteo dataset
├── checkpoint/                # Model checkpoints
├── run.sh                     # Training execution script
├── run_data.sh                # Data processing script
├── environment.yml            # Conda environment (CPU)
├── environment_gpu.yaml       # Conda environment (GPU)
└── README.md                  # Project documentation
```

## Quick Start

### 1. **Data Preparation**

#### **Criteo Dataset**:
```bash
# Download and extract Criteo dataset (TSV format)
# Place the data file as: ./data/Criteo/train.txt

# Step 1: Run preprocessing with optimized processor
python -c "
from dataprocess.criteo_optimized import preprocess_criteo_dataset
feature_size = preprocess_criteo_dataset(
    source_path='./data/Criteo/train.txt',
    output_path='./data/Criteo/',
    verbose=1
)
print(f'Criteo processing completed with {feature_size:,} total features')
"

# Step 2: Create k-fold splits
python -c "
from dataprocess.kfold_split import create_stratified_splits
from dataprocess.config import CriteoConfig
config = CriteoConfig(data_path='./data/Criteo/')
create_stratified_splits(config)
"

# Step 3: Scale numerical features using log scaling
python -c "
from dataprocess.base import DataScaler
from dataprocess.config import CriteoConfig
config = CriteoConfig(data_path='./data/Criteo/')
numerical_columns = list(range(13))  
DataScaler.scale_data_parts(config, numerical_columns, scale_method='log')
"

```

#### **Avazu Dataset**:
```bash
# Download Avazu dataset from Kaggle
# Place the data file as: ./data/Avazu/train.csv

python -c "
from dataprocess.avazu_optimized import preprocess_avazu_dataset
preprocess_avazu_dataset(
    './data/Avazu/train.csv',
    './data/Avazu/'
)
"

# Step 2: Create k-fold splits  
python -c "
from dataprocess.kfold_split import create_stratified_splits
from dataprocess.config import AvazuConfig
config = AvazuConfig(data_path='./data/Avazu/')
create_stratified_splits(config)
"
# Note: Avazu dataset contains all categorical features, no scaling needed
```

#### **Alibaba Dataset**:
```bash
# Download Alibaba dataset from Tianchi
# Dataset URL: https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408
# Place the data files as:
#   - ./data/Aliccp/sample_skeleton_train.csv (training skeleton file)
#   - ./data/Aliccp/common_features_train.csv (training common features)
#   - ./data/Aliccp/sample_skeleton_test.csv (test skeleton file)
#   - ./data/Aliccp/common_features_test.csv (test common features)

# Step 1: Run preprocessing with optimized processor
python -c "
from dataprocess.aliccp_optimized import preprocess_aliccp_dataset
feature_size = preprocess_aliccp_dataset(
    source_path='./data/Aliccp/',
    output_path='./data/Aliccp/',
    process_test=False,
    use_log_scaling=True,
    verbose=1
)
print(f'Alibaba processing completed with {feature_size:,} total features')
"

# Step 2: Create k-fold splits
python -c "
from dataprocess.kfold_split import create_stratified_splits
from dataprocess.config import AliccpConfig
config = AliccpConfig(data_path='./data/Aliccp/')
create_stratified_splits(config)
"

```

#### **Expected Data Structure After Processing**:
```bash
# Expected data structure
data/
├── Avazu/
│   ├── train_i.txt         # Feature indices (text format)
│   ├── train_x.txt         # Feature values (text format)
│   ├── train_y.txt         # Labels (text format)
│   ├── feature_size.npy    # Total feature size (IMPORTANT!)
│   ├── fold_index.pkl      # K-fold indices (pickle format)
│   ├── fold_index.npy      # K-fold indices (numpy format)
│   ├── part1/              # Fold 1 data (test set)
│   │   ├── train_i.npy     # Feature indices
│   │   ├── train_x.npy     # Feature values
│   │   └── train_y.npy     # Labels
│   ├── part2/              # Fold 2 data (validation set)
│   └── part3-10/           # Folds 3-10 (training sets)
├── Criteo/
│   ├── train_i.txt         # Feature indices (text format)
│   ├── train_x.txt         # Feature values (text format)
│   ├── train_y.txt         # Labels (text format)
│   ├── train_examples.txt  # Training examples metadata
│   ├── feature_size.npy    # Total feature size (IMPORTANT!)
│   ├── fold_index.pkl      # K-fold indices (pickle format)
│   ├── fold_index.npy      # K-fold indices (numpy format)
│   ├── part1/              # Fold 1 data (test set)
│   │   ├── train_i.npy     # Feature indices
│   │   ├── train_x.npy     # Feature values
│   │   ├── train_x2.npy    # Scaled feature values
│   │   └── train_y.npy     # Labels
│   ├── part2/              # Fold 2 data (validation set)
│   └── part3-10/           # Folds 3-10 (training sets)
├── Aliccp/                 # Ali-CCP dataset
│   ├── train_i.txt         # Feature indices (text format)
│   ├── train_x.txt         # Feature values (text format)
│   ├── train_y.txt         # Combined labels (text format)
│   ├── train_y_click.txt   # Click labels
│   ├── train_y_purchase.txt # Purchase labels
│   ├── ctrcvr_enum.pkl     # CTR/CVR enumeration
│   ├── feature_size.npy    # Total feature size
│   ├── fold_index.pkl      # K-fold indices
│   ├── part1/              # Fold 1 data (test set)
│   │   ├── train_i.npy
│   │   ├── train_x.npy
│   │   └── train_y.npy
│   ├── part2/              # Fold 2 data (validation set)
│   └── part3-10/           # Folds 3-10 (training sets)


# Note: 
# - feature_size.npy is REQUIRED for model training (stores total feature dimensions)
# - fold_index.pkl stores the indices for each fold (saved as pickle, not npy)
# - train_x2.npy contains scaled feature values (recommended for Criteo dataset)
```

### 2. **Training Models**

```bash
# Using run.sh (recommended) 
# Please specify the model trainer params before running. 
bash run.sh

# SSRNet Random 
"""
python runners/train_ssrnet.py \
    --data avazu \
    --data_path ./data/ \
    --embedding_size 16 \
    --b_matrices 16 16 \
    --d_mid_cols 128 64 \
    --out_units 128 128 \
    --num_hidden_layers 1 \
    --batch_size 1024 \
    --epoch 3 \
    --learning_rate 0.001 \
    --optimizer_type adam \
    --run_times 1 \
    --save_path ./checkpoint/avazu_ssrnet_experiment/ \
    --is_save true \
    --verbose 1
"""

# SSRNet Trainable 
"""
python runners/train_ssrnet_t.py \
    --data avazu \
    --data_path ./data/ \
    --block_version t21 \
    --embedding_size 16 \
    --tokennum_list 8 8 \
    --hidden_unit_list 128 128 \
    --top_k_list 128 128 \
    --out_unit_list 128 128 \
    --iterations 5 \
    --alpha_init 0.1 0.1 \
    --scale_init 1.0 1.0 \
    --use_ssr_linear False \
    --use_block_dense True \
    --use_block_mean_pooling False \
    --dropout_rates 0.0 0.0 0.0 \
    --l2_reg 0.0 \
    --batch_size 1024 \
    --epoch 3 \
    --optimizer_type adam \
    --learning_rate 0.001 \
    --num_runs 1 \
    --save_path ./checkpoint/avazu_ssrnet_t_experiment/ \
    --is_save 0 \
    --verbose 1
"""
```

## Supported Datasets

| Dataset | Features | Notes |
|---------|----------|--------|
| **Avazu** | 23 categorical | No scaling needed |
| **Criteo** | 13 numerical + 26 categorical | Requires log scaling |
| **Ali-CCP** | 24 features | Requires log scaling |

**Datasets**:
- [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction), [Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge), [Alibaba](https://tianchi.aliyun.com/dataset/408
)


## Monitoring & Logging
Training logs: `./logs/{datetime}_{dataset}_{model}.log`  
Model checkpoints: `./checkpoint/{experiment_name}/`


---

