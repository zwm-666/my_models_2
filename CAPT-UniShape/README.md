# CAPT-UniShape

**Condition-Aware Prototype Transport for PEMFC Fault Classification**

> 当前真实论文实验请优先阅读 `docs/README_训练测试绘图.md` 和 `docs/KAGGLE_运行说明.md`。这两个文件中的命令已经按四比例基准对比、消融实验和一组噪声鲁棒性实验整理，并默认不启用类别感知滑窗补偿。

A multi-modal deep learning framework for Proton Exchange Membrane Fuel Cell (PEMFC) fault classification. CAPT-UniShape extends UniShape (AAAI 2026) with condition-aware attention pooling, cross-modal shape interaction, and a Condition-Aware Prototype Transport (CAPT) classification head.

> 2026-04 official redesign: the active model code is now the top-level
> `models/` package.  It wraps the official `ZLiu21/UniShape` code vendored in
> `external/unishape/` and implements `Official-CAPT-UniShape-RBF-KANFusion`
> plus the fair `Official-CAPT-UniShape-KANFusion-NoRBF` control.  The old
> lightweight `src/models/reskan` CNN prototype was removed and only a legacy
> compatibility factory remains.

## Official UniShape RBF/KANFusion Quick Start

Run the required no-data forward demo:

```bash
python demo_forward.py
```

The demo builds both variants with fake tensors (`B=8`, `C_op=6`, `T=256`,
`C_eis=4`, `F=128`, `D_cond=10`, `num_classes=3`) and writes
`results/official_unishape_demo/demo_forward_metrics.json` with logits shape,
total loss, CE loss, transport loss, KAN regularisation, parameter count and
forward time.

Train on real data saved as an NPZ file:

```bash
python train.py --config configs/rbf_kanfusion.yaml --data data/processed/your_multisource_dataset.npz
python train.py --config configs/kanfusion_no_rbf.yaml --data data/processed/your_multisource_dataset.npz
```

The NPZ file must contain `x_op` (`[N,C_op,T]`), `x_eis` (`[N,C_eis,F]`),
`x_cond` or `cond` (`[N,D_cond]`) and `labels` or `y` (`[N]`).  The training
entry auto-syncs channel counts, sequence lengths, condition dimension and class
count from the NPZ arrays before building the model.

Evaluate a saved checkpoint and export paper metrics:

```bash
python evaluate.py \
  --config configs/rbf_kanfusion.yaml \
  --data data/processed/your_multisource_dataset.npz \
  --checkpoint results/official_unishape_rbf_kanfusion/best.ckpt \
  --output-dir results/official_unishape_eval_rbf
```

Each train/eval run saves `metrics.json`, `confusion_matrix.csv`,
`predictions.csv`, `summary.csv`, macro-F1, accuracy, inference time per sample
and parameter count.  To load official UniShape pretrained weights in code, call
`model.load_official_unishape_weights(op_checkpoint=..., eis_checkpoint=...)`;
the wrapper accepts either raw state dicts or checkpoints containing
`model_state_dict` / `state_dict`.

RBF vs No-RBF control: use identical data and training settings while switching
only `configs/rbf_kanfusion.yaml` (`use_rbf_head: true`) versus
`configs/kanfusion_no_rbf.yaml` (`use_rbf_head: false`).  Ablations can toggle
`use_residual_kan_fusion`, `kan_lambda`, `freeze_unishape_backbone` and
`channel_aggregation` in those YAML files.

---

## Architecture Overview

CAPT-UniShape integrates three input modalities:

1. **Operational time series** (voltage, current, temperature, pressure, flow, humidity)
2. **EIS spectra** (Re(Z), -Im(Z), |Z|, angle(Z))
3. **Operating condition variables** (load, current density, temperature, humidity, stoichiometry)

Key components:

| Component | Description |
|---|---|
| OperationShapeAdapter | Multi-scale shape encoding of operational time series via depthwise-separable convolution |
| EISShapeAdapter | Multi-scale shape encoding of EIS frequency spectra with frequency-band embeddings |
| ConditionEncoder | MLP mapping operating conditions to a condition token |
| UniShapeBackbone | Transformer encoder with positional and scale embeddings |
| ConditionAwareAttentionPooling | Condition-modulated attention over multi-scale shape tokens |
| CrossModalFusion | Bidirectional cross-attention with gated residual fusion |
| CAPTHead | Condition-Aware Prototype Transport classification head |

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- NumPy
- PyYAML
- scikit-learn
- (Optional) Optuna -- for hyperparameter optimisation

```bash
pip install torch numpy pyyaml scikit-learn
pip install optuna   # only needed for HPO
```

## Directory Structure

```
CAPT-UniShape/
  configs/
    dataset/
      default.yaml            # Dataset config (NPZ / multi-modal)
      csv_pemfc.yaml           # Dataset config (CSV / operational-only)
    model/
      capt_unishape.yaml       # Model config (full multi-modal)
      capt_unishape_csv.yaml   # Model config (CSV, no EIS branch)
    train/
      finetune.yaml            # Training config (NPZ path)
      finetune_csv.yaml        # Training config (CSV path)
    hpo/
      hpo_config.yaml          # HPO search config (defaults to CSV path)
    experiment/
      default.yaml             # Evaluation experiment settings
  data/
    processed/                 # Data files (CSV and/or NPZ)
      eis_pemfc_dataset.csv    # Primary CSV dataset
    splits/                    # Train/val/test split manifests
    metadata/                  # label_map.json, norm_stats.json
  outputs/
    checkpoints/               # Saved model checkpoints
    figures/                   # Generated plots
    hpo/                       # HPO study results
    logs/                      # Training logs
    tables/                    # Evaluation outputs (see below)
  scripts/
    run_finetune.bat / .sh     # Fine-tuning launcher
    run_hpo.bat / .sh          # HPO launcher
    run_eval.bat / .sh         # Evaluation launcher
    run_pretrain.bat / .sh     # Pretraining launcher
  src/
    __init__.py
    main.py                    # Entry point and CAPTUniShape model wrapper
    datasets/
      __init__.py
      pemfc_dataset.py         # PEMFCDataset (NPZ-based PyTorch Dataset)
      csv_dataset.py           # CSVFuelCellDataset + build_csv_datasets()
      feature_engineering.py   # Feature engineering for CSV pipeline
      transforms.py            # Data augmentation transforms
      collate.py               # Custom collate functions
    models/
      __init__.py
      op_adapter.py            # OperationShapeAdapter
      eis_adapter.py           # EISShapeAdapter
      condition_encoder.py     # ConditionEncoder + ConditionAwareAttentionPooling
      capt_head.py             # CAPTHead (Prototype Transport)
      unishape_backbone.py     # UniShapeBackbone
      cross_modal_fusion.py    # CrossModalFusion
      losses.py                # CAPTUniShapeLoss (class-balanced + regularisation)
    trainers/
      __init__.py
      evaluator.py             # Evaluator (metrics, confusion matrix, state quantification)
      finetune_trainer.py      # Fine-tune training loop
      pretrain_trainer.py      # Pretrain training loop
      hpo_objective.py         # HPO objective function
    utils/
      __init__.py
      logger.py
      metrics.py
      profiler.py
      seed.py
      visualization.py
```

## Quick Start (CSV Path)

The CSV pipeline is the currently validated runnable path. It uses `data/processed/eis_pemfc_dataset.csv` with sliding-window feature engineering and automatic train/val/test splitting.

### 1. Dry Run (validate config + model)

```bash
python -m src.main --mode finetune --config configs/train/finetune_csv.yaml --dry-run
```

### 2. Fine-tuning

```bash
# Full training (3 seeds, 80 epochs, early stopping)
python -m src.main --mode finetune --config configs/train/finetune_csv.yaml

# Quick smoke test (1 seed, 1 epoch)
python -m src.main --mode finetune \
    --config configs/train/finetune_csv.yaml \
    train.max_epochs=1 train.seeds=[42] train.frozen_epochs=0 train.patience=1
```

### 3. Evaluation

```bash
python -m src.main --mode evaluate \
    --config configs/train/finetune_csv.yaml \
    --checkpoint outputs/checkpoints/best_seed42.ckpt
```

Evaluation outputs (saved to `outputs/tables/`):

| File | Description |
|---|---|
| `metrics.json` | Accuracy, Macro-F1, Balanced Accuracy, ROC-AUC |
| `confusion.npy` | Confusion matrix (NumPy array) |
| `roc.pkl` | ROC curve data (pickled) |
| `state_quant.csv` | Per-sample state quantification (probability-derived) |
| `state_quant.json` | Same as above in JSON format |

The evaluation also prints a full classification report and confusion matrix to the console.

**State quantification** columns: `sample_id`, `predicted_class`, `predicted_label`, `p_<ClassName>`, `risk_<ClassName>`, `abnormal_score`, `prediction_confidence`.

### 4. Hyperparameter Optimisation (HPO)

Requires `optuna` (`pip install optuna`).

```bash
# Full HPO study (100 trials, multi-fidelity)
python -m src.main --mode hpo --config configs/hpo/hpo_config.yaml

# Smoke test (1 trial, 1 epoch)
python -m src.main --mode hpo \
    --config configs/hpo/hpo_config.yaml \
    hpo.n_trials=1 train.max_epochs=1
```

HPO results are saved to `outputs/hpo/hpo_results.yaml`.

## Configuration Guide

All configurations use YAML. Training configs merge base configs via the `_base_` key:

```yaml
# Example: configs/train/finetune_csv.yaml
_base_:
  - "configs/dataset/csv_pemfc.yaml"
  - "configs/model/capt_unishape_csv.yaml"
  - "configs/experiment/default.yaml"
```

### CLI Overrides

Override any config value from the command line using dotted notation:

```bash
python -m src.main --mode finetune \
    --config configs/train/finetune_csv.yaml \
    model.d_model=512 \
    train.lr=5e-5 \
    train.batch_size=16
```

### Ablation Switches

Disable individual components for ablation studies by setting the corresponding flag in the model config:

| Switch | Default | Effect when disabled |
|---|---|---|
| `use_eis_branch` | true (false in CSV config) | Remove EIS modality (operational only) |
| `use_condition_attention` | true | Replace with mean pooling |
| `use_cross_modal` | true (false in CSV config) | Replace with concatenation + linear |
| `use_capt` | true | Replace with standard linear classifier |
| `use_cb_loss` | true | Use standard cross-entropy |
| `use_transport_reg` | true | Remove prototype offset regularisation |
| `use_residual_consistency` | true | Remove prototype smoothness penalty |

## Data Formats

### CSV (primary runnable path)

The CSV pipeline (`configs/dataset/csv_pemfc.yaml`) loads tabular data from a single CSV file and applies sliding-window segmentation with automatic feature engineering. Channel counts, condition dimensions, and class labels are auto-detected at runtime.

### NPZ (multi-modal path)

Each sample is stored as a `.npz` file with these keys:

| Key | Shape | Description |
|---|---|---|
| `x_op` | `[C_o, T]` | Operational time series |
| `x_eis` | `[C_e, F]` | EIS spectrum (optional) |
| `cond` | `[d_u]` | Operating condition vector |
| `label` | scalar | Fault class label |

The NPZ path (`configs/dataset/default.yaml`) requires pre-processed `.npz` files under `data/processed/` and split manifests under `data/splits/`.

## Current Status

- **Working**: CSV fine-tuning, evaluation (with metrics/confusion/classification report/state quantification export), HPO, dry-run validation.
- **Experiment-specific evaluations** (noise robustness, small-sample, class imbalance, cross-condition): stubs present in `run_evaluate()`, not yet implemented.
- **NPZ multi-modal path**: code exists but requires pre-processed NPZ data not included in the repository.

## Notebook / Kaggle Support

```python
from src.main import run_notebook

run_notebook(
    mode="finetune",
    dataset_name="your-dataset-folder",  # folder name under /kaggle/input/
    seed=42,
    overrides={
        "train": {"max_epochs": 50, "batch_size": 32},
        "model": {"d_model": 256},
    },
)
```

## Citation

```bibtex
@article{capt_unishape2026,
  title     = {CAPT-UniShape: Condition-Aware Prototype Transport for
               Multi-Modal PEMFC Fault Classification},
  author    = {},
  journal   = {},
  year      = {2026},
}
```

## License

This project is provided for academic research purposes.
