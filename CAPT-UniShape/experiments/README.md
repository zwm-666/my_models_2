# 实验脚本目录

按实验类型组织，所有脚本通过 `sys.path` 引用项目根目录下的 `train.py`、`models/`、`scripts/` 等模块。

## 目录结构

```
experiments/
├── data_builders/       # 数据构建脚本
│   ├── build_self_npz.py              # 自测数据 Excel → NPZ
│   ├── build_ac_voltage_npz.py        # 公开 AC Voltage 数据集 → NPZ
│   └── inspect_ac_voltage_dataset.py  # 公开数据集检查
├── baselines/           # 对比模型实验
│   ├── run_baselines.py               # 自测数据集基线对比（ML + DL）
│   └── run_public_baselines.py        # 公开数据集基线对比
├── ablation/            # 消融实验
│   └── run_ablation.py                # 模块消融 + 输入分支置零
├── noise/               # 噪声鲁棒性实验
│   ├── run_noise.py                   # noise_std 口径噪声测试
│   └── refresh_snr_noise.py           # SNR 口径噪声测试
├── multiseed/           # 多seed稳定性实验
│   ├── run_multiseed.py               # 多seed多比例实验
│   ├── run_single_seed_folds.py       # 单seed交叉验证
│   └── run_proposed_accuracy_search.py # 超参/划分搜索
├── utils/               # 实验工具（绘图、诊断）
│   ├── plot_results.py
│   ├── plot_feature_embeddings.py
│   ├── plot_attention_maps.py
│   ├── export_training_curves.py
│   └── diagnose_results.py
└── manuscript/          # 论文排版脚本
    ├── build_elsevier_revised_manuscript.py
    ├── rebuild_journal_docx_format.py
    └── remove_math_script_parentheses.py
```

## 运行方式

所有脚本从项目根目录运行：

```bash
python experiments/baselines/run_baselines.py --data data/processed/xxx.npz
python experiments/ablation/run_ablation.py --data data/processed/xxx.npz
python experiments/noise/run_noise.py --data data/processed/xxx.npz
```
