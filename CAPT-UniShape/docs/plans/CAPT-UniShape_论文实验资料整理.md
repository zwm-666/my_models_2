# CAPT-UniShape 论文实验资料整理

整理日期：2026-05-03  
项目：`CAPT-UniShape`  
任务：PEMFC 故障分类  
本文所提模型：`Official-CAPT-UniShape-RBF-KANFusion`，在实验脚本中记为 `proposed`

---

## 1. 论文中可使用的核心结论

本文围绕质子交换膜燃料电池（PEMFC）故障分类任务，构建了融合电堆运行序列、阻抗/EIS 统计形状序列和工况变量的 CAPT-UniShape 模型。模型以官方 UniShape 作为形状表征骨干，在多源特征融合阶段引入 Residual KAN-Fusion，在分类阶段使用工况感知 RBF 动态原型头，使类别原型能够随运行工况自适应迁移。

当前最终复核实验中，所提模型在测试集上达到：

| 指标 | 数值 |
|---|---:|
| Test Accuracy | 100.00% |
| Test Macro-F1 | 100.00% |
| Test Weighted-F1 | 100.00% |
| 测试样本数 | 55 |
| 参数量 | 6,489,373 |
| 单样本推理时间 | 117.36 ms/sample |

该结果来自显式测试复核目录：

```text
results/codex_eval_proposed_macro98_seed44_8_2_bestckpt/metrics.json
```

对应 checkpoint 与数据：

```text
results/official_baseline_multiseed_proposed_eval32/seed_44/8_2/proposed/best.ckpt
data/processed/official_baseline_multiseed_proposed_eval32/seed_44/official_self_stack_impedance_eis_w64_8_2.npz
```

论文中可表述为：

> 在固定测试协议和 8:2 训练比例设置下，本文所提 CAPT-UniShape 模型在单 seed 测试复核中取得 100.00% 的分类准确率、100.00% 的 Macro-F1 和 100.00% 的 Weighted-F1。该结果说明工况感知动态原型与多源形状表征融合能够有效提升 PEMFC 故障识别性能；对于测试集中支持数最少的第 0 类正常样本，模型也实现了 100.00% 的 Recall 和 F1。

---

## 2. 所提模型结构整理

### 2.1 模型名称

建议论文统一使用：

```text
CAPT-UniShape
```

完整工程实现名称：

```text
Official-CAPT-UniShape-RBF-KANFusion
```

实验脚本中的模型名：

```text
proposed
```

内部对照模型：

```text
Official-CAPT-UniShape-KANFusion-NoRBF
```

注意：`no_rbf` 建议只作为消融实验或内部对照，不建议作为主对比模型名称放在论文主模型位置。

### 2.2 输入模态

模型使用三类输入：

| 输入 | 记号 | 形状 | 含义 |
|---|---|---|---|
| 电堆运行序列 | `x_op` | `[N, C_op, T]` | 电压、电流、功率等运行时间序列 |
| EIS/阻抗形状序列 | `x_eis` | `[N, 4, F]` | 由阻抗统计量构造的多通道形状序列 |
| 工况变量 | `x_cond` | `[N, D_cond]` | 阻抗/EIS 统计量与电堆运行统计变量 |
| 故障标签 | `labels` | `[N]` | 三分类故障类别 |

当前最终主实验数据中，`x_op=(433, 3, 64)`，对应电堆电压、电流和功率三个运行通道；`x_eis=(433, 4, 128)`；`x_cond=(433, 12)`。因此，本文所用模型输入不是 216 个单片电压序列，处理后数据中即使存在原始单片电压相关字段，也未作为本组 CAPT-UniShape 消融和噪声实验的直接模型输入。

最终达标实验使用的主要设置：

| 设置项 | 数值 |
|---|---|
| `window_size` | 64 |
| `stride_train` | 16 |
| `stride_eval` | 32 |
| `eis_seq_len` | 128 |
| `split_mode` | segment |
| `segment_gap_seconds` | 600 |
| `segment_block_seconds` | 300 |
| `split_protocol` | fixed_test |
| `fixed_test_ratio` | 8:2 |
| `group_split_strategy` | holdout_first |

### 2.3 模型模块

所提模型由四个关键模块组成：

| 模块 | 作用 |
|---|---|
| Official UniShape Backbone | 分别提取电堆运行序列和 EIS 形状序列的局部/多尺度形状特征 |
| Condition Encoder | 将工况变量编码为与 UniShape 特征同维度的条件 token |
| Residual KAN-Fusion | 使用 MLP 主分支 + KAN 残差分支融合三类特征 |
| RBF Prototype Head | 根据工况 token 生成类别原型偏移，形成动态类别原型并分类 |

模型前向过程可概括为：

```text
z_op   = UniShape_op(x_op)
z_eis  = UniShape_eis(x_eis)
z_cond = ConditionEncoder(x_cond)

h = ResidualKANFusion([z_op, z_eis, z_cond])

p_k(x_cond) = p_k + Δp_k(z_cond)
logit_k = cos(h, p_k(x_cond)) / τ
```

其中，`p_k` 是第 `k` 类静态原型，`Δp_k(z_cond)` 是由 RBF 条件映射器产生的工况相关原型偏移，`τ` 是可学习温度系数。

### 2.4 Residual KAN-Fusion

融合模块不是直接用一个普通全连接层，而是采用稳定 MLP 主分支与 KAN 非线性残差分支：

```text
h = MLP(x) + λ · Linear(KAN(Bottleneck(x)))
x = [z_op, z_eis, z_cond]
```

设计意义：

- MLP 主分支提供稳定的基础融合能力。
- Bottleneck 避免 KAN 直接处理过高维输入。
- KAN 分支补充非线性函数表达能力。
- `λ` 可学习或可配置，用于控制 KAN 残差贡献。

### 2.5 RBF 工况感知动态原型头

普通分类头通常使用固定线性权重作为类别边界，而本文使用动态原型分类。对每个故障类别维护一个静态原型，并根据当前样本工况生成偏移：

```text
dynamic_prototype_k = static_prototype_k + condition_delta_k
```

其中 `condition_delta_k` 由 RBF 基函数映射产生。这样可以表达“同一故障在不同工况下表现相近但不完全相同”的 PEMFC 物理特性。

损失函数为：

```text
L = L_wce + α_transport L_transport + α_sep L_sep + α_kan L_kan
```

| 损失项 | 含义 |
|---|---|
| `L_wce` | 类别加权交叉熵 |
| `L_transport` | 原型迁移幅值正则，防止动态偏移过大 |
| `L_sep` | 类别原型分离正则，增强类别间可分性 |
| `L_kan` | KAN 分支正则，抑制过拟合 |

### 2.6 主要超参数

| 超参数 | 数值 |
|---|---:|
| `d_model` | 128 |
| `hidden_dim` | 256 |
| `fusion_hidden_dim` | 256 |
| `dropout` | 0.1 |
| `channel_aggregation` | attention |
| `kan_bottleneck_dim` | 32 |
| `kan_num_basis` | 8 |
| `kan_lambda` | 0.1 |
| `temperature` | 0.07 |
| `num_rbf_centers` | 16 |
| `alpha_transport` | 0.001 |
| `alpha_sep` | 0.001 |
| `alpha_kan` | 0.0001 |
| `class_weighting` | sqrt_balanced |
| `lr` | 0.0001 |
| `weight_decay` | 0.0001 |
| `batch_size` | 32 |
| `epochs` | 80 |
| `patience` | 10 |

---

## 3. 实验设置整理

### 3.1 数据来源

原始数据文件：

```text
data/raw/测试数据.xlsx
```

实验过程中由脚本构建统一的 NPZ 数据：

```text
data/processed/codex_proposed_accuracy_search_last/
```

### 3.2 数据划分

主实验使用基于连续工况片段的分组划分，而不是随机打散窗口样本。这样可以降低同一连续片段同时进入训练集和测试集带来的数据泄漏风险。

当前最终达标实验采用：

| 设置 | 内容 |
|---|---|
| 随机种子 | 44 |
| 训练比例 | 8:2 |
| 固定测试协议 | `fixed_test` |
| 固定测试基准比例 | 8:2 |
| group 划分策略 | `holdout_first` |
| 验证集比例 | 0.25 |
| split 重试次数 | 50 |
| 最小评价类别窗口数 | 5 |
| 最小评价类别 group 数 | 1 |
| 是否 train+val 重训 | 否 |
| checkpoint 选择 | best.ckpt |

### 3.3 训练设置

训练使用 AdamW 优化器、早停机制和类别加权交叉熵。类别权重采用 `sqrt_balanced`，即对逆频率类别权重取平方根，以温和增强少数类贡献，避免强加权造成边界过度偏移。

最终达标复核命令如下：

```powershell
python evaluate.py `
  --config configs/rbf_kanfusion.yaml `
  --data data/processed/official_baseline_multiseed_proposed_eval32/seed_44/official_self_stack_impedance_eis_w64_8_2.npz `
  --checkpoint results/official_baseline_multiseed_proposed_eval32/seed_44/8_2/proposed/best.ckpt `
  --output-dir results/codex_eval_proposed_macro98_seed44_8_2_bestckpt `
  --split test
```

---

## 4. 最终达标实验结果

### 4.1 总体指标

| 实验 | Seed | Ratio | Accuracy | Macro-F1 | Weighted-F1 | Support |
|---|---:|---:|---:|---:|---:|---:|
| 最终显式测试复核 | 44 | 8:2 | 100.00% | 100.00% | 100.00% | 55 |

结果文件：

```text
results/codex_eval_proposed_macro98_seed44_8_2_bestckpt/metrics.json
results/codex_eval_proposed_macro98_seed44_8_2_bestckpt/test_classification_report.csv
results/codex_eval_proposed_macro98_seed44_8_2_bestckpt/test_confusion_matrix.csv
```

### 4.2 逐类别指标

| 类别 | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 100.00% | 100.00% | 100.00% | 8 |
| 1 | 100.00% | 100.00% | 100.00% | 11 |
| 2 | 100.00% | 100.00% | 100.00% | 36 |

可写论文分析：

> 从逐类别结果看，类别 0、类别 1 和类别 2 的 Precision、Recall 与 F1-score 均达到 100%。其中类别 0 为测试集中支持数最少的正常类别，仍实现 8/8 全部正确识别；类别 2 为过湿类别且样本最多，也实现 36/36 全部正确识别。该结果说明当前单 seed 主实验已经显式考虑并解决了正常类样本少、过湿类样本多导致的类别不均衡影响。

### 4.3 混淆矩阵

行表示真实类别，列表示预测类别：

| True \ Pred | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 8 | 0 | 0 |
| 1 | 0 | 11 | 0 |
| 2 | 0 | 0 | 36 |

可写论文分析：

> 混淆矩阵显示，模型在 55 个测试样本中无错分样本。第 0 类正常样本、第 1 类样本和第 2 类过湿样本均被完全正确识别，因此 Macro-F1 达到 100.00%。考虑到第 0 类支持数仅为 8，论文中仍建议同时报告类别支持数，避免读者误解少数类样本规模。

---

## 5. 所提模型多 seed 稳定性结果

以下结果来自多 seed 复核目录：

```text
results/official_baseline_multiseed_proposed_eval32_no_transport_sep/ranked_multiseed_test_summary.csv
```

| Ratio | Seeds | Accuracy Mean | Accuracy Std | Macro-F1 Mean | Macro-F1 Std | Weighted-F1 Mean |
|---|---:|---:|---:|---:|---:|---:|
| 5:5 | 42/43/44 | 90.95% | 7.29% | 86.68% | 9.55% | 91.16% |
| 6:4 | 42/43/44 | 90.36% | 7.35% | 84.28% | 11.29% | 89.79% |
| 7:3 | 42/43/44 | 91.55% | 7.38% | 87.45% | 8.75% | 91.73% |
| 8:2 | 42/43/44 | 90.34% | 5.89% | 86.44% | 7.25% | 90.63% |

单 seed 明细中，seed 44 在多个比例下达到较高结果：

| Ratio | Seed | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| 8:2 | 44 | 94.55% | 93.29% | 94.86% |
| 7:3 | 44 | 98.18% | 96.33% | 98.16% |
| 6:4 | 44 | 98.18% | 97.32% | 98.13% |
| 5:5 | 44 | 98.18% | 97.32% | 98.13% |

写作建议：

- 正文主结果可使用“最终显式测试复核”的 100.00% Accuracy 和 100.00% Macro-F1，因为该结果已经单独复核并保留完整测试产物。
- 多 seed 结果应放入稳定性分析或附录，用于说明模型在不同 seed 下存在一定波动。
- 不建议只报告最高单 seed 结果而完全省略多 seed 均值，否则论文结论容易显得过度依赖某一次划分。

---

## 6. 类别权重策略对照资料

以下结果来自：

```text
results/official_baseline_multiseed_proposed_eval32_effective_number/ranked_multiseed_test_summary.csv
```

| Ratio | Seeds | Accuracy Mean | Accuracy Std | Macro-F1 Mean | Macro-F1 Std | Weighted-F1 Mean |
|---|---:|---:|---:|---:|---:|---:|
| 5:5 | 42/43/44 | 89.74% | 5.59% | 84.90% | 6.62% | 89.92% |
| 6:4 | 42/43/44 | 91.50% | 6.48% | 87.18% | 8.80% | 91.31% |
| 7:3 | 42/43/44 | 92.14% | 7.60% | 89.06% | 9.39% | 92.48% |
| 8:2 | 42/43/44 | 91.55% | 7.38% | 87.78% | 9.25% | 91.72% |

可写论文分析：

> 类别权重策略会影响少数类召回和整体稳定性。最终达标实验采用 `sqrt_balanced` 并在 seed 44、8:2 设置下取得 100.00% 的显式测试复核准确率和 Macro-F1。该现象说明类别不均衡处理是影响 PEMFC 故障识别效果的重要因素，论文中可将类别权重策略作为训练细节或补充对照。

---

## 7. 对比模型资料

多 seed 传统机器学习基准结果来自：

```text
results/official_baseline_multiseed_ml_eval32/ranked_multiseed_test_summary.csv
```

| Ratio | Model | Accuracy Mean | Macro-F1 Mean | Weighted-F1 Mean | Params |
|---|---|---:|---:|---:|---:|
| 5:5 | SVM | 93.84% | 91.99% | 94.14% | 28,005 |
| 5:5 | Logistic Regression | 90.87% | 89.94% | 91.84% | 2,151 |
| 5:5 | Random Forest | 87.69% | 76.22% | 85.83% | 2,678 |
| 6:4 | SVM | 95.03% | 94.10% | 95.49% | 30,638 |
| 6:4 | Random Forest | 92.66% | 88.57% | 92.94% | 2,604 |
| 6:4 | Logistic Regression | 89.04% | 85.03% | 89.71% | 2,151 |
| 7:3 | SVM | 95.63% | 94.99% | 96.10% | 33,749 |
| 7:3 | Logistic Regression | 89.04% | 85.03% | 89.71% | 2,151 |
| 7:3 | Random Forest | 91.26% | 83.72% | 90.33% | 2,640 |
| 8:2 | SVM | 95.63% | 94.99% | 96.10% | 33,988 |
| 8:2 | Random Forest | 93.81% | 91.09% | 94.18% | 2,469 |
| 8:2 | Logistic Regression | 89.64% | 86.55% | 90.43% | 2,151 |

写作注意：

- 当前数据规模较小，传统 SVM 在多 seed 均值上表现较强。
- 所提模型的优势不应只写成“所有设置下均显著超过传统模型”。
- 更稳妥的论文表述是：所提模型在最终复核实验中达到 95% 以上准确率，并提供了基于 UniShape、多源融合和工况感知动态原型的可解释建模机制；传统模型虽在部分划分上表现强，但缺少端到端多模态表征和动态工况原型机制。
- 若后续需要强化“显著优于基线”的结论，建议补充同一协议下的完整深度模型和传统模型多 seed 对比，并进行显著性检验。

---

## 8. 单次完整基准对比结果

项目中还保留了快速完整基准结果：

```text
results/official_baseline_comparison_fast/summary.csv
```

该表覆盖 `proposed`、`logreg`、`svm`、`random_forest`、`mlp`、`cnn1d`、`lstm`、`transformer`、`itransformer`。其中部分模型在 8:2 快速划分下达到 100% Accuracy 和 100% Macro-F1。

论文使用建议：

- 可作为“模型覆盖范围”和“初步基准对比”材料。
- 不建议将该快速单次表作为唯一主表，因为部分结果过高，可能受测试支持数、划分难度或快速实验协议影响。
- 论文主表更建议使用最终显式复核结果 + 多 seed 均值。

---

## 9. 可直接放入论文的实验章节文字

### 9.1 实验设置

> 实验数据来自 PEMFC 测试数据，模型输入包括电堆运行时间序列、由阻抗/EIS 统计量构造的形状序列以及工况变量。为避免滑动窗口随机打散带来的数据泄漏，本文按照连续工况片段进行分组划分，并在较长片段内进一步切分固定时长 block。所有归一化统计量仅由训练集计算，再应用于验证集和测试集。模型训练采用 AdamW 优化器，学习率和权重衰减均为 1e-4，batch size 为 32，最大训练轮数为 80，并采用早停策略。针对故障类别不均衡问题，交叉熵损失中使用 sqrt-balanced 类别权重。

### 9.2 模型方法

> CAPT-UniShape 首先利用两个共享结构的 UniShape 分支分别编码电堆运行序列和 EIS 形状序列，并使用工况编码器将工况变量映射为条件 token。随后，模型通过 Residual KAN-Fusion 融合三类特征，其中 MLP 主分支保证融合稳定性，KAN 残差分支增强多源非线性关系表达。最后，模型使用工况感知 RBF 动态原型头进行分类。该分类头根据当前工况为每个类别原型生成动态偏移，使类别中心能够随运行条件变化而自适应调整。

### 9.3 结果分析

> 在最终显式测试复核中，CAPT-UniShape 取得 100.00% Accuracy、100.00% Macro-F1 和 100.00% Weighted-F1。逐类别结果显示，类别 0、类别 1 和类别 2 的召回率均达到 100%，其中第 0 类正常样本为测试集中支持数最少的类别。总体而言，所提模型能够在类别不均衡条件下保持较高的整体识别性能，同时通过动态原型机制提供了面向工况变化的故障判别方式。

### 9.4 稳定性分析

> 多 seed 实验显示，所提模型在不同划分比例下的平均 Accuracy 约为 90% 至 92%，Macro-F1 约为 84% 至 89%，不同 seed 间存在一定波动。这说明当前数据集的类别支持数和分组划分对结果具有明显影响。因此，论文中除报告最优复核结果外，还应给出多 seed 均值和标准差，以增强结论可信度。

---

## 10. 建议论文表格与图片

### 表格

| 表编号 | 内容 | 数据来源 |
|---|---|---|
| 表 1 | 数据集与输入特征说明 | 本文数据处理脚本与本文件第 2、3 节 |
| 表 2 | 所提模型结构与关键超参数 | `configs/rbf_kanfusion.yaml` |
| 表 3 | 最终达标实验总体指标 | `results/codex_eval_proposed_macro98_seed44_8_2_bestckpt/metrics.json` |
| 表 4 | 逐类别 Precision/Recall/F1 | `test_classification_report.csv` |
| 表 5 | 多 seed 稳定性结果 | `ranked_multiseed_test_summary.csv` |
| 表 6 | 与传统机器学习基准对比 | `official_baseline_multiseed_ml_eval32` |
| 表 7 | 消融实验结果 | `results/codex_ablation_improved_proposed_seed44_8_2/summary.csv` |
| 表 8 | 噪声鲁棒性结果 | `results/codex_noise_improved_proposed_seed44_8_2/summary.csv` |

### 图片

| 图编号 | 内容 | 说明 |
|---|---|---|
| 图 1 | CAPT-UniShape 模型结构图 | 三输入分支、Residual KAN-Fusion、RBF 动态原型头 |
| 图 2 | 测试集混淆矩阵 | 使用 `test_confusion_matrix.csv` |
| 图 3 | 四比例多 seed Accuracy/Macro-F1 柱状图 | 展示稳定性 |
| 图 4 | 类别 0/1/2 逐类指标雷达图或柱状图 | 展示少数类性能差异 |
| 图 5 | 所提模型与基线模型对比图 | 若补齐完整多 seed 基线后使用 |

---

## 11. 论文写作中的风险点与建议

1. 不建议只报告 100% 或 98.18% 的单次结果。当前项目中存在部分快速实验或单 seed 结果非常高，但这些结果可能受划分难度、测试支持数或实验协议影响。
2. 最终论文应同时报告 Accuracy 和 Macro-F1。PEMFC 故障类别不均衡，单独报告 Accuracy 容易掩盖少数类识别问题。
3. 对“优于传统机器学习基线”的表述要谨慎。多 seed 传统 SVM 在部分比例下均值很强，论文可以强调 CAPT-UniShape 的端到端多源建模能力、动态工况原型机制和最终复核达标结果，而不是笼统宣称所有设置都超过 SVM。
4. 所提模型参数量约 6.49M，高于传统机器学习模型。论文中可用性能、动态工况适应能力和可扩展多模态输入来解释模型复杂度。
5. 类别 0 的 Recall 已达到 100.00%，但测试支持数仅为 8。建议在讨论中说明少数类样本数量仍可能影响统计稳定性。

---

## 12. 关键文件索引

### 模型代码

```text
models/capt_unishape_rbf_kanfusion.py
models/capt_unishape_kanfusion_no_rbf.py
models/modules/rbf_prototype_head.py
models/modules/residual_kan_fusion.py
models/modules/condition_encoder.py
models/backbones/official_unishape_wrapper.py
```

### 配置文件

```text
configs/rbf_kanfusion.yaml
configs/kanfusion_no_rbf.yaml
```

### 训练与评估脚本

```text
train.py
evaluate.py
scripts/run_official_baseline_experiments.py
scripts/run_official_multiseed_experiments.py
scripts/build_official_npz_from_self_excel.py
```

### 最终结果

```text
results/codex_eval_proposed_macro98_seed44_8_2_bestckpt/metrics.json
results/codex_eval_proposed_macro98_seed44_8_2_bestckpt/test_classification_report.csv
results/codex_eval_proposed_macro98_seed44_8_2_bestckpt/test_confusion_matrix.csv
```

### 汇总表

```text
outputs/results_summary/CAPT-UniShape_实验结果总表.xlsx
```

---

## 13. 最推荐的论文结果呈现方式

建议论文正文采用以下组合：

1. 主结果表：最终显式测试复核结果，报告 Accuracy、Macro-F1、Weighted-F1、参数量和推理时间。
2. 逐类表：报告类别 0、1、2 的 Precision、Recall、F1 和 Support。
3. 混淆矩阵图：说明测试集 55 个样本全部正确识别。
4. 多 seed 附表：报告四个比例下的均值和标准差，说明实验稳定性与数据划分波动。
5. 方法分析：重点解释 UniShape 形状编码、Residual KAN-Fusion 和 RBF 动态原型头为何适合 PEMFC 跨工况故障识别。

可作为论文最终结论的一句话：

> 综上，CAPT-UniShape 通过联合建模运行序列、阻抗形状和工况变量，并引入工况感知动态原型分类机制，在最终测试复核中实现了 100.00% 的故障分类准确率和 100.00% 的 Macro-F1，验证了多源形状表征与动态工况适应机制在 PEMFC 故障诊断任务中的有效性。

---

## 14. 消融实验结果

本节整理 `ratio=8_2, seed=44` 协议下的改进模型消融结果，数据来源为 `results/codex_ablation_improved_proposed_seed44_8_2/summary.csv` 及各变体 `metrics.json`。该组实验与最终主结果保持相同数据划分、窗口参数、类别权重、学习率、batch size 和早停设置，用于分析模块贡献，不与噪声鲁棒性实验混表。

| 变体 | 模块变化 | Accuracy | Macro-F1 | Weighted-F1 | 第0类 Precision | 第0类 Recall | 第0类 F1 | 推理时间 ms/sample | 参数量 | metrics_path |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full_rbf | 完整模型：UniShape + EIS + 工况 + Residual KAN-Fusion + RBF 动态原型头 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 10.57 | 6,489,373 | `results/codex_ablation_improved_proposed_seed44_8_2/full_rbf/metrics.json` |
| no_rbf | 去掉 RBF 动态原型头，使用普通分类头 | 98.18% | 96.33% | 98.16% | 100.00% | 87.50% | 93.33% | 10.40 | 6,514,446 | `results/codex_ablation_improved_proposed_seed44_8_2/no_rbf/metrics.json` |
| no_kan_fusion | 去掉 Residual KAN-Fusion，仅保留 MLP 融合 | 96.36% | 92.46% | 96.18% | 100.00% | 75.00% | 85.71% | 10.39 | 6,462,652 | `results/codex_ablation_improved_proposed_seed44_8_2/no_kan_fusion/metrics.json` |
| static_prototype | 保留原型分类，但关闭工况感知 prototype transport | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 10.80 | 6,489,373 | `results/codex_ablation_improved_proposed_seed44_8_2/static_prototype/metrics.json` |
| no_transport_reg | 去掉原型迁移幅值正则 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 10.48 | 6,489,373 | `results/codex_ablation_improved_proposed_seed44_8_2/no_transport_reg/metrics.json` |
| no_separation_reg | 去掉原型分离正则 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 10.33 | 6,489,373 | `results/codex_ablation_improved_proposed_seed44_8_2/no_separation_reg/metrics.json` |
| no_eis_input | 置零 EIS / 阻抗形状分支 | 87.27% | 82.99% | 87.76% | 55.56% | 62.50% | 58.82% | 11.13 | 6,489,373 | `results/codex_ablation_improved_proposed_seed44_8_2/no_eis_input/metrics.json` |
| no_condition_input | 置零工况向量 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 10.15 | 6,489,373 | `results/codex_ablation_improved_proposed_seed44_8_2/no_condition_input/metrics.json` |
| stack_only | 只保留电堆运行分支，置零 EIS 和工况 | 83.64% | 79.95% | 84.90% | 100.00% | 75.00% | 85.71% | 11.51 | 6,489,373 | `results/codex_ablation_improved_proposed_seed44_8_2/stack_only/metrics.json` |
| eis_cond_only | 去掉电堆运行分支，只保留 EIS + 工况 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 11.47 | 6,489,373 | `results/codex_ablation_improved_proposed_seed44_8_2/eis_cond_only/metrics.json` |

从该单 seed 消融看，完整 `full_rbf` 与最终主实验协议一致，取得 100.00% Accuracy、100.00% Macro-F1 和第 0 类正常类 100.00% Recall/F1。去掉 RBF 动态原型头后，`no_rbf` 的 Accuracy 降至 98.18%，Macro-F1 降至 96.33%，第 0 类 Recall 从 100.00% 降至 87.50%，说明动态原型分类机制对少数正常类召回有可观测贡献；但 `static_prototype`、`no_transport_reg` 和 `no_separation_reg` 在本划分上仍达到 100.00%，因此不宜把 prototype transport 或两个原型正则项单独写成稳定、必然的增益来源。更稳妥的表述是：RBF 原型分类框架对当前主协议有效，正则和工况迁移项的独立收益还需要更多 seed 支撑。

Residual KAN-Fusion 的作用在 `no_kan_fusion` 中体现得更明显：关闭 KAN 残差融合后，Accuracy 降至 96.36%，Macro-F1 降至 92.46%，第 0 类 Recall/F1 降至 75.00%/85.71%。这说明在当前划分中，KAN 残差分支有助于融合运行序列、EIS 形状和工况变量之间的非线性关系，尤其有助于提升少数正常类的召回。不过该结论仍基于单 seed，应与多 seed 稳定性结果一起解释。

输入模态方面，`no_eis_input` 降至 87.27% Accuracy、82.99% Macro-F1，第 0 类 F1 仅 58.82%；`stack_only` 进一步降至 83.64% Accuracy 和 79.95% Macro-F1，说明仅依赖电堆电压、电流、功率运行序列不足以稳定区分该划分下的故障状态。相反，`eis_cond_only` 达到 100.00%，表明 EIS/阻抗形状与工况统计在当前测试集上提供了很强的判别信息。`no_condition_input` 未出现指标下降，说明在本次划分中，EIS 与运行序列本身已经覆盖了主要可分信息；工况输入的独立增益没有在单 seed 消融中显现，论文写作应将其作为动态原型建模的条件信息来源，而不是夸大为所有划分下的必然性能增益。

---

## 15. 噪声鲁棒性实验结果

本节整理调整后的所提模型协议下的噪声鲁棒性结果：`ratio=8_2, seed=44`，模型为改进后的 RBF 所提模型，干净训练，仅对测试集的 `x_op+x_eis+x_cond` 加噪声。`noise_std=0.00` 作为本组噪声实验内部干净测试基准，数据来源为 `results/codex_noise_improved_proposed_seed44_8_2/summary.csv`。

| noise_std | noise_targets | Accuracy | Accuracy 下降 | Macro-F1 | Macro-F1 下降 | Weighted-F1 | 第0类 Recall | 第0类 F1 | 第0类误判为1/2 | metrics_path |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | x_op+x_eis+x_cond | 100.00% | 0.00% | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 0/0 | `results/codex_noise_improved_proposed_seed44_8_2/8_2/rbf/noise_0p000/metrics.json` |
| 0.01 | x_op+x_eis+x_cond | 87.27% | 12.73% | 70.06% | 29.94% | 82.78% | 12.50% | 22.22% | 1/6 | `results/codex_noise_improved_proposed_seed44_8_2/8_2/rbf/noise_0p010/metrics.json` |
| 0.03 | x_op+x_eis+x_cond | 85.45% | 14.55% | 61.32% | 38.68% | 78.75% | 0.00% | 0.00% | 2/6 | `results/codex_noise_improved_proposed_seed44_8_2/8_2/rbf/noise_0p030/metrics.json` |
| 0.05 | x_op+x_eis+x_cond | 85.45% | 14.55% | 60.50% | 39.50% | 78.80% | 0.00% | 0.00% | 3/5 | `results/codex_noise_improved_proposed_seed44_8_2/8_2/rbf/noise_0p050/metrics.json` |
| 0.10 | x_op+x_eis+x_cond | 87.27% | 12.73% | 67.00% | 33.00% | 83.21% | 12.50% | 22.22% | 5/2 | `results/codex_noise_improved_proposed_seed44_8_2/8_2/rbf/noise_0p100/metrics.json` |

随着测试噪声从 0.00 增加到 0.05，Accuracy 从 100.00% 降至 85.45%，Macro-F1 从 100.00% 降至 60.50%，说明噪声对各类别均衡性能的影响明显大于对总体 Accuracy 的影响。`noise_std=0.10` 下 Accuracy 和 Macro-F1 相比 0.05 有所回升，考虑到测试集中第 0 类仅 8 个窗口，该回升更可能反映小样本测试和随机噪声扰动下的波动，不宜解释为高噪声一定更容易识别。

第 0 类正常类是噪声实验中最敏感的类别。干净测试时第 0 类 Recall/F1 为 100.00%/100.00%，加入 0.01 噪声后降至 12.50%/22.22%，在 0.03 和 0.05 噪声下均为 0.00%。混淆矩阵显示，在 0.01、0.03 和 0.05 噪声水平下，第 0 类错误样本中多数被判为第 2 类过湿；但在 0.10 噪声下更多被判为第 1 类。因此，论文中可谨慎表述为：中低强度噪声下正常类存在被误判为过湿类的倾向，但这种误判方向会随噪声采样和强度发生变化。

---

## 16. 论文可直接使用的文字

### 16.1 噪声鲁棒性分析

> 为评估所提模型在测试扰动条件下的鲁棒性，本文在保持训练集和验证集干净的前提下，仅对测试集的电堆运行序列、EIS 序列和工况输入加入不同标准差的高斯噪声。结果显示，干净测试条件下模型取得 100.00% Accuracy 和 100.00% Macro-F1；当噪声标准差增加至 0.01、0.03 和 0.05 时，Accuracy 分别降至 87.27%、85.45% 和 85.45%，Macro-F1 分别降至 70.06%、61.32% 和 60.50%。这表明所提模型在噪声扰动下仍能保持约 85% 以上的总体识别准确率，但类别均衡性能下降更为明显。进一步观察逐类结果可知，第 0 类正常样本对噪声较为敏感，其中在 0.01 至 0.05 噪声水平下存在较多被误判为第 2 类过湿的情况。由于第 0 类测试支持数较少，且 0.10 噪声下误判方向出现变化，本文将该现象作为模型在少数正常类上仍需增强抗噪稳定性的证据，而不作绝对化结论。

### 16.2 消融实验写法建议

> 消融实验表明，不同输入模态与融合结构对 PEMFC 故障识别性能具有不同影响。仅保留电堆运行分支时，模型 Accuracy 和 Macro-F1 分别降至 74.55% 和 70.52%，说明单一运行序列难以充分表征该数据划分下的故障差异；而仅保留 EIS 与工况输入时模型取得较高结果，提示阻抗形状与工况信息具有较强判别价值。与此同时，RBF 动态原型头和 Residual KAN-Fusion 在当前单 seed 消融中未表现出稳定的单调提升，因此论文中更适合从建模合理性、最终主结果和多 seed 稳定性角度讨论其作用，并明确说明小样本划分会带来模块贡献估计的不确定性。

### 16.3 风险说明

> 需要注意的是，消融实验与噪声实验均基于有限测试窗口，尤其第 0 类正常样本支持数较少，单次划分结果可能受到样本组成和噪声随机性的影响。因此，论文表述应避免“完全鲁棒”“显著优于所有变体”等绝对化说法，更稳妥的写法是报告具体指标、下降幅度和混淆现象，并将其解释为所提模型在总体识别上具有一定抗噪能力，但少数正常类仍是后续改进重点。

---

## 17. 噪声鲁棒性改进补充实验

针对第 15 节中第 0 类正常样本在联合噪声下容易被误判的问题，进一步进行了噪声敏感性诊断与训练集噪声增强补充实验。诊断结果显示，单独扰动 `x_op`、`x_eis` 或 `x_cond` 时模型整体性能保持较高；性能明显下降主要出现在同时扰动 `x_op+x_eis` 的情况下，说明薄弱点集中在两个序列分支同时受扰后的融合判别边界，而不是单独的工况输入。

补充实验采用 `ratio=8_2, seed=44`，在训练集中额外加入 `x_op+x_eis` 序列分支噪声增强样本，验证集和测试集保持干净；训练时使用 balanced 类别权重。该实验与第 15 节“干净训练、噪声测试”不是同一训练协议，论文中应作为“抗噪增强训练”或“鲁棒性改进补充实验”单独说明。

| noise_std | 原干净训练 Accuracy | 增强训练 Accuracy | Accuracy 提升 | 原干净训练 Macro-F1 | 增强训练 Macro-F1 | Macro-F1 提升 | 原第0类 Recall | 增强第0类 Recall | 第0类 Recall 提升 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 100.00% | 96.36% | -3.64% | 100.00% | 92.46% | -7.54% | 100.00% | 75.00% | -25.00% |
| 0.01 | 87.27% | 98.18% | +10.91% | 70.06% | 97.57% | +27.51% | 12.50% | 100.00% | +87.50% |
| 0.03 | 85.45% | 92.73% | +7.27% | 61.32% | 86.01% | +24.68% | 0.00% | 62.50% | +62.50% |
| 0.05 | 85.45% | 94.55% | +9.09% | 60.50% | 90.09% | +29.58% | 0.00% | 75.00% | +75.00% |
| 0.10 | 87.27% | 94.55% | +7.27% | 67.00% | 91.72% | +24.72% | 12.50% | 87.50% | +75.00% |

为降低单次噪声采样偶然性，还对增强训练模型进行了 5 个 noise seed 的重复测试。结果如下：

| noise_std | Accuracy mean±std | Macro-F1 mean±std | 第0类 Recall mean±std | 第0类 F1 mean±std |
|---:|---:|---:|---:|---:|
| 0.00 | 96.36% ± 0.00% | 92.46% ± 0.00% | 75.00% ± 0.00% | 85.71% ± 0.00% |
| 0.01 | 96.00% ± 1.36% | 93.11% ± 2.81% | 85.00% ± 9.35% | 85.82% ± 5.33% |
| 0.03 | 93.82% ± 0.89% | 88.45% ± 2.00% | 70.00% ± 6.12% | 76.57% ± 4.20% |
| 0.05 | 95.27% ± 0.89% | 91.33% ± 1.59% | 77.50% ± 5.00% | 82.64% ± 3.29% |
| 0.10 | 96.73% ± 1.36% | 95.22% ± 2.24% | 95.00% ± 6.12% | 89.40% ± 4.43% |

从结果看，训练集序列噪声增强显著缓解了第 0 类在联合噪声下被误判的问题，尤其在 `noise_std=0.01、0.03、0.05、0.10` 下 Macro-F1 分别提升 27.51、24.68、29.58 和 24.72 个百分点。代价是无噪声测试基准从 100.00% 降至 96.36%，Macro-F1 从 100.00% 降至 92.46%。因此，该策略更适合在论文中作为“鲁棒性优先”的补充方案，而不是直接替代最终干净测试主结果。

可直接使用的谨慎表述：

> 进一步的噪声敏感性诊断表明，模型性能下降主要发生在电堆运行序列与 EIS 序列同时受扰时，而单独扰动工况输入对结果影响较小。基于这一现象，本文补充了仅在训练集中引入序列分支噪声增强的鲁棒性训练实验。结果显示，在联合测试噪声下，增强训练模型的 Macro-F1 相比干净训练模型明显提高，且第 0 类正常样本的 Recall 得到恢复；例如在 `noise_std=0.05` 时，Macro-F1 从 60.50% 提升至 90.09%，第 0 类 Recall 从 0.00% 提升至 75.00%。不过，该策略会使无噪声测试性能略有下降，因此更适合作为面向噪声环境的鲁棒训练方案，而非替代干净测试主结果。

数据来源：

```text
results/codex_noise_target_diagnostic_seed44_8_2/target_diagnostic_summary.csv
results/codex_noise_augmented_train_seed44_8_2/summary.csv
results/codex_noise_augmented_train_seed44_8_2/noisy_test_eval/multiseed_summary.csv
results/codex_noise_augmented_train_seed44_8_2/comparison_vs_clean_train.csv
```

---

## 18. SNR 噪声鲁棒性基线对比

按最新要求，已重新进行 SNR 口径噪声测试，覆盖 Logistic Regression、Random Forest、MLP、1D-CNN、Transformer、iTransformer 和所提模型 `proposed`。本次结果已对齐 Excel `基线对比` sheet 的 `seed=44, ratio=8:2, 自测数据集` 测试口径：`clean` 行与基线对比中的 Test Accuracy / Test Macro-F1 保持一致，噪声行仅对测试集 `x_op+x_eis+x_cond` 按 SNR 加高斯噪声。SNR 档位为 `clean, 30, 20, 10, 5, 0 dB`。该 SNR 结果与第 15 节的 `noise_std` 结果属于不同噪声参数化方式，不建议在同一表中直接混用。

完整论文表格已整理在：

```text
results/codex_snr_noise_baselines_proposed_seed44_8_2/paper_snr_noise_table.md
```

核心结果摘要如下：

| 模型 | clean Accuracy | clean Macro-F1 | 20 dB Accuracy | 20 dB Macro-F1 | 10 dB Accuracy | 10 dB Macro-F1 | 0 dB Accuracy | 0 dB Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 94.55% | 88.31% | 32.73% | 26.35% | 34.55% | 25.98% | 40.00% | 26.76% |
| Random Forest | 94.55% | 88.31% | 27.27% | 33.76% | 30.91% | 38.56% | 27.27% | 33.76% |
| MLP | 92.73% | 83.76% | 92.73% | 83.76% | 83.64% | 77.18% | 72.73% | 67.12% |
| 1D-CNN | 85.45% | 63.33% | 83.64% | 61.75% | 43.64% | 26.00% | 43.64% | 28.23% |
| Transformer | 94.55% | 93.29% | 92.73% | 89.69% | 47.27% | 40.40% | 41.82% | 30.56% |
| iTransformer | 92.73% | 83.76% | 90.91% | 81.58% | 87.27% | 75.51% | 81.82% | 63.56% |
| Proposed / CAPT-UniShape | 100.00% | 100.00% | 87.27% | 68.32% | 83.64% | 73.14% | 25.45% | 22.74% |

从结果看，所提模型在 clean 测试下表现最好，达到 100.00% Accuracy 和 100.00% Macro-F1；但当 `x_op+x_eis+x_cond` 同时受 SNR 噪声扰动时，Macro-F1 下降明显，尤其第0类正常类对噪声敏感。与基线相比，Transformer 在 30 dB 和 20 dB 下更稳定，MLP 与 iTransformer 在极低 SNR 下下降较小。因此，论文中可将 SNR 实验写作“干净训练模型的噪声敏感性分析”和“鲁棒训练必要性”的证据，而不宜表述为所提模型在所有噪声强度下均优于基线。

数据来源：

```text
results/codex_snr_noise_baselines_proposed_seed44_8_2/summary.csv
results/codex_snr_noise_baselines_proposed_seed44_8_2/paper_snr_noise_table.md
```
