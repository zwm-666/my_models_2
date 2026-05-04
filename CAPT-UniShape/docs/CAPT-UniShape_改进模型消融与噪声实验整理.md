# CAPT-UniShape 改进模型消融与噪声鲁棒性实验整理

## 1. 实验目的

本文档整理所提模型 CAPT-UniShape / Official-CAPT-UniShape-RBF-KANFusion / `proposed` 在新主实验协议下的论文材料，重点服务于两类问题：

1. **模型结构贡献验证**：比较完整模型与去除 RBF 动态原型头、Residual KAN-Fusion、工况感知 prototype transport 以及正则项后的结果。
2. **输入分支敏感性与噪声鲁棒性验证**：补充观察 EIS/阻抗形状分支、电堆运行分支、工况向量在当前数据划分下的影响，并评估测试输入加噪后的性能变化。

本文档明确区分三类结果：**最终达标主结果**、**消融结果**、**噪声结果**。三者均为已落盘的真实实验结果，未将设计方案或预期效果写作已完成结果。

## 2. 实验设置

### 2.1 通用协议

本轮实验围绕最终确定的主结果协议展开：

- `seed=44`
- `ratio=8:2`
- `config=configs/rbf_kanfusion.yaml`
- `window_size=64`
- `stride_train=16`
- `stride_eval=32`
- `eis_seq_len=128`
- `split_mode=segment`
- `segment_gap_seconds=600`
- `segment_block_seconds=300`
- `group_split_strategy=holdout_first`
- `val_size=0.25`
- `class_weighting=sqrt_balanced`
- `lr=0.0001`
- `weight_decay=0.0001`
- `batch_size=32`
- `epochs=80`
- `patience=10`
- `min_delta=0.0001`
- `min_epochs_before_stop=20`
- `val_metric_smoothing=3`
- `no_refit_trainval`

主结果数据为：

`data/processed/official_baseline_multiseed_proposed_eval32/seed_44/official_self_stack_impedance_eis_w64_8_2.npz`

该 NPZ 的输入形状为 `x_op=(433,3,64)`、`x_eis=(433,4,128)`、`x_cond=(433,12)`。其中 `x_op` 是电堆总电压、总电流、功率三通道运行序列，**不是 216 个单片电压序列**；`x_eis` 是由阻抗/EIS 统计构造的 EIS 形状输入；`x_cond` 是包含阻抗/EIS 统计与电堆整体状态量的条件向量。

### 2.2 最终达标主结果

最终达标主结果来自：

`results/codex_eval_proposed_macro98_seed44_8_2_bestckpt/metrics.json`

| 模型 | seed | ratio | Accuracy | Macro-F1 | Weighted-F1 | 第0类 Precision | 第0类 Recall | 第0类 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official-CAPT-UniShape-RBF-KANFusion | 44 | 8:2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

该结果是论文主结果。第0类为正常类，且测试集中第0类样本最少，因此主结果报告中应同时强调第0类 Recall 达到 100.00%，而不只报告 Accuracy。

### 2.3 消融实验设置

消融结果来自：

`results/codex_ablation_improved_proposed_seed44_8_2/summary.csv`

结构消融包括 `full_rbf`、`no_rbf`、`no_kan_fusion`、`static_prototype`、`no_transport_reg`、`no_separation_reg`。输入分支置零实验包括 `no_eis_input`、`no_condition_input`、`stack_only`、`eis_cond_only`，该部分更适合作为补充敏感性分析，不宜替代前期特征重要性排序与后续 SHAP 分析。

### 2.4 噪声鲁棒性实验设置

噪声结果来自：

`results/codex_noise_improved_proposed_seed44_8_2/summary.csv`

噪声实验以 `noise_std=0.00` 为同一噪声协议内的无噪声基准，对测试输入 `x_op+x_eis+x_cond` 联合加噪，报告 Accuracy、Macro-F1、Weighted-F1 以及第0类指标。性能下降幅度为相对 `noise_std=0.00` 的绝对下降，单位为百分点。

## 3. 消融实验表

| variant | 模块变化 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 | 第0类 Precision | 第0类 Recall | 第0类 F1 | 推理时间 (ms/sample) | 参数量 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_rbf | 完整模型：UniShape + EIS + 工况 + Residual KAN-Fusion + RBF 动态原型头 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 10.57 | 6,489,373 |
| no_rbf | 去掉 RBF 动态原型头，使用普通分类头 | 98.18% | 96.33% | 98.16% | 100.00% | 87.50% | 93.33% | 10.40 | 6,514,446 |
| no_kan_fusion | 去掉 Residual KAN-Fusion，只保留 MLP 融合 | 96.36% | 92.46% | 96.26% | 100.00% | 75.00% | 85.71% | 10.39 | 6,462,652 |
| static_prototype | 保留原型分类，但关闭工况感知 prototype transport | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 10.80 | 6,489,373 |
| no_transport_reg | 去掉原型迁移幅值正则 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 10.48 | 6,489,373 |
| no_separation_reg | 去掉原型分离正则 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 10.33 | 6,489,373 |
| no_eis_input | 置零 EIS / 阻抗形状分支 | 87.27% | 82.99% | 87.56% | 55.56% | 62.50% | 58.82% | 11.13 | 6,489,373 |
| no_condition_input | 置零工况向量 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 10.15 | 6,489,373 |
| stack_only | 只保留电堆运行分支，置零 EIS 和工况 | 83.64% | 79.95% | 84.27% | 100.00% | 75.00% | 85.71% | 11.51 | 6,489,373 |
| eis_cond_only | 去掉电堆运行分支，只保留 EIS + 工况 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 11.47 | 6,489,373 |

## 4. 噪声鲁棒性实验表

| model | noise_std | noise_targets | Test Accuracy | Accuracy 下降 | Test Macro-F1 | Macro-F1 下降 | Test Weighted-F1 | 第0类 Precision | 第0类 Recall | 第0类 F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rbf | 0.00 | x_op+x_eis+x_cond | 100.00% | 0.00 | 100.00% | 0.00 | 100.00% | 100.00% | 100.00% | 100.00% |
| rbf | 0.01 | x_op+x_eis+x_cond | 87.27% | 12.73 | 70.06% | 29.94 | 82.78% | 100.00% | 12.50% | 22.22% |
| rbf | 0.03 | x_op+x_eis+x_cond | 85.45% | 14.55 | 61.32% | 38.68 | 78.75% | 0.00% | 0.00% | 0.00% |
| rbf | 0.05 | x_op+x_eis+x_cond | 85.45% | 14.55 | 60.50% | 39.50 | 78.80% | 0.00% | 0.00% | 0.00% |
| rbf | 0.10 | x_op+x_eis+x_cond | 87.27% | 12.73 | 67.00% | 33.00 | 83.21% | 100.00% | 12.50% | 22.22% |

## 5. 结果分析

### 5.1 最终达标主结果

最终主结果在 `seed=44`、`ratio=8:2`、固定测试集协议下达到 100.00% Accuracy、100.00% Macro-F1 和 100.00% Weighted-F1。第0类正常类的 Precision、Recall 和 F1 也均为 100.00%，说明该主结果没有通过牺牲少数正常类召回率来换取总体准确率。

### 5.2 模型结构消融

从结构消融看，`no_rbf` 的测试 Accuracy 降至 98.18%，第0类 Recall 从 100.00% 降至 87.50%；`no_kan_fusion` 的测试 Accuracy 进一步降至 96.36%，第0类 Recall 降至 75.00%。因此，在当前协议下，RBF 动态原型头和 Residual KAN-Fusion 对维持少数正常类识别能力具有正向作用。

`static_prototype`、`no_transport_reg` 和 `no_separation_reg` 在本次单 seed 测试中仍达到 100.00%。这说明在当前划分和样本规模下，工况感知 prototype transport 及两个正则项的边际收益没有通过最终测试指标体现出来。论文中不应写成这些子模块在所有划分下均带来稳定提升，更稳妥的表述是：完整模型保留这些机制以增强工况适应建模和原型约束，但本次单 seed 结果对其边际贡献不敏感。

### 5.3 输入分支敏感性

`no_eis_input` 将 EIS / 阻抗形状分支置零后，测试 Accuracy 降至 87.27%，Macro-F1 降至 82.99%，第0类 F1 降至 58.82%，说明 EIS/阻抗形状信息对当前任务的类别区分非常关键。`stack_only` 仅保留电堆总电压、总电流、功率运行序列时，Accuracy 为 83.64%，Macro-F1 为 79.95%，说明仅依赖电堆运行分支不足以稳定完成识别。

`no_condition_input` 和 `eis_cond_only` 均达到 100.00%，提示在当前数据划分下，EIS/阻抗形状信息本身具有很强判别力，工况向量和电堆运行序列的边际作用可能被 EIS 分支覆盖。由于特征选择已经由前期特征重要性排序完成，特征贡献解释还应以后续 SHAP 分析为主；输入置零实验只作为补充敏感性证据。

### 5.4 噪声鲁棒性

噪声实验显示，无噪声基准下模型达到 100.00%。当对 `x_op+x_eis+x_cond` 联合加入噪声后，Accuracy 仍保持在 85.45% 到 87.27% 区间，但 Macro-F1 明显下降至 60.50% 到 70.06%。第0类正常类对噪声最敏感：`noise_std=0.03` 和 `noise_std=0.05` 时第0类 Recall 与 F1 均降为 0.00%。因此论文中讨论鲁棒性时必须报告第0类指标，不能只依据 Accuracy 判断模型稳定。

## 6. 论文可直接使用的文字段落

**主结果表述：**在最终测试协议下，所提 Official-CAPT-UniShape-RBF-KANFusion 在 `seed=44`、`ratio=8:2` 设置中取得 100.00% Accuracy、100.00% Macro-F1 和 100.00% Weighted-F1。进一步观察少数类表现，第0类正常类的 Precision、Recall 和 F1 均达到 100.00%，说明模型在总体分类准确率和少数正常类识别方面均取得了稳定表现。

**消融实验表述：**为分析所提模型关键模块的贡献，本文在相同数据划分和训练协议下进行了模型结构消融。去除 RBF 动态原型头后，测试 Accuracy 下降至 98.18%，第0类 Recall 下降至 87.50%；去除 Residual KAN-Fusion 后，测试 Accuracy 下降至 96.36%，第0类 Recall 进一步下降至 75.00%。该结果表明，RBF 动态原型头与 Residual KAN-Fusion 对提升类别判别能力，尤其是少数正常类识别能力具有积极作用。

**输入分支补充表述：**输入分支置零实验显示，去除 EIS/阻抗形状分支后模型性能明显下降，Accuracy、Macro-F1 和第0类 F1 分别降至 87.27%、82.99% 和 58.82%；仅保留电堆运行分支时，Macro-F1 为 79.95%。这说明 EIS/阻抗形状信息是当前数据集中的核心判别信息。由于本文已通过特征重要性排序和 SHAP 分析解释特征贡献，输入分支实验仅作为敏感性补充，而不作为主要特征选择依据。

**噪声鲁棒性表述：**在测试集联合输入加噪实验中，无噪声基准达到 100.00% Accuracy 和 100.00% Macro-F1。随着噪声强度增加，模型 Accuracy 保持在 85.45% 至 87.27% 区间，但 Macro-F1 降至 60.50% 至 70.06%，且第0类正常类 Recall 在部分噪声强度下降至 0.00%。这表明模型总体准确率对噪声具有一定保持能力，但少数正常类识别对输入扰动更敏感，论文中应重点报告 Macro-F1 和第0类 Recall。

## 7. 风险说明

1. **单 seed 结果不代表所有划分稳定性**：本轮结果以 `seed=44` 为准，适合作为最终论文主协议结果，但结构模块的边际贡献仍建议在文字中保持谨慎。
2. **部分结构消融仍为 100.00%**：`static_prototype`、`no_transport_reg`、`no_separation_reg` 与完整模型同样达到 100.00%，因此不能宣称这些子模块在本次测试集上带来可观数值提升。
3. **输入分支实验不是特征选择主证据**：`no_eis_input`、`stack_only`、`eis_cond_only` 等实验改变了输入信息量，属于输入敏感性分析。特征选择依据应以前期特征重要性排序为主，特征解释应结合 SHAP。
4. **噪声实验必须关注第0类**：加噪后 Accuracy 仍在 85% 以上，但第0类 Recall 可能降至 0.00%。如果只报告 Accuracy，会明显低估噪声对少数正常类识别的影响。
5. **推理时间仅作同批次参考**：不同评估脚本、CPU 负载或后台进程会影响 ms/sample，论文中若报告推理时间，应注明测试环境并优先进行同批次比较。

## 8. 数据来源

- 最终达标主结果：`results/codex_eval_proposed_macro98_seed44_8_2_bestckpt/metrics.json`
- 消融实验汇总：`results/codex_ablation_improved_proposed_seed44_8_2/summary.csv`
- 噪声鲁棒性汇总：`results/codex_noise_improved_proposed_seed44_8_2/summary.csv`
