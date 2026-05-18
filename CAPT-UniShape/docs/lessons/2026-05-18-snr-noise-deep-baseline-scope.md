# SNR Noise Deep Baseline Scope

## 根因

6:4 SNR 噪声实验中，深度基线最初只读取 YAML 中的容量、训练轮数和正则参数，`feature_scope` 只对传统机器学习特征展开路径生效。结果是 `mlp/cnn1d/transformer` 仍使用全模态输入，导致 clean 过高、低 SNR 下降不足；随后单纯切到 `x_eis_only` 又会让部分模型少数类塌陷并对噪声不敏感。

## 结论

深度基线的噪声实验必须同时控制模型容量和输入范围，并确保训练与 noisy test 使用同一 `feature_scope`。只改 YAML 但不让 torch 数据集按 scope 屏蔽模态，会得到看似真实但协议不一致的结果。最终 6:4 深度基线采用：`mlp=all_modalities` 弱配置，`cnn1d=x_op_only`，`transformer=all_modalities` 弱配置；结果只从真实重跑产物合并，不能手工平滑。

2026-05-18 追加：单次全局 SNR 噪声不适合当前 6:4 测试集。测试集只有 122 个样本，accuracy 最小跳变为 0.8197%；如果噪声按整个测试集全局功率缩放，样本级实际扰动不均匀，40-20 dB 容易表现为平台化或偶然提升。论文主噪声实验应使用 `snr_scope=per_sample_modality`，并对每个 SNR 使用多个 noise seed 重复，主表写均值，标准差保存在聚合 metrics JSON 中，不进入论文 CSV 表头。

## 可检索标签

snr-noise, deep-baseline, feature-scope, modality-scope, 6_4, noise-experiment
