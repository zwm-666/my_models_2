# CAPT-UniShape 模块结构说明

> 本文档按代码层级整理 `CAPT-UniShape` 仓库中两个在跑模型（`Official-CAPT-UniShape-RBF-KANFusion` 与 `Official-CAPT-UniShape-KANFusion-NoRBF`）里涉及的各个部件，说明每个部件放在哪里、它接收什么张量、做了什么运算、产生什么输出，以及它在整条前向路径里的角色。阅读顺序建议先看第 1 节的整体结构图和张量约定，再顺着 2~6 节依次读每个子模块。

---

## 1. 整体结构与数据流

两个正式模型文件位于仓库根部 `models/` 包：

- `models/capt_unishape_rbf_kanfusion.py`：完整增强版 `OfficialCAPTUniShapeRBFKANFusion`，使用 RBF 工况感知动态原型分类头。
- `models/capt_unishape_kanfusion_no_rbf.py`：公平 No-RBF 对照 `OfficialCAPTUniShapeKANFusionNoRBF`，分类头换成标准 MLP。

两者共享同一条前向骨架，只在最后的分类头位置不同。骨架如下：

```text
x_op   [B, C_op, T]  --+-- OfficialUniShapeBackboneWrapper(op)  -- z_op   [B, d_model]
x_eis  [B, C_eis, F] --+-- OfficialUniShapeBackboneWrapper(eis) -- z_eis  [B, d_model]
x_cond [B, D_cond]   ---- ConditionEncoder                      -- z_cond [B, d_model]

g_op  = σ(op_gate (z_cond))       # [B, d_model]
g_eis = σ(eis_gate(z_cond))       # [B, d_model]
z_fused = g_op * z_op + g_eis * z_eis + z_cond            # [B, d_model]

h, fusion_aux = ResidualKANFusion(z_fused, z_cond)        # h: [B, d_model]

# RBF 版
logits, head_aux = RBFPrototypeHead(h, z_cond)            # [B, num_classes]

# NoRBF 版
logits = MLPClassifier(h)                                 # [B, num_classes]
```

统一的输入约束：

| 张量 | 形状 | 含义 |
|---|---|---|
| `x_op` | `[B, C_op, T]` | 电堆运行序列（电压/电流/功率/温度等） |
| `x_eis` | `[B, C_eis, F]` | EIS 构造序列（幅值、相位、差分、累积等） |
| `x_cond` | `[B, D_cond]` | 工况 + 阻抗统计向量 |
| `labels` | `[B]` | 故障类别（可选，训练时提供） |

所有内部表示通过 `d_model` 维对齐，默认 `d_model=128`（`configs/rbf_kanfusion.yaml` 把 RBF 版设为 96）。

---

## 2. UniShape 部分（时间序列形状表征骨架）

### 2.1 原始 UniShape（官方 AAAI 2026 代码）

路径：`external/unishape/`（论文随附源码 vendored 进来）。核心模型类 `UniShapeModel` 位于 `external/unishape/models/unishapemodel_finetune.py`。它是一个**单变量时间序列分类器**，由以下模块级联：

- `TokenGeneratorUnit`（`networks.py`）：把输入序列 `[B, 1, L]` 在指定 `window_size` 下做重叠 patch，生成两种特征：
  - 一阶差分与原序列分别通过 `SameConv1d` + `LayerNorm` 生成 patch 级嵌入；
  - 每个 patch 的均值、标准差通过 `MultiScaledScalarEncoder` 做多尺度对数频率编码；
  - 最后由 `LinearEncoder` 把拼接特征压到 `hidden_dim=128`。
- `InceptionModule` ×2（`inceptime_token`）：在 patch 序列上用多尺度 1D 卷积（Inception 结构）进一步提取局部形状，并在 `attention_head` 的软注意力加权下聚合成一个类 token `cls_tokens`。
- `TransformerEnc`：标准 Transformer（6 层 / 8 头 / MLP dim 512），使用 sinusoidal 位置编码。输入是 `[cls_tokens, x_embed]`，输出 `trans_enc_class_token`（类级别表示）和 `shape_tokens`（patch 级别形状表示）。
- `fc_token_shape`：MLP 映射头（128→256→128），把 Transformer 的表示投影回 `hidden_dim`。
- `class_proto_centers`：一个 `[num_classes, hidden_dim]` 的可学习类别原型（在官方 fine-tune 代码里会用 EMA 更新），用于 shape-level 对比损失。
- 另外 `unit_scale_list` 和 `unit_scale_list_finetune` 提供 5 种窗口尺寸 `[64,32,16,8,4]` 的 token 生成单元，由 `scale_len∈[1,5]` 选择其中一档。

换句话说：UniShape 的贡献点是**多尺度 patch 化 + shape-aware token + shape 级别原型 + Transformer 主干**。CAPT-UniShape 只调用它作为**单通道特征提取器**，丢掉它自带的分类头和对比损失，保留 token 生成 + Inception + Transformer + shape 投影层。

### 2.2 CAPT-UniShape 的骨架封装

路径：`models/backbones/official_unishape_wrapper.py`，类名 `OfficialUniShapeBackboneWrapper`。职责：把官方单变量 UniShape 适配到 PEMFC 的多通道输入，并统一输出到 `d_model`。关键行为：

1. **构造时** 初始化一份 `UniShapeModel(series_size=L, in_channels=128, out_channels=num_classes, scale_len=...)`，只复用它的 token/Inception/Transformer 结构。
2. **`extract_feature(x)`**（即 `forward`）：
   - 单通道 (`C==1`)：直接走 UniShape。先按 `scale_len` 选用 `unit_scale_list` 或 `unit_scale_list_finetune` 的 token 生成单元生成 `x_embed`；再经 `inceptime_token` + `attention_head` 聚合出 `cls_tokens`；再送 `transformer_enc` 得到 `trans_enc_class_token`；最后经 `fc_token_shape` 投影；最后走 `output_projection`（当 `hidden_dim ≠ d_model` 时是 `LayerNorm + Linear`）。
   - 多通道 (`C>1`)：把 channel 折叠进 batch 维 → `[B*C, 1, L]` 共享 UniShape 编码 → 得到 `[B, C, d_model]` → 再按通道聚合。
3. **通道聚合** `_aggregate_channels`，三选一（由 `channel_aggregation` 指定）：
   - `mean`：平均；
   - `learnable_weighted`：对通道索引的可学习 logit 做 softmax；
   - `attention`（默认）：`LayerNorm → Linear → Tanh → Linear` 得到通道分数，再 softmax 加权。
4. **`load_pretrained`**：可从 `external/unishape/pretrained_model_ckpt/unishape_checkpoint_finetune.pth` 这类官方 checkpoint 加载权重，会自动剥掉 `module.` / `model.` / `unishape_backbone.` 等前缀。
5. **`freeze_unishape_backbone=True`** 时冻结官方参数，只训练通道聚合和投影头。

在完整模型里出现两次：一次编码电堆运行序列 (`op_backbone`)、一次编码 EIS 序列 (`eis_backbone`)。它们**不共享权重**，但结构完全一致。

---

## 3. 工况编码器（`ConditionEncoder`）

路径：`models/modules/condition_encoder.py`。结构非常简单：

```text
Linear(D_cond → hidden) → LayerNorm → GELU → Dropout → Linear(hidden → d_model)
```

输入 `x_cond: [B, D_cond]`，输出 `z_cond: [B, d_model]`。`hidden_dim` 缺省取 `max(d_model, 2*D_cond)`。

它的产物 `z_cond` 在后续会被同时用作三种用途：

1. 作为融合残差加入 `z_fused`；
2. 作为门控信号生成 `g_op`、`g_eis`（即 `op_gate`、`eis_gate`，两个 `nn.Linear(d_model, d_model)` + sigmoid）；
3. 作为 FiLM/RBF 的条件输入（见 4.3 与 5）。

---

## 4. Residual KAN-Fusion 融合模块

### 4.1 模块总览

路径：`models/modules/residual_kan_fusion.py`，类名 `ResidualKANFusion`。设计目标：在一条数值稳定的 MLP 主分支上，叠加一条可关闭的 KAN 非线性残差分支，并通过工况驱动的 FiLM 做条件调制。公式：

```text
x       = LayerNorm(z_fused)
x       = Linear → GELU → Dropout → FeatureDropout → SE → Linear → LayerNorm   # main branch
z_main  = 主分支输出，形状 [B, d_model]

z_kan   = kan_to_model( KAN( Bottleneck(z_fused) ) )        # 仅当 use_residual_kan=True
z_kan   = StochasticDepth(z_kan)                            # 训练期随机整个残差置零
h       = z_main + λ_kan · z_kan
h       = γ(z_cond) ⊙ h + β(z_cond)                         # FiLM
```

### 4.2 组件细节

- **`input_norm`**：`LayerNorm` 稳定融合输入。
- **主分支**：`Linear(input_dim→hidden) → GELU → Dropout → FeatureDropout → SEBlock → Linear(hidden→d_model) → LayerNorm`。
  - `FeatureDropout` 是按通道的 Dropout（基于 `dropout1d`）。
  - `SEBlock` 对 hidden 特征做 `Linear→GELU→Linear→Sigmoid` 的通道级再加权，返回 `(x*weights, weights)`。
- **KAN 分支**（`use_residual_kan_fusion=True` 时启用）：
  - `bottleneck = Linear(input_dim→bottleneck_dim) + LayerNorm + Tanh`，默认 `bottleneck_dim=32`，避免 KAN 直接处理高维特征。
  - `kan = SimpleKANLayer(bottleneck_dim → bottleneck_dim)`（见 4.4）。
  - `kan_to_model = Linear(bottleneck_dim → d_model)`。
- **Stochastic Depth**：训练期以概率 `stochastic_depth_p` 把整条残差置零，用于正则化。
- **`λ_kan`**：可学习或固定的残差权重。`learnable_kan_lambda=True` 时是 `nn.Parameter`；否则注册为 buffer。`use_residual_kan_fusion=False` 时直接固定为 0。
- **FiLM 调制**：`film_gamma`、`film_beta` 是两个 `Linear(cond_dim→d_model)`，初始化使 `γ=1, β=0`，即初始为恒等变换。让工况对融合表示做按通道缩放/偏移。

### 4.3 输出辅助信息（`fusion_aux`）

返回字典除 `z_main`、`z_kan` 外还包含：`film_gamma`、`film_beta`、`se_weights`、`stochastic_depth_mask`、`lambda_kan`，以及 `kan_regularization`（供训练损失使用）。

### 4.4 `SimpleKANLayer`（KAN 非线性层）

路径：`models/modules/simple_kan.py`。它是一个最小可用的 KAN 风格层，不是完整的 pykan/efficient-kan。对输入 `x: [B, D]`：

```text
x_norm  = LayerNorm(x)
basis   = exp(-((x_norm - centers) / width)^2)   # [B, D, M] 高斯 RBF 基
weighted = basis * basis_weight                  # 每个输入维度有自己的一组基权重
out     = basis_mixer( flatten(weighted) )       # Linear((D*M) → O)
out     = out + base_path(x)                     # 可选的线性直连
```

关键参数：

- `centers: [D, M]`：高斯基中心，沿 `[grid_min, grid_max]=[-2, 2]` 线性初始化；
- `log_width: [D, M]`：每个维度/基的 log 宽度，保证正；
- `basis_weight: [D, M]`：逐维度基权重；
- `basis_mixer: Linear(D*M → O)`：把展开后的基函数线性混合到输出；
- `base_path: Linear(D → O)`：稳定用的线性捷径。

另外 `regularization()` 返回 `mean(basis_weight²) + 0.01·mean((Δcenter)²)`，惩罚过大的基权重和基中心不均匀分布，防止过拟合。这个标量会从 `ResidualKANFusion` 一路透出，在 `loss_dict['kan_regularization']` 中被训练循环乘以 `alpha_kan` 加进总损失。

---

## 5. RBF 工况感知原型分类头

### 5.1 `RBFConditionMapper`

路径：`models/modules/rbf_prototype_head.py` 中的内部类。把 `z_cond: [B, d_model_cond]` 映射为每类原型偏移 `Δp_k(z_cond): [B, num_classes, d_model]`：

```text
dist²  = ||z_cond - centers||²             # centers: [num_centers, cond_dim]
rbf    = exp(-dist² / (2·width²))          # width 由 log_width 经 exp 得到
delta  = Linear(num_centers → C*d_model) ( rbf )
delta  = output_scale · delta.view(B, C, d_model)
```

`output_scale` 是可学习的标量，默认初值 0.02，起“软化”偏移幅度、避免训练初期原型跑飞的作用。`linear` 的权重用 `trunc_normal_(std=0.01)` 初始化，bias 为 0，进一步约束初始扰动。

### 5.2 `RBFPrototypeHead`

参数：

- `prototypes: [num_classes, d_model]`：静态类别原型，`xavier_uniform_` 初始化；
- `mapper`：5.1 中的 RBF 映射器；
- `log_temperature`：可学习温度 `τ = exp(log_temp)`，被裁剪到 `[0.01, 1.0]`。默认初值来自配置中的 `temperature=0.07`。

前向：

```text
delta               = mapper(z_cond)                         # [B, C, d_model]
dynamic_prototypes  = prototypes.unsqueeze(0) + delta        # [B, C, d_model]
h_norm              = F.normalize(h, dim=-1)
p_norm              = F.normalize(dynamic_prototypes, dim=-1)
logits              = einsum('bd,bkd->bk', h_norm, p_norm) / τ
```

本质上是按余弦相似度 / 温度做 softmax 分类，但**每个样本的类别中心会随它的工况平移**。若 `use_condition_transport=False`，`delta` 强制为 0，退化为“静态原型 + 余弦分类器”。

### 5.3 辅助损失

`RBFPrototypeHead.forward` 同时返回两项正则：

- `loss_transport = mean(delta²)`：约束动态原型偏移幅度，防止工况把类别语义拉偏；
- `loss_separation = mean(ReLU(cos(pᵢ, pⱼ) - margin))`：静态原型两两余弦需低于 `separation_margin`（默认 0.2），否则受惩罚，用于拉开类别。

这两项会被模型 forward 的 `loss_dict` 透出，分别乘以 `alpha_transport`、`alpha_sep` 进入总损失。

### 5.4 No-RBF 分支里的分类头

`OfficialCAPTUniShapeKANFusionNoRBF` 不使用 RBFPrototypeHead，而是：

```text
Classifier = Sequential(
    LayerNorm(d_model),
    Linear(d_model → classifier_hidden),
    GELU,
    Dropout,
    Linear(classifier_hidden → num_classes),
)
```

即普通 MLP 分类头。其他部分（双 UniShape 分支、ConditionEncoder、op/eis 门控、ResidualKANFusion、FiLM 调制）**与 RBF 版完全一致**，这样两者差异只落在“分类头 + 对应的两项原型正则”上，构成公平消融。

---

## 6. 总模型封装与训练目标

### 6.1 `OfficialCAPTUniShapeRBFKANFusion`

组合 2~5 节全部模块，完整 forward 如 1 节所示。训练时返回的 `loss_dict` 中：

- `ce_loss`：对 `logits + logit_adjustment` 做交叉熵（支持 `class_weights`、`label_smoothing`，以及 Mixup 时的双标签线性混合）。
- `transport_loss`：来自 RBFPrototypeHead。
- `separation_loss`：来自 RBFPrototypeHead。
- `kan_regularization`：来自 ResidualKANFusion → SimpleKANLayer。

总损失：

```text
total_loss = ce_loss
           + α_transport · transport_loss
           + α_sep       · separation_loss
           + α_kan       · kan_regularization
```

默认 `α_transport = α_sep = 1e-3`，`α_kan = 1e-4`（见 `configs/rbf_kanfusion.yaml`）。

此外，`_mix_encoded_features` 在训练且 `mixup_alpha > 0` 时，会在**编码后的特征空间**（`z_op`/`z_eis`/`z_cond`）做 Manifold Mixup，并返回 `mixup_lambda`、`mixup_labels_b` 供外部进一步做 label mixup（`train.py` 内的 `train_one_epoch` 也能在输入侧做额外的 sequence-level mixup）。

### 6.2 `OfficialCAPTUniShapeKANFusionNoRBF`

与 6.1 同构，但 `loss_dict` 里的 `transport_loss` / `separation_loss` 被填成 `zeros(())`，总损失简化为：

```text
total_loss = ce_loss + α_kan · kan_regularization
```

### 6.3 `build_model_from_config`

路径：`models/__init__.py`。按配置的 `use_rbf_head`（以及 `model_name` 里是否含 `no_rbf`/`rbf` 关键字）返回两者之一，并在两者冲突时抛错。`train.py` / `evaluate.py` 统一用它实例化模型，配置与 NPZ 数据（`x_op`/`x_eis`/`x_cond`/`labels`）的维度在 `sync_config_with_dataset` 中自动同步。

---

## 7. 总结：每个部件解决了什么问题

| 部件 | 所在路径 | 解决的问题 |
|---|---|---|
| `OfficialUniShapeBackboneWrapper` | `models/backbones/official_unishape_wrapper.py` | 把官方单变量 UniShape 扩展到多通道 PEMFC 输入，复用其多尺度 patch + shape-aware Transformer 表征 |
| `ConditionEncoder` | `models/modules/condition_encoder.py` | 把低维工况/阻抗统计向量映射到 `d_model`，供后续融合与条件调制共用 |
| `op_gate` / `eis_gate` | 内联于两个模型文件 | 让工况动态决定各分支的贡献权重，类似门控 MoE |
| `ResidualKANFusion` 主分支 | `models/modules/residual_kan_fusion.py` | 在标准 MLP + SE 上构造稳定的融合表征 |
| `ResidualKANFusion` KAN 分支 | 同上 + `SimpleKANLayer` | 在低维 bottleneck 上用可学习 RBF 基做非线性残差，增强非线性融合表达力 |
| `FiLM (γ, β)` | `ResidualKANFusion` 内 | 用工况对融合表示做按通道缩放/偏移 |
| `RBFConditionMapper` | `models/modules/rbf_prototype_head.py` | 把工况映射成每类原型的偏移 `Δp_k(z_cond)` |
| `RBFPrototypeHead` | 同上 | 用“静态原型 + 工况偏移 + 余弦/温度”实现工况感知分类，并附带 transport/separation 正则 |
| `MLPClassifier` | 内联于 NoRBF 模型文件 | 公平对照分类头，用于消融 RBF 动态原型的贡献 |
| `CAPTUniShape 总损失` | 两个模型 forward 内 | 统一交叉熵 + 原型迁移正则 + 原型分离正则 + KAN 正则 |

在论文行文里，这套实现可以概括为：**以 UniShape 做形状表征骨架（shape tokens + Transformer），以 ConditionEncoder + 门控 + FiLM 引入工况条件，以 Residual KAN-Fusion 完成多源非线性融合，最终以 RBF Condition-Aware Prototype Transport（CAPT）作为分类头**；其公平对照只替换最后的分类头，使得消融真正归因到“工况感知动态原型”这一机制。
