# CAPT-UniShape 模块结构说明

> 本文档按代码层级整理 `CAPT-UniShape` 仓库中两个在跑模型（`Official-CAPT-UniShape-RBF-KANFusion` 与 `Official-CAPT-UniShape-KANFusion-NoRBF`）里涉及的各个部件，说明每个部件放在哪里、它接收什么张量、做了什么运算、产生什么输出，以及它在整条前向路径里的角色。阅读顺序建议先看第 1 节的整体结构图和张量约定，再顺着 2\\\~6 节依次读每个子模块。

\---

## 1\. 整体结构与数据流

两个正式模型文件位于仓库根部 `models/` 包：

* `models/capt\\\_unishape\\\_rbf\\\_kanfusion.py`：完整增强版 `OfficialCAPTUniShapeRBFKANFusion`，使用 RBF 工况感知动态原型分类头。
* `models/capt\\\_unishape\\\_kanfusion\\\_no\\\_rbf.py`：公平 No-RBF 对照 `OfficialCAPTUniShapeKANFusionNoRBF`，分类头换成标准 MLP。

两者共享同一条前向骨架，只在最后的分类头位置不同。骨架如下：

```text
x\\\_op   \\\[B, C\\\_op, T]  --+-- OfficialUniShapeBackboneWrapper(op)  -- z\\\_op   \\\[B, d\\\_model]
x\\\_eis  \\\[B, C\\\_eis, F] --+-- OfficialUniShapeBackboneWrapper(eis) -- z\\\_eis  \\\[B, d\\\_model]
x\\\_cond \\\[B, D\\\_cond]   ---- ConditionEncoder                      -- z\\\_cond \\\[B, d\\\_model]

g\\\_op  = σ(op\\\_gate (z\\\_cond))       # \\\[B, d\\\_model]
g\\\_eis = σ(eis\\\_gate(z\\\_cond))       # \\\[B, d\\\_model]
z\\\_fused = g\\\_op \\\* z\\\_op + g\\\_eis \\\* z\\\_eis + z\\\_cond            # \\\[B, d\\\_model]

h, fusion\\\_aux = ResidualKANFusion(z\\\_fused, z\\\_cond)        # h: \\\[B, d\\\_model]

# RBF 版
logits, head\\\_aux = RBFPrototypeHead(h, z\\\_cond)            # \\\[B, num\\\_classes]

# NoRBF 版
logits = MLPClassifier(h)                                 # \\\[B, num\\\_classes]
```

统一的输入约束：

|张量|形状|含义|
|-|-|-|
|`x\\\_op`|`\\\[B, C\\\_op, T]`|电堆运行序列（电压/电流/功率/温度等）|
|`x\\\_eis`|`\\\[B, C\\\_eis, F]`|EIS 构造序列（幅值、相位、差分、累积等）|
|`x\\\_cond`|`\\\[B, D\\\_cond]`|工况 + 阻抗统计向量|
|`labels`|`\\\[B]`|故障类别（可选，训练时提供）|

所有内部表示通过 `d\\\_model` 维对齐，默认 `d\\\_model=128`（`configs/rbf\\\_kanfusion.yaml` 把 RBF 版设为 96）。

\---

## 2\. UniShape 部分（时间序列形状表征骨架）

### 2.1 原始 UniShape（官方 AAAI 2026 代码）

路径：`external/unishape/`（论文随附源码 vendored 进来）。核心模型类 `UniShapeModel` 位于 `external/unishape/models/unishapemodel\\\_finetune.py`。它是一个**单变量时间序列分类器**，由以下模块级联：

* `TokenGeneratorUnit`（`networks.py`）：把输入序列 `\\\[B, 1, L]` 在指定 `window\\\_size` 下做重叠 patch，生成两种特征：

  * 一阶差分与原序列分别通过 `SameConv1d` + `LayerNorm` 生成 patch 级嵌入；
  * 每个 patch 的均值、标准差通过 `MultiScaledScalarEncoder` 做多尺度对数频率编码；
  * 最后由 `LinearEncoder` 把拼接特征压到 `hidden\\\_dim=128`。
* `InceptionModule` ×2（`inceptime\\\_token`）：在 patch 序列上用多尺度 1D 卷积（Inception 结构）进一步提取局部形状，并在 `attention\\\_head` 的软注意力加权下聚合成一个类 token `cls\\\_tokens`。
* `TransformerEnc`：标准 Transformer（6 层 / 8 头 / MLP dim 512），使用 sinusoidal 位置编码。输入是 `\\\[cls\\\_tokens, x\\\_embed]`，输出 `trans\\\_enc\\\_class\\\_token`（类级别表示）和 `shape\\\_tokens`（patch 级别形状表示）。
* `fc\\\_token\\\_shape`：MLP 映射头（128→256→128），把 Transformer 的表示投影回 `hidden\\\_dim`。
* `class\\\_proto\\\_centers`：一个 `\\\[num\\\_classes, hidden\\\_dim]` 的可学习类别原型（在官方 fine-tune 代码里会用 EMA 更新），用于 shape-level 对比损失。
* 另外 `unit\\\_scale\\\_list` 和 `unit\\\_scale\\\_list\\\_finetune` 提供 5 种窗口尺寸 `\\\[64,32,16,8,4]` 的 token 生成单元，由 `scale\\\_len∈\\\[1,5]` 选择其中一档。

换句话说：UniShape 的贡献点是**多尺度 patch 化 + shape-aware token + shape 级别原型 + Transformer 主干**。CAPT-UniShape 只调用它作为**单通道特征提取器**，丢掉它自带的分类头和对比损失，保留 token 生成 + Inception + Transformer + shape 投影层。

### 2.2 CAPT-UniShape 的骨架封装

路径：`models/backbones/official\\\_unishape\\\_wrapper.py`，类名 `OfficialUniShapeBackboneWrapper`。职责：把官方单变量 UniShape 适配到 PEMFC 的多通道输入，并统一输出到 `d\\\_model`。关键行为：

1. **构造时** 初始化一份 `UniShapeModel(series\\\_size=L, in\\\_channels=128, out\\\_channels=num\\\_classes, scale\\\_len=...)`，只复用它的 token/Inception/Transformer 结构。
2. **`extract\\\_feature(x)`**（即 `forward`）：

   * 单通道 (`C==1`)：直接走 UniShape。先按 `scale\\\_len` 选用 `unit\\\_scale\\\_list` 或 `unit\\\_scale\\\_list\\\_finetune` 的 token 生成单元生成 `x\\\_embed`；再经 `inceptime\\\_token` + `attention\\\_head` 聚合出 `cls\\\_tokens`；再送 `transformer\\\_enc` 得到 `trans\\\_enc\\\_class\\\_token`；最后经 `fc\\\_token\\\_shape` 投影；最后走 `output\\\_projection`（当 `hidden\\\_dim ≠ d\\\_model` 时是 `LayerNorm + Linear`）。
   * 多通道 (`C>1`)：把 channel 折叠进 batch 维 → `\\\[B\\\*C, 1, L]` 共享 UniShape 编码 → 得到 `\\\[B, C, d\\\_model]` → 再按通道聚合。
3. **通道聚合** `\\\_aggregate\\\_channels`，三选一（由 `channel\\\_aggregation` 指定）：

   * `mean`：平均；
   * `learnable\\\_weighted`：对通道索引的可学习 logit 做 softmax；
   * `attention`（默认）：`LayerNorm → Linear → Tanh → Linear` 得到通道分数，再 softmax 加权。
4. **`load\\\_pretrained`**：可从 `external/unishape/pretrained\\\_model\\\_ckpt/unishape\\\_checkpoint\\\_finetune.pth` 这类官方 checkpoint 加载权重，会自动剥掉 `module.` / `model.` / `unishape\\\_backbone.` 等前缀。
5. **`freeze\\\_unishape\\\_backbone=True`** 时冻结官方参数，只训练通道聚合和投影头。

在完整模型里出现两次：一次编码电堆运行序列 (`op\\\_backbone`)、一次编码 EIS 序列 (`eis\\\_backbone`)。它们**不共享权重**，但结构完全一致。

\---

## 3\. 工况编码器（`ConditionEncoder`）

路径：`models/modules/condition\\\_encoder.py`。结构非常简单：

```text
Linear(D\\\_cond → hidden) → LayerNorm → GELU → Dropout → Linear(hidden → d\\\_model)
```

输入 `x\\\_cond: \\\[B, D\\\_cond]`，输出 `z\\\_cond: \\\[B, d\\\_model]`。`hidden\\\_dim` 缺省取 `max(d\\\_model, 2\\\*D\\\_cond)`。

它的产物 `z\\\_cond` 在后续会被同时用作三种用途：

1. 作为融合残差加入 `z\\\_fused`；
2. 作为门控信号生成 `g\\\_op`、`g\\\_eis`（即 `op\\\_gate`、`eis\\\_gate`，两个 `nn.Linear(d\\\_model, d\\\_model)` + sigmoid）；
3. 作为 FiLM/RBF 的条件输入（见 4.3 与 5）。

\---

## 4\. Residual KAN-Fusion 融合模块

### 4.1 模块总览

路径：`models/modules/residual\\\_kan\\\_fusion.py`，类名 `ResidualKANFusion`。设计目标：在一条数值稳定的 MLP 主分支上，叠加一条可关闭的 KAN 非线性残差分支，并通过工况驱动的 FiLM 做条件调制。公式：

```text
x       = LayerNorm(z\\\_fused)
x       = Linear → GELU → Dropout → FeatureDropout → SE → Linear → LayerNorm   # main branch
z\\\_main  = 主分支输出，形状 \\\[B, d\\\_model]

z\\\_kan   = kan\\\_to\\\_model( KAN( Bottleneck(z\\\_fused) ) )        # 仅当 use\\\_residual\\\_kan=True
z\\\_kan   = StochasticDepth(z\\\_kan)                            # 训练期随机整个残差置零
h       = z\\\_main + λ\\\_kan · z\\\_kan
h       = γ(z\\\_cond) ⊙ h + β(z\\\_cond)                         # FiLM
```

### 4.2 组件细节

* **`input\\\_norm`**：`LayerNorm` 稳定融合输入。
* **主分支**：`Linear(input\\\_dim→hidden) → GELU → Dropout → FeatureDropout → SEBlock → Linear(hidden→d\\\_model) → LayerNorm`。

  * `FeatureDropout` 是按通道的 Dropout（基于 `dropout1d`）。
  * `SEBlock` 对 hidden 特征做 `Linear→GELU→Linear→Sigmoid` 的通道级再加权，返回 `(x\\\*weights, weights)`。
* **KAN 分支**（`use\\\_residual\\\_kan\\\_fusion=True` 时启用）：

  * `bottleneck = Linear(input\\\_dim→bottleneck\\\_dim) + LayerNorm + Tanh`，默认 `bottleneck\\\_dim=32`，避免 KAN 直接处理高维特征。
  * `kan = SimpleKANLayer(bottleneck\\\_dim → bottleneck\\\_dim)`（见 4.4）。
  * `kan\\\_to\\\_model = Linear(bottleneck\\\_dim → d\\\_model)`。
* **Stochastic Depth**：训练期以概率 `stochastic\\\_depth\\\_p` 把整条残差置零，用于正则化。
* **`λ\\\_kan`**：可学习或固定的残差权重。`learnable\\\_kan\\\_lambda=True` 时是 `nn.Parameter`；否则注册为 buffer。`use\\\_residual\\\_kan\\\_fusion=False` 时直接固定为 0。
* **FiLM 调制**：`film\\\_gamma`、`film\\\_beta` 是两个 `Linear(cond\\\_dim→d\\\_model)`，初始化使 `γ=1, β=0`，即初始为恒等变换。让工况对融合表示做按通道缩放/偏移。

### 4.3 输出辅助信息（`fusion\\\_aux`）

返回字典除 `z\\\_main`、`z\\\_kan` 外还包含：`film\\\_gamma`、`film\\\_beta`、`se\\\_weights`、`stochastic\\\_depth\\\_mask`、`lambda\\\_kan`，以及 `kan\\\_regularization`（供训练损失使用）。

### 4.4 `SimpleKANLayer`（KAN 非线性层）

路径：`models/modules/simple\\\_kan.py`。它是一个最小可用的 KAN 风格层，不是完整的 pykan/efficient-kan。对输入 `x: \\\[B, D]`：

```text
x\\\_norm  = LayerNorm(x)
basis   = exp(-((x\\\_norm - centers) / width)^2)   # \\\[B, D, M] 高斯 RBF 基
weighted = basis \\\* basis\\\_weight                  # 每个输入维度有自己的一组基权重
out     = basis\\\_mixer( flatten(weighted) )       # Linear((D\\\*M) → O)
out     = out + base\\\_path(x)                     # 可选的线性直连
```

关键参数：

* `centers: \\\[D, M]`：高斯基中心，沿 `\\\[grid\\\_min, grid\\\_max]=\\\[-2, 2]` 线性初始化；
* `log\\\_width: \\\[D, M]`：每个维度/基的 log 宽度，保证正；
* `basis\\\_weight: \\\[D, M]`：逐维度基权重；
* `basis\\\_mixer: Linear(D\\\*M → O)`：把展开后的基函数线性混合到输出；
* `base\\\_path: Linear(D → O)`：稳定用的线性捷径。

另外 `regularization()` 返回 `mean(basis\\\_weight²) + 0.01·mean((Δcenter)²)`，惩罚过大的基权重和基中心不均匀分布，防止过拟合。这个标量会从 `ResidualKANFusion` 一路透出，在 `loss\\\_dict\\\['kan\\\_regularization']` 中被训练循环乘以 `alpha\\\_kan` 加进总损失。

\---

## 5\. RBF 工况感知原型分类头

### 5.1 `RBFConditionMapper`

路径：`models/modules/rbf\\\_prototype\\\_head.py` 中的内部类。把 `z\\\_cond: \\\[B, d\\\_model\\\_cond]` 映射为每类原型偏移 `Δp\\\_k(z\\\_cond): \\\[B, num\\\_classes, d\\\_model]`：

```text
dist²  = ||z\\\_cond - centers||²             # centers: \\\[num\\\_centers, cond\\\_dim]
rbf    = exp(-dist² / (2·width²))          # width 由 log\\\_width 经 exp 得到
delta  = Linear(num\\\_centers → C\\\*d\\\_model) ( rbf )
delta  = output\\\_scale · delta.view(B, C, d\\\_model)
```

`output\\\_scale` 是可学习的标量，默认初值 0.02，起“软化”偏移幅度、避免训练初期原型跑飞的作用。`linear` 的权重用 `trunc\\\_normal\\\_(std=0.01)` 初始化，bias 为 0，进一步约束初始扰动。

### 5.2 `RBFPrototypeHead`

参数：

* `prototypes: \\\[num\\\_classes, d\\\_model]`：静态类别原型，`xavier\\\_uniform\\\_` 初始化；
* `mapper`：5.1 中的 RBF 映射器；
* `log\\\_temperature`：可学习温度 `τ = exp(log\\\_temp)`，被裁剪到 `\\\[0.01, 1.0]`。默认初值来自配置中的 `temperature=0.07`。

前向：

```text
delta               = mapper(z\\\_cond)                         # \\\[B, C, d\\\_model]
dynamic\\\_prototypes  = prototypes.unsqueeze(0) + delta        # \\\[B, C, d\\\_model]
h\\\_norm              = F.normalize(h, dim=-1)
p\\\_norm              = F.normalize(dynamic\\\_prototypes, dim=-1)
logits              = einsum('bd,bkd->bk', h\\\_norm, p\\\_norm) / τ
```

本质上是按余弦相似度 / 温度做 softmax 分类，但**每个样本的类别中心会随它的工况平移**。若 `use\\\_condition\\\_transport=False`，`delta` 强制为 0，退化为“静态原型 + 余弦分类器”。

### 5.3 辅助损失

`RBFPrototypeHead.forward` 同时返回两项正则：

* `loss\\\_transport = mean(delta²)`：约束动态原型偏移幅度，防止工况把类别语义拉偏；
* `loss\\\_separation = mean(ReLU(cos(pᵢ, pⱼ) - margin))`：静态原型两两余弦需低于 `separation\\\_margin`（默认 0.2），否则受惩罚，用于拉开类别。

这两项会被模型 forward 的 `loss\\\_dict` 透出，分别乘以 `alpha\\\_transport`、`alpha\\\_sep` 进入总损失。

### 5.4 No-RBF 分支里的分类头

`OfficialCAPTUniShapeKANFusionNoRBF` 不使用 RBFPrototypeHead，而是：

```text
Classifier = Sequential(
    LayerNorm(d\\\_model),
    Linear(d\\\_model → classifier\\\_hidden),
    GELU,
    Dropout,
    Linear(classifier\\\_hidden → num\\\_classes),
)
```

即普通 MLP 分类头。其他部分（双 UniShape 分支、ConditionEncoder、op/eis 门控、ResidualKANFusion、FiLM 调制）**与 RBF 版完全一致**，这样两者差异只落在“分类头 + 对应的两项原型正则”上，构成公平消融。

\---

## 6\. 总模型封装与训练目标

### 6.1 `OfficialCAPTUniShapeRBFKANFusion`

组合 2\~5 节全部模块，完整 forward 如 1 节所示。训练时返回的 `loss\\\_dict` 中：

* `ce\\\_loss`：对 `logits + logit\\\_adjustment` 做交叉熵（支持 `class\\\_weights`、`label\\\_smoothing`，以及 Mixup 时的双标签线性混合）。
* `transport\\\_loss`：来自 RBFPrototypeHead。
* `separation\\\_loss`：来自 RBFPrototypeHead。
* `kan\\\_regularization`：来自 ResidualKANFusion → SimpleKANLayer。

总损失：

```text
total\\\_loss = ce\\\_loss
           + α\\\_transport · transport\\\_loss
           + α\\\_sep       · separation\\\_loss
           + α\\\_kan       · kan\\\_regularization
```

默认 `α\\\_transport = α\\\_sep = 1e-3`，`α\\\_kan = 1e-4`（见 `configs/rbf\\\_kanfusion.yaml`）。

此外，`\\\_mix\\\_encoded\\\_features` 在训练且 `mixup\\\_alpha > 0` 时，会在**编码后的特征空间**（`z\\\_op`/`z\\\_eis`/`z\\\_cond`）做 Manifold Mixup，并返回 `mixup\\\_lambda`、`mixup\\\_labels\\\_b` 供外部进一步做 label mixup（`train.py` 内的 `train\\\_one\\\_epoch` 也能在输入侧做额外的 sequence-level mixup）。

### 6.2 `OfficialCAPTUniShapeKANFusionNoRBF`

与 6.1 同构，但 `loss\\\_dict` 里的 `transport\\\_loss` / `separation\\\_loss` 被填成 `zeros(())`，总损失简化为：

```text
total\\\_loss = ce\\\_loss + α\\\_kan · kan\\\_regularization
```

### 6.3 `build\\\_model\\\_from\\\_config`

路径：`models/\\\_\\\_init\\\_\\\_.py`。按配置的 `use\\\_rbf\\\_head`（以及 `model\\\_name` 里是否含 `no\\\_rbf`/`rbf` 关键字）返回两者之一，并在两者冲突时抛错。`train.py` / `evaluate.py` 统一用它实例化模型，配置与 NPZ 数据（`x\\\_op`/`x\\\_eis`/`x\\\_cond`/`labels`）的维度在 `sync\\\_config\\\_with\\\_dataset` 中自动同步。

\---

## 7\. 总结：每个部件解决了什么问题

|部件|所在路径|解决的问题|
|-|-|-|
|`OfficialUniShapeBackboneWrapper`|`models/backbones/official\\\_unishape\\\_wrapper.py`|把官方单变量 UniShape 扩展到多通道 PEMFC 输入，复用其多尺度 patch + shape-aware Transformer 表征|
|`ConditionEncoder`|`models/modules/condition\\\_encoder.py`|把低维工况/阻抗统计向量映射到 `d\\\_model`，供后续融合与条件调制共用|
|`op\\\_gate` / `eis\\\_gate`|内联于两个模型文件|让工况动态决定各分支的贡献权重，类似门控 MoE|
|`ResidualKANFusion` 主分支|`models/modules/residual\\\_kan\\\_fusion.py`|在标准 MLP + SE 上构造稳定的融合表征|
|`ResidualKANFusion` KAN 分支|同上 + `SimpleKANLayer`|在低维 bottleneck 上用可学习 RBF 基做非线性残差，增强非线性融合表达力|
|`FiLM (γ, β)`|`ResidualKANFusion` 内|用工况对融合表示做按通道缩放/偏移|
|`RBFConditionMapper`|`models/modules/rbf\\\_prototype\\\_head.py`|把工况映射成每类原型的偏移 `Δp\\\_k(z\\\_cond)`|
|`RBFPrototypeHead`|同上|用“静态原型 + 工况偏移 + 余弦/温度”实现工况感知分类，并附带 transport/separation 正则|
|`MLPClassifier`|内联于 NoRBF 模型文件|公平对照分类头，用于消融 RBF 动态原型的贡献|
|`CAPTUniShape 总损失`|两个模型 forward 内|统一交叉熵 + 原型迁移正则 + 原型分离正则 + KAN 正则|

在论文行文里，这套实现可以概括为：**以 UniShape 做形状表征骨架（shape tokens + Transformer），以 ConditionEncoder + 门控 + FiLM 引入工况条件，以 Residual KAN-Fusion 完成多源非线性融合，最终以 RBF Condition-Aware Prototype Transport（CAPT）作为分类头**；其公平对照只替换最后的分类头，使得消融真正归因到“工况感知动态原型”这一机制。

