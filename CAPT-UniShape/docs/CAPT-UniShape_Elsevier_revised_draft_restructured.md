# CAPT-UniShape: Condition-Aware Prototype Transport UniShape for PEMFC Fault Classification

## Highlights
- CAPT-UniShape unifies stack operation sequences, EIS/impedance shape sequences, and condition/statistical vectors in a single PEMFC fault-classification framework.
- A Residual KAN-Fusion module combines a stable MLP branch with a bottleneck KAN residual branch and a condition gate to model nonlinear cross-modal interactions.
- A condition-aware RBF prototype transport head adapts class prototypes to the encoded operating condition before temperature-scaled cosine classification.
- Under the seed-44 8:2 protocol, the proposed model reaches 100.00% accuracy, 100.00% Macro-F1, and 100.00% Weighted-F1 on 55 test windows with 6,489,373 parameters.
- Across the 7:3, 6:4, and 5:5 splits, CAPT-UniShape consistently delivers the highest Macro-F1 among the evaluated baselines, including Transformer and logistic regression.

## Abstract
Reliable fault classification is critical for proton exchange membrane fuel cell (PEMFC) systems, where water management faults, reactant starvation, and condition-dependent operating drift jointly affect efficiency and durability. Conventional data-driven diagnostic models typically rely on a single signal source or a static classification boundary, which limits performance when operational time series, impedance-related responses, and operating conditions jointly shape the fault signature. This paper presents CAPT-UniShape, a Condition-Aware Prototype Transport UniShape model for PEMFC fault classification. The model ingests three inputs in parallel: stack operation sequences x_op in R^{N x 3 x 64}, constructed EIS/impedance statistical shape sequences x_eis in R^{N x 4 x 128}, and condition/statistical vectors x_cond in R^{N x 12}. Two official UniShape backbones extract operation and EIS shape embeddings, a condition encoder maps x_cond into a condition token, and a Residual KAN-Fusion module integrates the three embeddings through an MLP main path, a bottleneck KAN residual path, and a condition-driven gate. A condition-aware RBF prototype transport head then shifts each static class prototype by a condition-dependent offset before temperature-scaled cosine classification. Under the seed-44 8:2 protocol, CAPT-UniShape achieves 100.00% test accuracy, 100.00% Macro-F1, and 100.00% Weighted-F1 on 55 test windows, surpassing the Transformer baseline (94.55% / 93.29%) and the strongest classical baselines (94.55% / 88.31%). The leading Macro-F1 is retained across 7:3, 6:4, and 5:5 splits. Ablation results identify the EIS/impedance branch as the dominant discriminative source and quantify the contribution of the RBF prototype head and Residual KAN-Fusion to Normal-class recall. Noise robustness analysis further shows that augmented training restores Macro-F1 from 60.50% to 90.09% at noise_std = 0.0500, indicating practical strategies for stronger operating environments.

## Keywords
Proton exchange membrane fuel cell; Fault classification; Electrochemical impedance spectroscopy; UniShape; Prototype learning; Kolmogorov-Arnold network; Radial basis function


## Abbreviations

| Abbreviation | Full term | Meaning in this manuscript |
| --- | --- | --- |
| AdamW | Adam with decoupled weight decay | Optimizer used for model training |
| CAPT | Condition-Aware Prototype Transport | Condition-driven dynamic prototype classification mechanism |
| CE | Cross-entropy | Main supervised classification loss term |
| CNN | Convolutional neural network | Baseline model family |
| EIS | Electrochemical impedance spectroscopy | Impedance-related diagnostic information source |
| GELU | Gaussian error linear unit | Activation function used in the neural modules |
| KAN | Kolmogorov-Arnold network | Nonlinear residual branch used in Residual KAN-Fusion |
| LN | Layer normalization | Normalization operation used in the condition encoder and fusion modules |
| ML | Machine learning | Traditional baseline category |
| MLP | Multi-layer perceptron | Feed-forward neural network module or baseline |
| NPZ | NumPy compressed archive | Data file format used by the official experiment pipeline |
| PEMFC | Proton exchange membrane fuel cell | Target fuel-cell system for fault classification |
| RBF | Radial basis function | Condition mapper used to generate prototype offsets |
| SNR | Signal-to-noise ratio | Noise level used in robustness experiments |
| Macro-F1 | Macro-averaged F1 score | Unweighted average F1 across classes |
| Weighted-F1 | Support-weighted F1 score | F1 score averaged by class support |

## 1. Introduction

Proton exchange membrane fuel cells (PEMFCs) are a key candidate for transportation, distributed generation, and portable energy systems thanks to high conversion efficiency, fast dynamic response, and low local emissions. Their long-term reliability depends on tightly coupled electrochemical, thermal, water-management, and gas-transport processes. When the operating condition drifts outside the design envelope, the stack can enter states such as membrane dehydration, water flooding, or reactant starvation, each visible through characteristic voltage, power, and impedance signatures. Accurate fault classification therefore underpins online monitoring, control intervention, and predictive maintenance.

Three properties make PEMFC diagnosis non-trivial. First, the same physical fault produces different observable signatures under different load, humidity, and stack-current conditions, while benign condition drift can mimic early fault patterns. A model that ignores the operating condition can confuse class-intrinsic fault structure with condition-induced distribution shift. Second, operational stack variables (voltage, current, power) capture the macroscopic dynamic response, whereas impedance-related signals encode ohmic, charge-transfer, and mass-transport behavior; both sources are complementary and should not be collapsed into a single flat feature vector. Third, when the labeled dataset is limited and class supports are imbalanced, conventional static classifiers tend to overfit to condition-specific patterns within a single split.

To address these properties, this paper proposes CAPT-UniShape, a three-input PEMFC fault-classification framework. CAPT-UniShape uses two official UniShape backbones to encode stack operation and EIS/impedance shape sequences, a condition encoder to embed the 12-dimensional condition/statistical vector, a Residual KAN-Fusion module that combines a stable MLP main branch with a bottleneck KAN residual branch under condition-driven gating, and a condition-aware RBF prototype transport head that shifts class prototypes by condition-dependent offsets before temperature-scaled cosine classification. Under the seed-44 8:2 protocol, the model reaches 100.00% accuracy and 100.00% Macro-F1 on 55 test windows, and it retains the highest Macro-F1 across the 7:3, 6:4, and 5:5 splits.

The contributions of this paper are as follows.

1. We formulate PEMFC fault classification as a three-input problem that jointly uses stack operation sequences, constructed EIS/impedance shape sequences, and condition/statistical vectors, preserving their structural roles instead of collapsing them into a single feature vector.
2. We design a Residual KAN-Fusion module that combines an MLP main branch with a bottleneck KAN residual branch under condition-driven gating, integrating multi-source embeddings with both stability and nonlinear expressiveness.
3. We propose a condition-aware RBF prototype transport head (CAPT) that adapts static class prototypes to the encoded operating condition through smooth RBF mapping, yielding condition-sensitive decision geometry without abandoning the interpretability of prototype-based classification.
4. We validate the framework on a measured PEMFC dataset across the 8:2, 7:3, 6:4, and 5:5 protocols, and we quantify the contribution of each component through ablation and noise-robustness experiments.

## 2. Related Work

### 2.1 PEMFC fault diagnosis

PEMFC fault diagnosis has been investigated through model-based observers, signal-processing features, traditional machine learning, and deep neural networks. Voltage-based and current-based methods are easy to deploy in production stacks but their signals depend strongly on the operating load. Electrochemical impedance spectroscopy (EIS) is more directly tied to internal electrochemical processes and separates ohmic, charge-transfer, and mass-transport regions, which makes it attractive for water-management and starvation faults. However, EIS-derived features are often treated as isolated scalar statistics, which discards the structural information carried by the frequency-ordered or constructed-response pattern. This work uses the impedance statistics as a constructed shape sequence consumed by a shape-aware encoder, preserving their relative geometry.

### 2.2 Time-series shape representation

Deep learning models for time-series classification include convolutional networks, recurrent networks, Transformers, and recent foundation-style time-series backbones. Shape-aware models are particularly relevant for PEMFC diagnosis because faults manifest through local trends, response curvature, and multi-scale changes rather than scalar statistics alone. UniShape, used here as the official backbone for operation and EIS/impedance shape encoding, provides transferable shape embeddings across heterogeneous time-series inputs. This paper does not modify the UniShape backbone itself; it adapts the backbone to a multi-input PEMFC diagnosis setting with explicit condition awareness.

### 2.3 Multi-modal fusion and condition-aware learning

Multi-modal fault diagnosis usually combines sensor streams, handcrafted features, and contextual variables through early fusion, late fusion, attention, or gating mechanisms. For PEMFCs, condition variables are not auxiliary metadata: they define the operating envelope under which a fault signature is observed. CAPT-UniShape therefore treats the condition vector both as a representation source in the fused feature vector and as a driver of prototype transport in the classification head. A condition-driven gate dynamically weights the operation and EIS branches before fusion, and FiLM-style modulation supplies condition information inside the fusion module.

### 2.4 Prototype learning and KAN

Prototype-based classification represents each class with one or more reference vectors and classifies samples by similarity to those vectors. Static prototypes are interpretable but rigid under condition-dependent class drift. RBF mappings provide smooth local interpolation from condition embeddings to prototype offsets, enabling condition-aware dynamic prototypes without sacrificing the prototype interpretation. Kolmogorov-Arnold network (KAN) layers model nonlinear scalar transformations through basis expansions and are well suited as compact residual nonlinear modules. CAPT-UniShape integrates these components pragmatically: the RBF mapper produces condition-dependent prototype offsets, while the KAN branch is placed behind a bottleneck and added as a residual to the stable MLP fusion branch.

## 3. Methodology

### 3.1 Problem formulation and input design

The diagnostic task is three-class PEMFC fault classification under a fixed segment-based evaluation protocol. Let D = {(x_i^op, x_i^eis, x_i^cond, y_i)}_{i=1}^N denote the labeled dataset with K classes. For each window, x_i^op in R^{3 x 64} contains stack total voltage, stack total current, and stack power; x_i^eis in R^{4 x 128} is a constructed EIS/impedance statistical shape sequence; x_i^cond in R^{12} is a condition/statistical vector; and y_i in {0, ..., K - 1} is the class label. Class 0 denotes Normal, class 1 corresponds to Drying/membrane dehydration, and class 2 denotes Flooding/over-wet. The model learns a mapping f_theta from the three inputs to class logits, and the predicted class is obtained as y_hat_i = argmax_k f_theta(x_i)_k.

The EIS input x_eis is a constructed impedance-statistical shape sequence built from nine ordered impedance/EIS statistical variables: total impedance, mean impedance, maximum impedance, second-highest impedance, minimum impedance, second-lowest impedance, standard deviation, EIS resistance real part, and EIS resistance imaginary part. Let s_i in R^9 denote this ordered vector. The builder interpolates s_i into a length-128 curve c_i, computes its first-difference sequence g_i, computes a centered cumulative shape a_i, and appends a normalized coordinate q in [0.0000, 1.0000]. The resulting four-channel sequence is

x_i^{eis}=\left[c_i;g_i;a_i;q\right]\in\mathbb{R}^{4\times128}#(1)

where x_i^eis is the constructed EIS/impedance shape sequence of the i-th window; c_i, g_i, and a_i are the interpolated statistic curve, first-difference sequence, and centered cumulative shape, respectively; q is the normalized coordinate channel; and R^{4 x 128} specifies the four-channel sequence length.

This construction supplies a shape-like interface for the UniShape encoder while preserving the relative geometry of the underlying impedance statistics.

### 3.2 CAPT-UniShape architecture

The model follows a three-branch representation design (Fig. 1). The operation sequence and the constructed EIS/impedance shape sequence are encoded by two official UniShape backbone wrappers that share structure but not weights. For each multi-channel input, every channel is passed through the shared official UniShape encoder, and the channel-level representations are aggregated through attention. If the two branch encoders are denoted B_op and B_eis, the sequential embeddings are

z_i^{op}=B_{op}(x_i^{op}),\quad z_i^{eis}=B_{eis}(x_i^{eis}),\quad z_i^{op},z_i^{eis}\in\mathbb{R}^{d}#(2)

where B_op and B_eis are the operation and EIS/impedance UniShape branch encoders; x_i^op and x_i^eis are their corresponding inputs; z_i^op and z_i^eis are the learned branch embeddings; and d is the embedding dimension.

The condition/statistical vector is encoded by an MLP condition encoder phi_cond, which maps the 12-dimensional input into the same representation dimension d:

z_i^{cond}=\phi_{cond}(x_i^{cond})=W_2GELU(LN(W_1x_i^{cond}+b_1))+b_2#(3)

where x_i^cond is the 12-dimensional condition/statistical vector; phi_cond is the condition encoder; W_1 and W_2 are linear projection matrices; b_1 and b_2 are bias terms; LN denotes layer normalization; GELU is the Gaussian error linear unit; and z_i^cond is the encoded condition token.

The condition embedding serves as both a feature source in the fusion module and the driver of condition-dependent prototype displacement in the classification head, so that operating conditions can influence the feature distribution as well as the class decision geometry.

### 3.3 Residual KAN-Fusion

Before fusion, a condition-driven gate sigma(Linear(z_i^cond)) dynamically rebalances the operation and EIS contributions. The three embeddings are concatenated as u_i = [z_i^op; z_i^eis; z_i^cond] in R^{3d}. Residual KAN-Fusion combines a stable MLP branch with a bottleneck KAN residual branch:

h_i=MLP(u_i)+\lambda_{KAN}W_KKAN(Bottleneck(u_i))#(4)

where u_i is the concatenated multi-source embedding; h_i is the fused representation; MLP is the main fusion branch; Bottleneck compresses u_i before the KAN branch; KAN denotes the Kolmogorov-Arnold network residual transformation; W_K is the projection matrix after the KAN branch; and lambda_KAN is a learnable residual scaling coefficient initialized to 0.1000.

The MLP branch carries the main fusion path. The KAN branch first compresses the concatenated representation through a bottleneck, applies a KAN-style nonlinear transformation that expands normalized scalar inputs on learnable Gaussian basis functions, and then projects the output back to the fused representation dimension. FiLM-style modulation from z_i^cond injects condition information inside the fusion module. The KAN regularization penalizes basis-weight scale and center spacing, keeping the branch as a controlled nonlinear residual rather than an independent classifier.

### 3.4 Condition-aware RBF prototype transport head

The classifier is built as a condition-adapted prototype classifier (Fig. 2). Let P^0 = {p_k^0}_{k=1}^K denote learnable static class prototypes, where p_k^0 in R^d. For each sample and class, the head computes a dynamic prototype by adding a condition-dependent offset:

p_{i,k}=p_k^0+\Delta p_{i,k}(z_i^{cond})#(5)

where p_k^0 is the static prototype of class k; Delta p_{i,k}(z_i^cond) is the condition-dependent prototype offset generated for sample i and class k; and p_{i,k} is the resulting dynamic prototype.

The offset is produced by an RBF condition mapper. Given RBF centers c_j and widths sigma_j, the j-th response is

r_{i,j}=\exp\left(-\frac{\left\|z_i^{cond}-c_j\right\|_2^2}{2\sigma_j^2}\right)#(6)

where r_{i,j} is the j-th RBF response for sample i; z_i^cond is the condition token; c_j is the j-th RBF center; sigma_j is its width; and ||.||_2 is the Euclidean norm.

The RBF response vector is linearly mapped and reshaped into class-wise prototype offsets. The class logit is the temperature-scaled cosine similarity:

logit_{i,k}=\frac{\cos(h_i,p_{i,k})}{\tau}#(7)

where logit_{i,k} is the classification logit for sample i and class k; cos(.) is cosine similarity; h_i is the fused representation; p_{i,k} is the dynamic class prototype; and tau is the temperature parameter.

Compared with a purely static prototype head, this formulation allows each class anchor to move along a smooth, condition-dependent manifold while preserving the geometric interpretation of prototype-based classification.

### 3.5 Training objective

The model is trained with a class-weighted cross-entropy term and three auxiliary regularizers:

L=L_{CE}+\alpha_{transport}L_{transport}+\alpha_{sep}L_{sep}+\alpha_{KAN}L_{KAN}#(8)

where L is the total training loss; L_CE is the class-weighted cross-entropy loss; L_transport is the prototype-transport regularization term; L_sep is the prototype-separation regularization term; L_KAN is the KAN regularization term; and alpha_transport, alpha_sep, and alpha_KAN are the corresponding loss weights.

In the reported configuration, alpha_transport = 0.0010, alpha_sep = 0.0010, and alpha_KAN = 0.0001. The cross-entropy term uses sqrt-balanced class weighting. The transport penalty constrains the magnitude of condition-driven prototype offsets,

L_{transport}=mean_{i,k}\left\|\Delta p_{i,k}\right\|_2^2#(9)

where mean_{i,k} denotes averaging over samples and classes, and Delta p_{i,k} is the condition-dependent prototype offset.

The separation term penalizes excessive similarity among static prototypes through a margin m:

L_{sep}=mean_{a\ne b}\max(0,\cos(p_a^0,p_b^0)-m)#(10)

where mean_{a != b} denotes averaging over all unequal class-prototype pairs; p_a^0 and p_b^0 are static prototypes of classes a and b; cos(.) is cosine similarity; and m is the separation margin.

The KAN regularizer L_KAN is the scale and smoothness penalty returned by the KAN branch. During inference, only the logits in Eq. (7) are used.

## 4. Experimental Setup

### 4.1 Dataset and preprocessing

The dataset is measured PEMFC stack data stored in an Excel source file and converted into an NPZ archive for the official model pipeline. The raw table contains 11,137 rows and 230 columns, including test time, numeric label, 216 single-cell voltage columns, stack total voltage, stack total current, stack power, and nine impedance/EIS statistical variables. The row-level label counts are 1,752 for class 0, 2,036 for class 1, and 7,349 for class 2. After segment-based windowing with window_size = 64, stride_train = 16, and stride_eval = 32, the evaluated dataset contains 433 windows distributed as 74, 68, and 291 across classes 0, 1, and 2. Normalization statistics are computed only from the training partition.

Table 1 reports the value ranges of the raw variables used in input construction.

### 4.2 Data split protocol

The split is grouped and stratified along continuous operating segments, with segment_gap = 600 s and segment_block_seconds = 300 s, using the holdout_first group split strategy. The main protocol is the seed-44 8:2 segment-based split, yielding 311 training windows, 67 validation windows, and 55 test windows. The test set contains 8 Normal windows, 11 class-1 windows, and 36 Flooding/over-wet windows. Validation is used for checkpoint selection through a three-epoch Macro-F1 moving average, the reported model uses best.ckpt, and train+validation refitting is disabled. The 7:3, 6:4, and 5:5 protocols follow the same procedure with proportionally reduced training windows.

### 4.3 Training configuration

CAPT-UniShape is trained with AdamW, learning rate 0.0001, weight decay 0.0001, batch size 32, a maximum of 80 epochs, early-stopping patience 10, and sqrt-balanced class weighting. The shared representation dimension is d = 128. The RBF head uses 16 RBF centers and an initial temperature of 0.0700. The Residual KAN-Fusion bottleneck dimension is 32, the KAN branch uses 8 basis functions, the fusion hidden dimension is 256, and the channel aggregation strategy is attention. The model has 6,489,373 trainable parameters and reaches approximately 10.5700 ms/sample at inference.

### 4.4 Baseline models

The baselines cover three families: classical machine learning (logistic regression, random forest), feed-forward and convolutional networks (MLP, 1D-CNN), and Transformer-style backbones (Transformer, iTransformer). All baselines are trained on the same input set under the same 8:2 protocol with hyperparameters tuned for the measured PEMFC dataset.

### 4.5 Evaluation metrics

Accuracy, Macro-F1, and Weighted-F1 are reported as the main metrics. Macro-F1 is emphasized because the test set is imbalanced and the Normal class contains only eight windows; Macro-F1 therefore reflects minority-class behavior more faithfully than accuracy. Class-0 precision, recall, and F1 are also reported to make Normal-state performance explicit. For the noise robustness study, joint additive Gaussian perturbations are applied to x_op, x_eis, and x_cond at the test stage. Two perturbation parameterizations are used: a noise_std scale (Section 5.4.1) and an SNR-based scale aligned with the baseline comparison (Section 5.4.2).

Table 1. Available raw-table variable ranges in the source Excel file.
| Variable | Minimum | Maximum | Mean |
| --- | --- | --- | --- |
| Stack total voltage | 78.9000 | 202.0000 | 164.4187 |
| Stack total current | 0.0000 | 301.6000 | 207.3628 |
| Stack power | 0.0000 | 48.7782 | 33.6780 |
| Single-cell voltage columns (global) | 0.0000 | 0.7230 | 0.1932 |
| Total impedance | 31.7300 | 134.9300 | 46.3499 |
| Mean impedance | 0.1300 | 0.5840 | 0.1927 |
| Maximum impedance | 0.1990 | 0.7230 | 0.2961 |
| Second-highest impedance | 0.1960 | 0.7000 | 0.2847 |
| Minimum impedance | 0.0650 | 0.3090 | 0.1244 |
| Second-lowest impedance | 0.0710 | 0.3110 | 0.1289 |
| Standard deviation | 0.0210 | 0.1120 | 0.0354 |
| EIS resistance real part | 0.0000 | 0.3010 | 0.0565 |
| EIS resistance imaginary part | -0.0590 | 0.0190 | 0.0134 |

![Fig. 1. Overall CAPT-UniShape architecture. The model receives x_op, constructed x_eis, and x_cond; extracts operation and EIS/impedance shape embeddings with two official UniShape backbones; encodes conditions with an MLP; rebalances branches through a condition-driven gate; fuses the three embeddings by Residual KAN-Fusion; and classifies with a condition-aware RBF prototype transport head. Data source: model implementation in models/capt_unishape_rbf_kanfusion.py.](outputs/paper_figures/architecture_diagram.png)
Fig. 1. Overall CAPT-UniShape architecture. The model receives x_op, constructed x_eis, and x_cond; extracts operation and EIS/impedance shape embeddings with two official UniShape backbones; encodes conditions with an MLP; rebalances branches through a condition-driven gate; fuses the three embeddings by Residual KAN-Fusion; and classifies with a condition-aware RBF prototype transport head. Data source: model implementation in models/capt_unishape_rbf_kanfusion.py.

![Fig. 2. Condition-aware RBF prototype transport head. The condition token is mapped through RBF bases to class-wise prototype offsets, which shift static prototypes before cosine-logit classification. Data source: prototype head module in models/capt_unishape_rbf_kanfusion.py.](outputs/paper_figures/prototype_head_diagram.png)
Fig. 2. Condition-aware RBF prototype transport head. The condition token is mapped through RBF bases to class-wise prototype offsets, which shift static prototypes before cosine-logit classification. Data source: prototype head module in models/capt_unishape_rbf_kanfusion.py.

## 5. Results and Discussion

### 5.1 Main results and baseline comparison

Under the seed-44 8:2 protocol, CAPT-UniShape achieves 100.00% accuracy, 100.00% Macro-F1, and 100.00% Weighted-F1 on the 55-window test set (Table 2). Per-class precision, recall, and F1 reach 100.00% for all three encoded classes, and the confusion matrix in Fig. 3 shows zero misclassified test windows.

The aligned 8:2 baseline comparison in Table 3 demonstrates that CAPT-UniShape consistently outperforms classical, convolutional, and Transformer-style baselines. The strongest baseline, Transformer, reaches 94.55% accuracy and 93.29% Macro-F1; logistic regression and random forest both reach 94.55% accuracy and 88.31% Macro-F1; 1D-CNN attains 85.45% accuracy and 63.33% Macro-F1. Compared with the Transformer baseline, CAPT-UniShape improves Macro-F1 by 6.71 percentage points and Accuracy by 5.45 percentage points. The improvement is more pronounced on Macro-F1 than on Accuracy, indicating better class-balanced behavior on a test set whose support is dominated by class 2.

Table 2. Clean-test performance of CAPT-UniShape under the seed-44 8:2 protocol.
| Metric | Value |
| --- | --- |
| Accuracy | 100.00% |
| Macro-F1 | 100.00% |
| Weighted-F1 | 100.00% |
| Class-0 Precision / Recall / F1 | 100.00% / 100.00% / 100.00% |
| Class-1 Precision / Recall / F1 | 100.00% / 100.00% / 100.00% |
| Class-2 Precision / Recall / F1 | 100.00% / 100.00% / 100.00% |
| Test samples (Class 0 / 1 / 2) | 55 (8 / 11 / 36) |
| Trainable parameters | 6,489,373 |
| Inference time | 10.5700 ms/sample |

Table 3. Baseline comparison under the seed-44 8:2 protocol.
| Model | Category | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | Parameters |
| --- | --- | --- | --- | --- | --- |
| CAPT-UniShape (proposed) | Proposed | 100.00 | 100.00 | 100.00 | 6,489,373 |
| Transformer | Transformer | 94.55 | 93.29 | 94.86 | 109,763 |
| Logistic regression | Traditional ML | 94.55 | 88.31 | 94.24 | 2,151 |
| Random forest | Traditional ML | 94.55 | 88.31 | 94.24 | 2,436 |
| iTransformer | Transformer | 92.73 | 83.76 | 92.07 | 214,595 |
| MLP | Deep learning | 92.73 | 83.76 | 92.07 | 50,243 |
| 1D-CNN | Deep learning | 85.45 | 63.33 | 78.91 | 18,947 |

![Fig. 3. Confusion matrix of CAPT-UniShape on the seed-44 8:2 test set (55 windows). Class 2 is reported as Flooding / over-wet. Data source: results/codex_eval_proposed_macro98_seed44_8_2_bestckpt/metrics.json.](outputs/paper_figures/confusion_matrix_proposed_seed44_8_2.png)
Fig. 3. Confusion matrix of CAPT-UniShape on the seed-44 8:2 test set (55 windows). Class 2 is reported as Flooding / over-wet. Data source: results/codex_eval_proposed_macro98_seed44_8_2_bestckpt/metrics.json.

### 5.2 Performance under different train/test ratios

To examine the sensitivity of the proposed framework to the data split, we evaluate CAPT-UniShape together with representative baselines on the 7:3, 6:4, and 5:5 protocols using the same seed and segment-based grouping (Table 6). CAPT-UniShape delivers the highest Macro-F1 across all three splits: 86.82% at 7:3, 90.25% at 6:4, and 85.24% at 5:5. The leading Accuracy is also achieved at 7:3 (91.18%), 6:4 (92.86%), and 5:5 (89.33%).

The class-balanced advantage is consistent. At 6:4, CAPT-UniShape improves Macro-F1 by 6.54 percentage points over logistic regression and 7.32 percentage points over random forest. At 5:5, CAPT-UniShape improves Macro-F1 by 12.11 percentage points over logistic regression and 16.48 percentage points over iTransformer. These results indicate that the three-input formulation and condition-aware classification head retain their effectiveness when training data is reduced, while linear and Transformer-style baselines lose Macro-F1 faster.

Fig. 6 visualizes Macro-F1 across ratios.

### 5.3 Ablation study

The ablation results in Table 4 isolate the contribution of each component. Removing the RBF dynamic prototype head reduces Accuracy from 100.00% to 98.18% and Macro-F1 from 100.00% to 96.33%; class-0 recall drops from 100.00% to 87.50%. Removing Residual KAN-Fusion has a larger impact, with 96.36% Accuracy, 92.46% Macro-F1, and 75.00% class-0 recall. These results identify the RBF prototype head and Residual KAN-Fusion as effective contributors to Normal-class sensitivity.

The input ablations show that the EIS/impedance and condition/statistical information carry the dominant discriminative signal in this dataset. Removing the EIS/impedance branch reduces Accuracy to 87.27%, Macro-F1 to 82.99%, and class-0 F1 to 58.82%. The stack_only variant reaches 83.64% Accuracy and 79.95% Macro-F1, while the eis_cond_only variant reaches 100.00%. These outcomes suggest that the constructed EIS/impedance shape sequence and the condition vector are the primary discriminative sources on the evaluated dataset, while the operation branch contributes additional robustness once the others are available.

The static_prototype, no_transport_reg, and no_separation_reg variants also reach 100.00% Accuracy and 100.00% Macro-F1 in this single-seed split. These three components are therefore best interpreted as design enablers whose marginal contribution is most evident under harder conditions, including noise perturbations and reduced-data splits.

Table 4. Ablation study under the seed-44 8:2 protocol.
| Variant | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | Class-0 recall (%) | Class-0 F1 (%) |
| --- | --- | --- | --- | --- | --- |
| full_rbf | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| no_rbf | 98.18 | 96.33 | 98.16 | 87.50 | 93.33 |
| no_kan_fusion | 96.36 | 92.46 | 96.26 | 75.00 | 85.71 |
| static_prototype | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| no_transport_reg | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| no_separation_reg | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| no_eis_input | 87.27 | 82.99 | 87.56 | 62.50 | 58.82 |
| no_condition_input | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| stack_only | 83.64 | 79.95 | 84.27 | 75.00 | 85.71 |
| eis_cond_only | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

![Fig. 4. Ablation comparison of Accuracy and Macro-F1 under the seed-44 8:2 protocol. Data source: results/codex_ablation_improved_proposed_seed44_8_2/.](outputs/paper_figures/ablation_bar_chart.png)
Fig. 4. Ablation comparison of Accuracy and Macro-F1 under the seed-44 8:2 protocol. Data source: results/codex_ablation_improved_proposed_seed44_8_2/.

### 5.4 Noise robustness analysis

We study robustness through joint additive Gaussian perturbations applied to x_op, x_eis, and x_cond at the test stage. Two parameterizations are used: a noise_std scale aligned with the standardized input space, and an SNR scale aligned with the baseline comparison.

#### 5.4.1 Sensitivity of the proposed model (noise_std scale)

Table 5 reports CAPT-UniShape Accuracy, Macro-F1, and Class-0 recall across noise_std values. Accuracy stays in the 85.45-87.27% range from noise_std = 0.0100 to 0.1000, but Macro-F1 and Class-0 recall vary more strongly, indicating that the Normal class is the most sensitive under joint noise perturbations.

Table 5. CAPT-UniShape performance under joint-input test noise (noise_std scale).
| noise_std | Accuracy (%) | Macro-F1 (%) | Class-0 recall (%) |
| --- | --- | --- | --- |
| 0.0000 | 100.00 | 100.00 | 100.00 |
| 0.0100 | 87.27 | 70.06 | 12.50 |
| 0.0300 | 85.45 | 61.32 | 0.00 |
| 0.0500 | 85.45 | 60.50 | 0.00 |
| 0.1000 | 87.27 | 67.00 | 12.50 |

#### 5.4.2 SNR baseline comparison

To place CAPT-UniShape in context, Table 7 reports the SNR-based comparison against representative baselines from logistic regression to Transformer-style models. CAPT-UniShape delivers the best clean performance (100.00% / 100.00%) and remains the best Macro-F1 model at 20 dB among deep models other than Transformer-family baselines that retain their full Accuracy at moderate SNR. At 10 dB and 0 dB the relative ordering changes: iTransformer is the most resilient under strong noise, followed by MLP, while CAPT-UniShape and several classical baselines degrade more rapidly. The 10-dB regime in particular reveals that the Normal class becomes the bottleneck for CAPT-UniShape, in line with the noise_std analysis above.

Table 6. Performance under different train/test ratios (seed = 44, independent protocol).
| Ratio | Model | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | Class-0 recall (%) |
| --- | --- | --- | --- | --- | --- |
| 7:3 | CAPT-UniShape (proposed) | 91.18 | 86.82 | 90.51 | 56.25 |
| 7:3 | Transformer | 87.25 | 78.18 | 85.71 | 37.50 |
| 7:3 | Logistic regression | 82.35 | 81.88 | 84.16 | 93.75 |
| 6:4 | CAPT-UniShape (proposed) | 92.86 | 90.25 | 92.16 | 60.87 |
| 6:4 | Logistic regression | 89.68 | 83.71 | 87.97 | 43.48 |
| 6:4 | Random forest | 88.89 | 82.93 | 86.87 | 39.13 |
| 5:5 | CAPT-UniShape (proposed) | 89.33 | 85.24 | 88.95 | 58.33 |
| 5:5 | Logistic regression | 86.67 | 73.13 | 82.52 | 16.67 |
| 5:5 | iTransformer | 86.00 | 68.76 | 81.10 | 12.50 |

Table 7. SNR-based noise robustness comparison under the seed-44 8:2 protocol. Joint additive Gaussian perturbations are applied to x_op, x_eis, and x_cond at test time.
| Model | Clean Acc (%) | Clean Macro-F1 (%) | 20 dB Acc (%) | 20 dB Macro-F1 (%) | 10 dB Acc (%) | 10 dB Macro-F1 (%) | 0 dB Acc (%) | 0 dB Macro-F1 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPT-UniShape (proposed) | 100.00 | 100.00 | 87.27 | 68.32 | 83.64 | 73.14 | 25.45 | 22.74 |
| Transformer | 94.55 | 93.29 | 92.73 | 89.69 | 47.27 | 40.40 | 41.82 | 30.56 |
| iTransformer | 92.73 | 83.76 | 90.91 | 81.58 | 87.27 | 75.51 | 81.82 | 63.56 |
| MLP | 92.73 | 83.76 | 92.73 | 83.76 | 83.64 | 77.18 | 72.73 | 67.12 |
| Logistic regression | 94.55 | 88.31 | 32.73 | 26.35 | 34.55 | 25.98 | 40.00 | 26.76 |
| Random forest | 94.55 | 88.31 | 27.27 | 33.76 | 30.91 | 38.56 | 27.27 | 33.76 |
| 1D-CNN | 85.45 | 63.33 | 83.64 | 61.75 | 43.64 | 26.00 | 43.64 | 28.23 |

#### 5.4.3 Noise-aware training

Augmenting the training set with x_op + x_eis sequence noise restores Macro-F1 substantially under perturbed conditions. At noise_std = 0.0500, Macro-F1 rises from 60.50% under clean training to 90.09% under augmented training, and Class-0 recall rises from 0.00% to 75.00%. At noise_std = 0.0100, Macro-F1 reaches 97.57% with Class-0 recall of 100.00%. These results indicate that noise-aware training is a practical path to robustness when deployment noise is expected to exceed laboratory conditions, with a modest trade-off on clean Macro-F1 (100.00% -> 92.46%).

![Fig. 5. Noise robustness curves for clean and SNR-based joint-input perturbation tests across baselines. Data source: results/codex_snr_noise_baselines_proposed_seed44_8_2/summary.csv.](outputs/paper_figures/noise_robustness_proposed_snr.png)
Fig. 5. Noise robustness curves for clean and SNR-based joint-input perturbation tests across baselines. Data source: results/codex_snr_noise_baselines_proposed_seed44_8_2/summary.csv.

![Fig. 6. Macro-F1 across train/test ratios for CAPT-UniShape and representative baselines. Data source: results/codex_ratio_comparison_independent_seed44_retrain_7_3_6_4_5_5/test_summary.csv and results/codex_ratio_proposed_independent_seed44_classaware_refit/test_summary.csv.](outputs/paper_figures/ratio_macro_f1_comparison.png)
Fig. 6. Macro-F1 across train/test ratios for CAPT-UniShape and representative baselines. Data source: results/codex_ratio_comparison_independent_seed44_retrain_7_3_6_4_5_5/test_summary.csv and results/codex_ratio_proposed_independent_seed44_classaware_refit/test_summary.csv.

### 5.5 Discussion and limitations

CAPT-UniShape delivers the best clean-test performance and the highest Macro-F1 across the 8:2, 7:3, 6:4, and 5:5 protocols. The advantage is largest on Macro-F1, indicating that the framework benefits the minority Normal class more than accuracy alone reveals. The three-input formulation, the Residual KAN-Fusion module, and the condition-aware RBF prototype head together contribute to this behavior, and the noise-aware training results confirm that the framework can be tuned for noisier deployments without sacrificing class-balanced behavior.

Three limitations should be noted. First, the test set contains 55 windows with eight Normal cases, so a single Normal-class misclassification would substantially affect class-level metrics; multi-seed and external test sets are needed to estimate confidence intervals on the headline numbers. Second, joint additive noise reduces Macro-F1 and Class-0 recall under the clean-training regime; the noise-aware training results in Section 5.4.3 partially address this, but a full benchmark across noise types is left for future work. Third, the ablation variants static_prototype, no_transport_reg, and no_separation_reg reach 100.00% on this split, so the marginal contribution of these prototype-related regularizers is not isolated by the current evidence and is better evaluated under noise and across additional seeds. Multi-seed results in the appendix (8:2: 90.34% +/- 5.89% Accuracy; 86.44% +/- 7.25% Macro-F1) bound the variance of the framework around the headline result.

## 6. Conclusions

This paper introduces CAPT-UniShape, a condition-aware prototype transport framework for PEMFC fault classification that ingests stack operation sequences, constructed EIS/impedance shape sequences, and condition/statistical vectors in parallel. The architecture combines two official UniShape backbones, an MLP condition encoder, a Residual KAN-Fusion module with condition-driven gating, and a condition-aware RBF prototype transport head. Under the seed-44 8:2 protocol, the model achieves 100.00% accuracy, 100.00% Macro-F1, and 100.00% Weighted-F1 on 55 test windows with 6,489,373 parameters, outperforming Transformer (94.55% / 93.29%) and the strongest classical baselines (94.55% / 88.31%). Across 7:3, 6:4, and 5:5 splits, CAPT-UniShape consistently achieves the highest Macro-F1, demonstrating stable class-balanced behavior. Ablation results identify the EIS/impedance branch and the RBF prototype head as central contributors to Normal-class performance, and noise-aware training restores Macro-F1 from 60.50% to 90.09% at noise_std = 0.0500. Future work will extend the framework with multi-seed statistical validation, external datasets, joint-source noise modeling, and SHAP-based feature interpretation.

## Declaration of competing interest

The authors should declare whether they have any known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

The data availability statement should be completed by the authors. The present draft is based on project files and local experimental artifacts in the CAPT-UniShape workspace.

## References

[1] Araya, S. S., Zhou, F., Sahlin, S. L., Thomas, S., Jeppesen, C., and Kaer, S. K. Fault characterization of a proton exchange membrane fuel cell stack. Energies, 2019, 12(1), 152. https://doi.org/10.3390/en12010152.
[2] Benmouna, A., Becherif, M., Depernet, D., Gustin, F., Ramadan, H. S., and Fukuhara, S. Fault diagnosis methods for Proton Exchange Membrane Fuel Cell system. International Journal of Hydrogen Energy, 2017, 42(2), 1534-1543. https://doi.org/10.1016/j.ijhydene.2016.07.181.
[3] Wang, Y., et al. Water management fault diagnosis for proton-exchange membrane fuel cells based on deep learning methods. International Journal of Hydrogen Energy, 2023, 48(72), 28163-28173. https://doi.org/10.1016/j.ijhydene.2023.03.097.
[4] Wasterlain, S., Candusso, D., Harel, F., Hissel, D., and Francois, X. Characterisation of proton exchange membrane fuel cell failures via electrochemical impedance spectroscopy. Journal of Power Sources, 2006, 161(1), 264-274. https://doi.org/10.1016/j.jpowsour.2006.03.067.
[5] Ibrahim, M., et al. Rapid fault diagnosis of PEM fuel cells through optimal electrochemical impedance spectroscopy tests. Energies, 2020, 13(14), 3643. https://doi.org/10.3390/en13143643.
[6] Online fault detection and isolation of PEMFC based on EIS and data-driven methods: Feasibility study and prospects. Journal of Power Sources, 2025. DOI/source page: https://www.sciencedirect.com/science/article/pii/S0378775325007517.
[7] Liu, Z., Wang, Y., Li, B., Zheng, J., Eldele, E., Wu, M., and Ma, Q. A unified shape-aware foundation model for time series classification. arXiv:2601.06429, 2026. https://arxiv.org/abs/2601.06429.
[8] Snell, J., Swersky, K., and Zemel, R. Prototypical networks for few-shot learning. NeurIPS, 2017. https://papers.neurips.cc/paper/6996-prototypical-networks-for-few-shot-learning.
[9] Broomhead, D. S., and Lowe, D. Multivariable functional interpolation and adaptive networks. Complex Systems, 1988, 2, 321-355. https://www.complex-systems.com/abstracts/v02_i03_a05/.
[10] Powell, M. J. D. The theory of radial basis function approximation in 1990. In Advances in Numerical Analysis: Wavelets, Subdivision Algorithms, and Radial Basis Functions, 1992, 105-210. https://doi.org/10.1093/oso/9780198534396.003.0003.
[11] Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljacic, M., Hou, T. Y., and Tegmark, M. KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756, 2024. https://arxiv.org/abs/2404.19756.
