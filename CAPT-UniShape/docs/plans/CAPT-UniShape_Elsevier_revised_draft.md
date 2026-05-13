# CAPT-UniShape: Condition-Aware Prototype Transport UniShape for PEMFC Fault Classification

## Highlights
- CAPT-UniShape combines stack operation sequences, EIS/impedance shape sequences, and condition/statistical vectors for PEMFC fault classification.
- Residual KAN-Fusion uses a stable MLP branch plus a bottleneck KAN residual branch to model nonlinear cross-modal interactions.
- A condition-aware RBF prototype transport head adapts class prototypes from the encoded condition token before cosine classification.
- Under the evaluated seed-44 fixed 8:2 test protocol, the model achieved 100.00% accuracy and 100.00% Macro-F1 on 55 test windows.
- Noise experiments show that joint input perturbations reduce Macro-F1 and expose sensitivity of the normal class, so strong-noise robustness remains an open issue.

## Abstract
Reliable fault classification is important for proton exchange membrane fuel cell (PEMFC) systems because water management faults, reactant starvation, and condition-dependent operating drift can reduce efficiency and accelerate degradation. Existing data-driven diagnostic models often rely on a single signal source or a static classification boundary, which may be insufficient when operational time series, impedance-related responses, and operating conditions jointly shape the observed fault signature. This paper presents CAPT-UniShape, a Condition-Aware Prototype Transport UniShape model for PEMFC fault classification. The model receives three inputs: stack operation sequences x_op in R^{N x 3 x 64}, constructed EIS/impedance statistical shape sequences x_eis in R^{N x 4 x 128}, and condition/statistical vectors x_cond in R^{N x 12}. Two official UniShape backbones extract operation and EIS shape embeddings, while a condition encoder maps x_cond into a condition token. The concatenated embeddings are fused by Residual KAN-Fusion, which combines an MLP branch with a bottleneck KAN residual branch. The final classifier uses a condition-aware RBF prototype transport head, where each static class prototype is shifted by a condition-dependent offset before cosine-logit classification. Under the evaluated seed-44 fixed 8:2 test protocol, Official-CAPT-UniShape-RBF-KANFusion achieved 100.00% test accuracy, 100.00% test Macro-F1, and 100.00% test Weighted-F1 on 55 test windows, with 6,489,373 trainable parameters. Ablation results indicate that the EIS/impedance branch is critical, and that RBF prototypes and Residual KAN-Fusion improve the normal-class recall in the evaluated split. However, static prototypes and removal of transport or separation regularization also reached 100.00%, so their marginal contribution should not be overstated. Joint-input noise experiments further show marked Macro-F1 degradation and normal-class sensitivity, indicating that robust training and broader validation are needed before drawing general claims.

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

Proton exchange membrane fuel cells (PEMFCs) are attractive power sources for transportation, distributed generation, and portable energy systems because of their high conversion efficiency, rapid dynamic response, and low local emissions [1-3]. Their practical reliability, however, is constrained by tightly coupled electrochemical, thermal, water-management, and gas-transport processes. When the operating condition departs from a suitable range, the stack may enter states such as membrane dehydration, water flooding, reactant starvation, or other condition-dependent degradation patterns [1-3]. These faults can appear as voltage fluctuations, power loss, impedance changes, or shifts in operating statistics. Accurate fault classification is therefore a prerequisite for online monitoring, control intervention, and maintenance planning.
A central difficulty in PEMFC fault diagnosis is that the same physical fault can present different observable signatures under different load, humidity, and stack-current conditions. Conversely, changes caused only by benign operating-condition drift may resemble early fault patterns. A model that ignores condition information may therefore confuse class-intrinsic fault structure with condition-induced distribution shift. This issue is especially important when the available dataset is limited and class supports are imbalanced, because the classifier may overfit to condition-specific patterns in a single split.
Operational stack variables and electrochemical impedance-related features provide complementary diagnostic information. Stack voltage, current, and power describe the macroscopic temporal response of the system, while EIS or impedance-derived shape sequences encode information related to ohmic resistance, charge-transfer behavior, and mass-transport effects [4-6]. In the present project, the model input explicitly separates these sources into x_op, x_eis, and x_cond. This design avoids reducing all information to a single flat feature vector, while still allowing the model to learn cross-modal interactions.
Recent time-series representation models, including shape-oriented backbone models such as UniShape, provide a useful basis for extracting transferable shape embeddings from sequential data [7]. Nevertheless, direct fine-tuning of a generic time-series classifier does not by itself address the condition drift problem. For PEMFC fault classification, a useful model should not only extract operation and impedance shapes, but also adapt its decision geometry according to the current operating condition.
This paper proposes CAPT-UniShape, a Condition-Aware Prototype Transport UniShape framework. CAPT-UniShape uses two official UniShape backbones to encode the operation and EIS/impedance sequences, a condition encoder to embed x_cond, a Residual KAN-Fusion module to model nonlinear interactions among the three embeddings, and a condition-aware RBF prototype transport head to generate dynamic class prototypes. In the evaluated seed-44 split, this design obtains 100.00% clean-test accuracy under the fixed test protocol, but the analysis is deliberately conservative because the test set contains only 55 windows and the noise experiments reveal sensitivity of the normal class.
The contributions of this paper are as follows. First, a three-input PEMFC fault-classification formulation is established using operation sequences, EIS/impedance shape sequences, and condition/statistical vectors. Second, CAPT-UniShape integrates official UniShape feature extraction with Residual KAN-Fusion and condition-aware RBF prototype transport. Third, a controlled ablation study is reported to separate the effects of the RBF head, Residual KAN-Fusion, prototype regularization, and input branches. Fourth, joint-input noise experiments are discussed explicitly, including failure modes under noisy conditions, rather than presenting the clean-test result as a broad robustness claim.
PEMFC fault diagnosis. PEMFC fault diagnosis has been studied using model-based observers, signal-processing features, traditional machine learning, and deep neural networks [1-3]. Voltage-based and current-based methods are easy to deploy, but their signals can be highly load-dependent. EIS-based methods are more directly connected to internal electrochemical processes and can distinguish changes in ohmic, charge-transfer, and mass-transport regions [4-6]. However, EIS measurements or impedance-derived features are often used as isolated features rather than as structured shape sequences, which may discard information in their ordered frequency or constructed-response pattern.
Time-series shape representation. Deep learning models for time-series classification include convolutional networks, recurrent networks, Transformers, and more recent foundation-style time-series backbones [7]. Shape-oriented models are relevant for PEMFC diagnosis because faults may be expressed through local trends, response curvature, and multi-scale changes rather than only through scalar statistics. UniShape is used in this work as the official backbone for operation and EIS/impedance shape encoding [7]. The present study does not claim to improve the UniShape backbone itself; instead, it adapts the backbone to a multi-input PEMFC diagnosis setting.
Multi-modal fusion under operating-condition drift. Multi-modal fault diagnosis commonly combines sensor streams, handcrafted features, and contextual variables through early fusion, late fusion, attention, or gating mechanisms [1-3]. For PEMFCs, condition variables are not merely auxiliary metadata. They define the operating envelope under which a fault signature is observed. CAPT-UniShape therefore treats x_cond as both a representation source in the fused feature vector and a driver of prototype transport in the classification head.
Prototype learning, RBF mapping, and KAN components. Prototype-based classification represents each class by one or more reference vectors and classifies samples by similarity to those vectors [8]. Static prototypes can be interpretable, but they may be too rigid under condition-dependent class drift. RBF mappings provide smooth local interpolation from condition embeddings to prototype offsets [9,10]. KAN-style layers model nonlinear scalar transformations through basis expansions and can be used as compact residual nonlinear modules [11]. In CAPT-UniShape, these ideas are used pragmatically: the RBF mapper produces condition-dependent prototype offsets, while the KAN branch is placed behind a bottleneck and added as a residual to a stable MLP fusion branch.

## 2. Materials and methods

Data representation and diagnostic setting. The diagnostic task considered in this study is three-class PEMFC fault classification under a fixed segment-based evaluation protocol. The available data contain stack-level operation variables, impedance/EIS-derived statistics, and condition-related statistical variables. These sources describe different aspects of the same operating state: stack voltage, current, and power capture the macroscopic dynamic response of the system; impedance-related variables reflect electrochemical and transport behavior; and condition/statistical variables provide the operating context under which the observed response occurs. CAPT-UniShape is designed around this separation instead of collapsing all variables into a single flat feature vector.
Let D = {(x_i^op, x_i^eis, x_i^cond, y_i)}_{i=1}^N denote the labeled dataset with K classes. For each window, x_i^op in R^{3 x 64} contains stack total voltage, stack total current, and stack power; x_i^eis in R^{4 x 128} is a constructed EIS/impedance statistical shape sequence; x_i^cond in R^{12} is a condition/statistical vector; and y_i in {0, ..., K - 1} is the class label. In the current manuscript, class 0 denotes Normal, class 1 is reported as Drying/membrane dehydration according to the available paper materials, and class 2 denotes Flooding/over-wet according to the author's confirmation. The model learns a mapping f_theta from the three inputs to class logits, and the predicted class is obtained as y_hat_i = argmax_k f_theta(x_i)_k.
Construction of the EIS/impedance shape input. The input x_eis used in this work should be interpreted as a constructed impedance-statistical shape sequence, not as a raw full-frequency EIS spectrum. It is generated from nine ordered impedance/EIS statistical variables: total impedance, mean impedance, maximum impedance, second-highest impedance, minimum impedance, second-lowest impedance, standard deviation, EIS resistance real part, and EIS resistance imaginary part. Let s_i in R^9 denote this ordered vector. The builder interpolates s_i into a length-128 curve c_i, computes its first-difference sequence g_i, computes a centered cumulative shape a_i, and appends a normalized coordinate q in [0, 1]. The resulting four-channel sequence is
x_i^{eis}=\left[c_i;g_i;a_i;q\right]\in\mathbb{R}^{4\times128}#(1)
where x_i^eis denotes the constructed EIS/impedance shape sequence of the i-th window; c_i, g_i, and a_i denote the interpolated statistic curve, first-difference sequence, and centered cumulative shape, respectively; q is the normalized coordinate channel; and R^{4 x 128} specifies the four-channel sequence length.
This construction provides a shape-like interface for the UniShape encoder while keeping the data description consistent with the available project files. If raw frequency-resolved EIS spectra are introduced in a later revision, the data description and corresponding claims should be updated accordingly.
CAPT-UniShape architecture. The model follows a three-branch representation design. The operation sequence and the constructed EIS/impedance shape sequence are encoded by two official UniShape backbone wrappers. For multi-channel inputs, each channel is passed through the shared official UniShape encoder and the channel-level representations are aggregated. Denoting the two branch encoders by B_op and B_eis, the sequential embeddings are
z_i^{op}=B_{op}(x_i^{op}),\quad z_i^{eis}=B_{eis}(x_i^{eis}),\quad z_i^{op},z_i^{eis}\in\mathbb{R}^{d}#(2)
where B_op and B_eis are the operation and EIS/impedance UniShape branch encoders; x_i^op and x_i^eis are their corresponding inputs; z_i^op and z_i^eis are the learned branch embeddings; and d is the embedding dimension.
The condition/statistical vector is encoded by an MLP condition encoder phi_cond, which maps the 12-dimensional input into the same representation dimension d:
z_i^{cond}=\phi_{cond}(x_i^{cond})=W_2GELU(LN(W_1x_i^{cond}+b_1))+b_2#(3)
where x_i^cond is the 12-dimensional condition/statistical vector; phi_cond denotes the condition encoder; W_1 and W_2 are linear projection matrices; b_1 and b_2 are bias terms; LN denotes layer normalization; GELU is the Gaussian error linear unit; and z_i^cond is the encoded condition token.
The condition embedding is used both as a feature source in the fusion module and as the driver of condition-dependent prototype displacement in the classification head. This design reflects the assumption that operating conditions can influence not only the feature distribution but also the appropriate class decision geometry.
Residual KAN-Fusion. The three embeddings are concatenated as u_i = [z_i^op; z_i^eis; z_i^cond] in R^{3d}. CAPT-UniShape fuses them using a residual structure composed of a stable MLP branch and a bottleneck KAN residual branch:
h_i=MLP(u_i)+\lambda_{KAN}W_KKAN(Bottleneck(u_i))#(4)
where u_i is the concatenated multi-source embedding; h_i is the fused representation; MLP is the main fusion branch; Bottleneck compresses u_i before the KAN branch; KAN denotes the Kolmogorov-Arnold network residual transformation; W_K is the projection matrix after the KAN branch; and lambda_KAN is the learnable residual scaling coefficient.
The MLP branch forms the main fusion path. The KAN branch first compresses the concatenated representation through a bottleneck, applies a KAN-style nonlinear transformation, and then projects the output back to the fused representation dimension. In the implemented model, lambda_KAN is learnable and is initialized to 0.1. The KAN-style layer expands normalized scalar inputs on learnable Gaussian basis functions and linearly mixes the basis responses. Its regularization penalizes basis-weight scale and center spacing, so the branch is used as a controlled nonlinear residual rather than as an independent classifier.
Condition-aware RBF prototype transport head. The classifier is formulated as a condition-adapted prototype classifier. Let P^0 = {p_k^0}_{k=1}^K denote learnable static class prototypes, where p_k^0 in R^d. For each sample and class, the head computes a dynamic prototype by adding a condition-dependent offset:
p_{i,k}=p_k^0+\Delta p_{i,k}(z_i^{cond})#(5)
where p_k^0 is the static prototype of class k; Delta p_{i,k}(z_i^cond) is the condition-dependent prototype offset generated for sample i and class k; and p_{i,k} is the resulting dynamic prototype.
The offset is produced by an RBF condition mapper. Given RBF centers c_j and widths sigma_j, the j-th response is
r_{i,j}=\exp\left(-\frac{\left\|z_i^{cond}-c_j\right\|_2^2}{2\sigma_j^2}\right)#(6)
where r_{i,j} is the j-th RBF response for sample i; z_i^cond is the condition token; c_j is the j-th RBF center; sigma_j is its width; and ||.||_2 denotes the Euclidean norm.
The RBF response vector is linearly mapped and reshaped into class-wise prototype offsets. The class logit is then computed by temperature-scaled cosine similarity:
logit_{i,k}=\frac{\cos(h_i,p_{i,k})}{\tau}#(7)
where logit_{i,k} is the classification logit for sample i and class k; cos(.) denotes cosine similarity; h_i is the fused representation; p_{i,k} is the dynamic class prototype; and tau is the temperature parameter.
Compared with a purely static prototype head, this formulation allows each class anchor to move in a restricted, condition-dependent manner. The claim made here is deliberately limited: the head is intended to model condition-related decision shifts in the evaluated dataset, and its necessity should be judged by the ablation results rather than assumed a priori.
Training objective. The model is trained with a class-weighted cross-entropy term and three auxiliary regularizers:
L=L_{CE}+\alpha_{transport}L_{transport}+\alpha_{sep}L_{sep}+\alpha_{KAN}L_{KAN}#(8)
where L is the total training loss; L_CE is the class-weighted cross-entropy loss; L_transport is the prototype-transport regularization term; L_sep is the prototype-separation regularization term; L_KAN is the KAN regularization term; and alpha_transport, alpha_sep, and alpha_KAN are the corresponding loss weights.
In the reported configuration, alpha_transport = 0.001, alpha_sep = 0.001, and alpha_KAN = 0.0001. The cross-entropy term uses sqrt-balanced class weighting. The transport penalty constrains the magnitude of condition-driven prototype offsets,
L_{transport}=mean_{i,k}\left\|\Delta p_{i,k}\right\|_2^2#(9)
where mean_{i,k} denotes averaging over samples and classes, and Delta p_{i,k} is the condition-dependent prototype offset.
and the separation term penalizes excessive similarity among static prototypes using a margin m:
L_{sep}=mean_{a\ne b}\max(0,\cos(p_a^0,p_b^0)-m)#(10)
where mean_{a != b} denotes averaging over all unequal class-prototype pairs; p_a^0 and p_b^0 are static prototypes of classes a and b; cos(.) is cosine similarity; and m is the separation margin.
The KAN regularizer L_KAN is the scale and smoothness penalty returned by the KAN branch. During inference, only the logits in Eq. (7) are used.
Dataset construction and split protocol. The evaluated project data are stored in an Excel source file and converted into an NPZ dataset for the official model pipeline. The raw table contains 11,137 rows and 230 columns, including test time, numeric label, 216 single-cell voltage columns, stack total voltage, stack total current, stack power, and nine impedance/EIS statistical variables. The row-level label counts are 1,752 for class 0, 2,036 for class 1, and 7,349 for class 2. After windowing, the evaluated 8:2 dataset contains 433 windows, with 74 windows from class 0, 68 from class 1, and 291 from class 2.
The fixed test protocol uses seed = 44, window_size = 64, stride_train = 16, stride_eval = 32, eis_seq_len = 128, split_mode = segment, segment_block_seconds = 300, ratio = 8:2, and val_size = 0.25. The resulting split contains 311 training windows, 67 validation windows, and 55 test windows. The test set contains 8 Normal windows, 11 class-1 windows, and 36 Flooding/over-wet windows. Validation is used for checkpoint selection, the reported model uses best.ckpt, and train+validation refitting is not used.
Training configuration, baselines, and metrics. Official-CAPT-UniShape-RBF-KANFusion is trained with AdamW, learning rate 0.0001, weight decay 0.0001, batch size 32, a maximum of 80 epochs, early-stopping patience 10, and sqrt-balanced class weighting. The shared representation dimension is d = 128. The RBF head uses 16 RBF centers and an initial temperature of 0.07. The Residual KAN-Fusion bottleneck dimension is 32 and the KAN branch uses 8 basis functions.
The available clean 8:2 comparison results include logistic regression, random forest, MLP, 1D CNN, Transformer, iTransformer, and the proposed model. The ablation variants include no_rbf, no_kan_fusion, static_prototype, no_transport_reg, no_separation_reg, no_eis_input, no_condition_input, stack_only, and eis_cond_only. Accuracy, Macro-F1, and Weighted-F1 are used as the main metrics. Macro-F1 is emphasized because the evaluated test set is imbalanced and the Normal class contains only eight test windows. Class-0 precision, recall, and F1 are also reported to make the normal-state behavior explicit. The raw-variable ranges, clean fixed-protocol model summary, and available baseline results are provided in Tables 1-3 to support reproducibility of the experimental setting; the interpretation of these results is discussed in Section 3.

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

Table 2. Clean-test performance of the proposed model under the evaluated seed-44 fixed 8:2 protocol.
| Model | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | Class-0 P/R/F1 (%) | Test windows | Params |
| --- | --- | --- | --- | --- | --- | --- |
| Official-CAPT-UniShape-RBF-KANFusion | 100.00 | 100.00 | 100.00 | 100.00 / 100.00 / 100.00 | 55 | 6,489,373 |

Table 3. Available aligned 8:2 clean baseline results from project artifacts.
| Model | Category | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | Params |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | Traditional ML | 94.55 | 88.31 | 94.24 | 2,151 |
| Random forest | Traditional ML | 94.55 | 88.31 | 94.24 | 2,436 |
| MLP | Deep learning | 92.73 | 83.76 | 92.07 | 50,243 |
| 1D CNN | Deep learning | 85.45 | 63.33 | 78.91 | 18,947 |
| Transformer | Transformer | 94.55 | 93.29 | 94.86 | 109,763 |
| iTransformer | Transformer | 92.73 | 83.76 | 92.07 | 214,595 |

![Fig. 1. Overall CAPT-UniShape architecture. The model receives x_op, constructed x_eis, and x_cond; extracts operation and EIS/impedance shape embeddings with two official UniShape backbones; encodes conditions with an MLP; fuses the three embeddings by Residual KAN-Fusion; and classifies with a condition-aware RBF prototype transport head.](outputs/paper_figures/architecture_diagram.png)
Fig. 1. Overall CAPT-UniShape architecture. The model receives x_op, constructed x_eis, and x_cond; extracts operation and EIS/impedance shape embeddings with two official UniShape backbones; encodes conditions with an MLP; fuses the three embeddings by Residual KAN-Fusion; and classifies with a condition-aware RBF prototype transport head.

![Fig. 2. Condition-aware RBF prototype transport head. The condition token is mapped through RBF bases to class-wise prototype offsets, which shift static prototypes before cosine-logit classification.](outputs/paper_figures/prototype_head_diagram.png)
Fig. 2. Condition-aware RBF prototype transport head. The condition token is mapped through RBF bases to class-wise prototype offsets, which shift static prototypes before cosine-logit classification.

## 3. Results and discussion

Scope of evaluation. The results in this section should be interpreted within the fixed experimental protocol described in Section 2. The main model is evaluated with seed = 44, an 8:2 segment-based split, validation-based selection of best.ckpt, and no train+validation refitting. The final test set contains 55 windows, including only 8 Normal windows, 11 class-1 windows, and 36 Flooding/over-wet windows. This setting is useful for comparing model variants under a controlled project protocol, but it is not sufficient by itself to support broad claims about cross-seed, cross-dataset, or deployment-level generalization.
Clean-test performance under the fixed protocol. Under the evaluated seed-44 fixed 8:2 test protocol, Official-CAPT-UniShape-RBF-KANFusion achieved 100.00% accuracy, 100.00% Macro-F1, and 100.00% Weighted-F1 on the 55-window test set (Table 2). The class-0 Normal category also reached 100.00% precision, recall, and F1 in this clean-test split. The corresponding confusion matrix in Fig. 3 shows no misclassified test windows among the three encoded classes. These results indicate that the proposed three-input representation and condition-aware classification design can separate the available test windows in this evaluated split.
The 100.00% result should not be read as evidence that the model is error-free in general. The score is obtained from a single seed and a small fixed test set, and the Normal class has limited support. A single misclassification in class 0 would substantially change the class-level recall and F1. Therefore, the clean-test result is best framed as strong split-specific evidence rather than as a general performance guarantee.
Comparison with available clean baselines. The available aligned 8:2 baseline results in Table 3 show that the dataset is informative even for simpler methods. Logistic regression and random forest both achieved 94.55% accuracy and 88.31% Macro-F1, while the Transformer baseline achieved 94.55% accuracy and 93.29% Macro-F1. The proposed model obtains the best clean-test scores among the available artifacts, but the absolute gap should be discussed with caution because all results are from the same fixed seed-44 protocol. The more informative evidence for the proposed design is therefore provided by the ablation behavior and the modality-sensitivity analysis, rather than by the headline accuracy alone.
Effect of the prototype head and Residual KAN-Fusion. The ablation results in Table 4 support a moderate contribution from the condition-aware RBF prototype head and the Residual KAN-Fusion module in the evaluated split. Removing the RBF dynamic prototype head reduced accuracy from 100.00% to 98.18% and Macro-F1 from 100.00% to 96.33%; class-0 recall decreased from 100.00% to 87.50%. Removing Residual KAN-Fusion caused a larger degradation, with 96.36% accuracy, 92.46% Macro-F1, and 75.00% class-0 recall. These results suggest that the dynamic prototype formulation and nonlinear residual fusion are useful for maintaining Normal-class sensitivity under the current clean protocol.
At the same time, the ablation evidence does not justify an overstatement of every subcomponent. The static_prototype, no_transport_reg, and no_separation_reg variants all achieved 100.00% accuracy and 100.00% Macro-F1 in the clean seed-44 split. Thus, the current evidence does not prove that condition-dependent prototype transport, transport regularization, or separation regularization is individually necessary for clean-test success. These components should be presented as part of a plausible condition-aware design whose marginal value requires further validation, especially across additional seeds and external test conditions.
Input-branch sensitivity. The input ablations show that the EIS/impedance and condition/statistical information carries most of the discriminative signal in the current dataset. Removing the EIS/impedance branch reduced accuracy to 87.27%, Macro-F1 to 82.99%, and class-0 F1 to 58.82%. The stack_only variant achieved 83.64% accuracy and 79.95% Macro-F1, indicating that stack voltage, current, and power alone are less sufficient under this split. By contrast, the eis_cond_only variant reached 100.00% accuracy and 100.00% Macro-F1. This pattern should not be interpreted as evidence that operation sequences are generally unnecessary for PEMFC diagnosis. It indicates that, in the evaluated dataset and split, the constructed EIS/impedance shape sequence and condition/statistical vector are highly informative, while the operation branch adds limited visible marginal gain in the clean setting.
Noise robustness and class-level failure mode. The joint-input noise experiments expose a different behavior from the clean protocol. When noise is added simultaneously to x_op, x_eis, and x_cond, the model no longer maintains balanced class performance. At 30 dB SNR, accuracy remained 85.45%, but Macro-F1 decreased to 62.26% and class-0 precision, recall, and F1 were all 0.00%. At 20 dB SNR, accuracy was 87.27%, but Macro-F1 was 68.32%, with class-0 recall of 12.50% and class-0 F1 of 22.22%. Stronger perturbations caused further degradation, including 60.00% accuracy at 5 dB and 25.45% accuracy at 0 dB (Table 5 and Fig. 5).
These results indicate that CAPT-UniShape performs strongly under the clean fixed protocol but remains sensitive to joint input perturbations, particularly for the Normal class. The discrepancy between accuracy and Macro-F1 under noise is important because the test set is imbalanced: acceptable-looking accuracy can coexist with poor recognition of the minority Normal class. The noise experiments therefore should be reported as a limitation and diagnostic failure mode, not as evidence of broad robustness. They also suggest that future work should consider noise-aware training, modality-specific perturbation analysis, calibration, and repeated evaluation across multiple random seeds.
Overall interpretation. Taken together, the experiments support a restrained conclusion. CAPT-UniShape is a useful multi-input architecture for the evaluated PEMFC dataset because it combines operation sequences, constructed EIS/impedance shape sequences, and condition/statistical information, and because its full configuration performs best in the clean seed-44 split. The ablations further indicate that EIS/impedance information is essential in the present dataset and that RBF prototypes and Residual KAN-Fusion help preserve Normal-class recall relative to their removal. However, the evidence is limited by the single seed, the 55-window test set, the small Normal-class support, and the observed noise sensitivity. Multi-seed testing, larger external test sets, and statistical comparison are needed before making stronger claims about general diagnostic robustness.

Table 4. Ablation study under seed-44 fixed 8:2 protocol.
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

Table 5. Proposed-model performance under joint-input test noise.
| SNR | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | Class-0 precision (%) | Class-0 recall (%) | Class-0 F1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Clean | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 30 dB | 85.45 | 62.26 | 78.79 | 0.00 | 0.00 | 0.00 |
| 20 dB | 87.27 | 68.32 | 82.84 | 100.00 | 12.50 | 22.22 |
| 10 dB | 83.64 | 73.14 | 83.26 | 42.86 | 37.50 | 40.00 |
| 5 dB | 60.00 | 52.38 | 62.86 | 15.38 | 25.00 | 19.05 |
| 0 dB | 25.45 | 22.74 | 14.82 | 22.22 | 25.00 | 23.53 |

![Fig. 3. Clean-test confusion matrix for the proposed model under the evaluated seed-44 fixed 8:2 protocol. Class 2 is reported as Flooding / over-wet.](outputs/paper_figures/confusion_matrix_proposed_seed44_8_2.png)
Fig. 3. Clean-test confusion matrix for the proposed model under the evaluated seed-44 fixed 8:2 protocol. Class 2 is reported as Flooding / over-wet.

![Fig. 4. Ablation comparison of Accuracy and Macro-F1 under the evaluated seed-44 fixed 8:2 protocol.](outputs/paper_figures/ablation_bar_chart.png)
Fig. 4. Ablation comparison of Accuracy and Macro-F1 under the evaluated seed-44 fixed 8:2 protocol.

![Fig. 5. Noise robustness curves for clean and SNR-based joint-input perturbation tests.](outputs/paper_figures/noise_robustness_proposed_snr.png)
Fig. 5. Noise robustness curves for clean and SNR-based joint-input perturbation tests.

## 4. Conclusions

This study presented CAPT-UniShape, a condition-aware prototype transport framework for PEMFC fault classification. The work addresses a practical diagnostic setting in which stack operation sequences, constructed EIS/impedance shape sequences, and condition/statistical variables provide complementary but condition-dependent information. CAPT-UniShape combines two official UniShape branch encoders, an MLP condition encoder, Residual KAN-Fusion, and a condition-aware RBF prototype transport head to integrate these inputs and adapt the prototype-based decision geometry to the encoded operating condition.
Under the evaluated seed-44 fixed 8:2 test protocol, Official-CAPT-UniShape-RBF-KANFusion achieved 100.00% accuracy, 100.00% Macro-F1, and 100.00% Weighted-F1 on 55 test windows. This result shows that the proposed configuration can separate the three encoded classes in the clean fixed split. The ablation results further indicate that the constructed EIS/impedance branch is the most important input source in the current dataset, and that removing either the RBF dynamic prototype head or Residual KAN-Fusion reduces Normal-class recall. These findings support the usefulness of multi-input representation learning and condition-aware classification in the evaluated setting.
The conclusions should remain limited to the available evidence. The main result is based on a single seed, a fixed 8:2 segment-based split, and a small test set with only eight Normal windows. In addition, static_prototype, no_transport_reg, and no_separation_reg also reached 100.00% accuracy and 100.00% Macro-F1 in the clean split, so the marginal contribution of these prototype-related regularization components cannot be claimed as decisive from the present experiments. The joint-input noise tests show a clear degradation of Macro-F1 and a pronounced sensitivity of the Normal class, indicating that the model should not be described as broadly robust to strong noise.
Future work should therefore prioritize multi-seed evaluation, larger and external test sets, cross-condition validation, statistical significance testing, and noise-aware training or modality-specific denoising. If raw full-frequency EIS spectra become available, the current constructed EIS/impedance statistical shape input should also be compared with spectrum-based representations. Overall, the present results support CAPT-UniShape as a carefully scoped multi-source diagnostic model for the evaluated PEMFC dataset, rather than as a general-purpose PEMFC fault diagnosis solution.

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
