# CAPT-UniShape 真实实验运行指南

下面命令均可直接复制到 **Windows PowerShell** 中运行。不要使用 CMD 的换行符 `^`。

先进入项目目录：

```powershell
cd "D:\learn\论文所需材料\论文2\CAPT-UniShape"
```

---

## 1. 真实实验设计

当前论文实验只保留三条主线：

1. **四组训练集/测试集比例的完整基准对比实验**
   - 比例：8:2、7:3、6:4、5:5。
   - 本文模型：`proposed`。
   - 传统机器学习基准：`logreg`、`svm`、`random_forest`。
   - 深度学习基准：`mlp`、`cnn1d`、`lstm`。
   - Transformer 类基准：`transformer`、`itransformer`。

2. **消融实验**
   - 验证 RBF 动态原型、Residual KAN-Fusion、EIS 输入、工况输入、正则项等模块贡献。
   - `no_rbf` 只属于消融/内部对照，不再作为完整对比实验的全部内容。

3. **一组比例下的噪声鲁棒性实验**
   - 默认使用 8:2。
   - 干净训练，只在测试集加入不同强度高斯噪声。
   - 噪声脚本中的 `rbf` 就是本文主模型，`no_rbf` 是内部对照。

当前训练使用 **AdamW 优化器 + 早停机制**。没有使用遗传算法、粒子群、贝叶斯优化等额外超参数优化算法。

---

## 2. 数据来源与输入特征

原始数据文件：

```text
data/processed/测试数据.xlsx
```

模型使用三类输入：

```text
x_op   = 电堆总电压 / 电堆总电流 / 电堆功率
x_eis  = 阻抗 / EIS 统计构造的序列
x_cond = 9 个阻抗 / EIS 统计 + 3 个电堆变量
labels = 故障类别
split  = 训练 / 验证 / 测试划分
```

真实实验默认使用 `segment` 分段划分，避免同一连续工况片段被错误拼接或跨类别边界滑窗。四比例实验和噪声实验默认把长连续片段切成 240 秒块，以便 5:5 等比例也能稳定分层。

之前尝试过 **训练集类别感知滑窗步长**，即少数类更密、多数类更稀；但实测会明显改变训练分布，尤其会减少多数类训练窗口覆盖，可能导致测试效果下降。因此当前主实验默认 **不启用** 类别感知滑窗补偿，只保留更大的验证集、更密的验证/测试窗口和多 seed 建议。类别感知滑窗只建议作为额外对照实验，不作为论文主结果默认设置。

为了缓解训练比例降低后“验证集好、测试集差”的问题，当前默认采用 **验证集选 epoch，但不默认 train+val 重训**：这样验证集和测试集保持独立，更接近第一次较稳定的实验协议。若想把验证集并回训练集做额外对照，可在训练脚本或实验脚本中手动追加 --refit-trainval，但不建议作为论文主结果默认设置。论文主表应优先比较测试集结果。

---

## 3. 主实验一：四比例完整基准对比

直接运行：

```powershell
python scripts/run_official_baseline_experiments.py --ratios 8_2 7_3 6_4 5_5 --models proposed logreg svm random_forest mlp cnn1d lstm transformer itransformer --epochs 80 --patience 10 --val-size 0.25 --stride-eval 64 --segment-block-seconds 240 --output-root results/official_baseline_comparison --data-root data/processed/official_baseline_comparison
```

该命令会自动生成四组比例数据，并训练全部基准模型。

输出目录示例：

```text
results/official_baseline_comparison/8_2/proposed/
results/official_baseline_comparison/8_2/logreg/
results/official_baseline_comparison/8_2/svm/
results/official_baseline_comparison/8_2/random_forest/
results/official_baseline_comparison/summary.csv
```

论文主要使用：

```text
results/official_baseline_comparison/summary.csv
results/official_baseline_comparison/test_summary.csv
```

`summary.csv` 汇总验证集和测试集指标，`test_summary.csv` 只保留最重要的测试集指标。两个表都会包含每个对比模型的测试效果，其中测试集汇总表字段为：

```text
ratio, model, category, val_accuracy, val_macro_f1,
test_accuracy, test_macro_f1, test_inference_ms,
parameter_count, metrics_path
```

Kaggle 终端中每个模型训练结束后也会打印类似下面的测试集效果：

```text
测试集效果 | ratio=8:2 | model=proposed | test_acc=... | test_macro_f1=...
```

如果中途只想补跑某几个模型，可以改 `--models`，例如：

```powershell
python scripts/run_official_baseline_experiments.py --ratios 8_2 7_3 6_4 5_5 --models transformer itransformer --epochs 80 --patience 10 --val-size 0.25 --stride-eval 64 --segment-block-seconds 240 --output-root results/official_baseline_comparison --data-root data/processed/official_baseline_comparison
```

注意：新版脚本会重写 `summary.csv`，避免绘图误读历史旧行。若要从零重新跑并清空旧模型目录，请先删除旧的 `results/official_baseline_comparison/` 和 `data/processed/official_baseline_comparison/`，或者换一个新的 `--output-root` / `--data-root`。

---

## 4. 主实验二：消融实验

先生成消融实验使用的 64 点窗口数据：

```powershell
python scripts/build_official_npz_from_self_excel.py --excel "data/processed/测试数据.xlsx" --output "data/processed/official_self_stack_impedance_eis_w64_stable.npz" --window-size 64 --stride-train 16 --stride-eval 64 --eis-seq-len 128 --split-mode segment --segment-block-seconds 240 --op-source stack --test-size 0.2 --val-size 0.25
```

然后运行完整消融：

```powershell
python scripts/run_official_ablation_experiments.py --data data/processed/official_self_stack_impedance_eis_w64_stable.npz --variants full_rbf no_rbf no_kan_fusion static_prototype no_transport_reg no_separation_reg no_eis_input no_condition_input stack_only eis_cond_only --epochs 80 --patience 10 --output-root results/official_ablation --data-root data/processed/official_ablation
```

输出目录示例：

```text
results/official_ablation/full_rbf/
results/official_ablation/no_rbf/
results/official_ablation/no_kan_fusion/
results/official_ablation/summary.csv
```

消融项含义：

- `full_rbf`：完整模型，官方 UniShape + EIS + 工况 + Residual KAN-Fusion + 动态 RBF 原型。
- `no_rbf`：去掉 RBF 动态原型头，使用普通分类头。
- `no_kan_fusion`：关闭 Residual KAN 分支，只保留普通融合 MLP。
- `static_prototype`：保留 RBF，但关闭工况感知 prototype transport。
- `no_transport_reg`：去掉原型迁移幅值正则。
- `no_separation_reg`：去掉原型分离正则。
- `no_eis_input`：置零 EIS/阻抗分支。
- `no_condition_input`：置零工况向量。
- `stack_only`：只保留电堆运行分支。
- `eis_cond_only`：去掉电堆运行分支，只保留 EIS/阻抗 + 工况。

论文主要使用：

```text
results/official_ablation/summary.csv
```

---

## 5. 主实验三：一组比例噪声鲁棒性实验

默认跑 8:2，一次训练干净模型，再对测试集加入多种噪声强度：

```powershell
python scripts/run_official_noise_experiments.py --ratio 8_2 --models rbf no_rbf --noise-stds 0.0 0.01 0.03 0.05 0.10 --epochs 80 --patience 10 --val-size 0.25 --stride-eval 64 --segment-block-seconds 240 --output-root results/official_noise_experiments --data-root data/processed/official_noise_experiments
```

说明：

- `--ratio 8_2` 表示噪声实验只使用 8:2；如果要换成 7:3，就改成 `--ratio 7_3`。
- `--noise-stds 0.0 0.01 0.03 0.05 0.10` 表示相对高斯噪声标准差。
- 默认对 `x_op`、`x_eis`、`x_cond` 都加噪声。
- 如果只想对电堆和 EIS 加噪声，可追加：`--noise-targets x_op x_eis`。
- 噪声只加到测试集，训练集和验证集保持干净。

输出目录示例：

```text
results/official_noise_experiments/8_2/rbf/clean_train/
results/official_noise_experiments/8_2/rbf/noise_0p050/
results/official_noise_experiments/8_2/no_rbf/clean_train/
results/official_noise_experiments/summary.csv
```

论文主要使用：

```text
results/official_noise_experiments/summary.csv
```

---

## 6. 绘制论文图

跑完三类实验后，一次性生成三张汇总图：

```powershell
python scripts/plot_official_results.py --baseline-summary results/official_baseline_comparison/summary.csv --ablation-summary results/official_ablation/summary.csv --noise-summary results/official_noise_experiments/summary.csv --output-dir figures/official_summaries
```

会生成：

```text
figures/official_summaries/official_baseline_comparison_summary.png
figures/official_summaries/official_ablation_summary.png
figures/official_summaries/official_noise_summary.png
```

也可以分开画：

```powershell
python scripts/plot_official_results.py --baseline-summary results/official_baseline_comparison/summary.csv --output-dir figures/official_baseline_comparison
```

```powershell
python scripts/plot_official_results.py --ablation-summary results/official_ablation/summary.csv --output-dir figures/official_ablation
```

```powershell
python scripts/plot_official_results.py --noise-summary results/official_noise_experiments/summary.csv --output-dir figures/official_noise_experiments
```

---

## 7. 推荐完整运行顺序

从头开始跑真实论文实验，按下面顺序执行：

```powershell
cd "D:\learn\论文所需材料\论文2\CAPT-UniShape"
```

```powershell
python scripts/run_official_baseline_experiments.py --ratios 8_2 7_3 6_4 5_5 --models proposed logreg svm random_forest mlp cnn1d lstm transformer itransformer --epochs 80 --patience 10 --val-size 0.25 --stride-eval 64 --segment-block-seconds 240 --output-root results/official_baseline_comparison --data-root data/processed/official_baseline_comparison
```

```powershell
python scripts/build_official_npz_from_self_excel.py --excel "data/processed/测试数据.xlsx" --output "data/processed/official_self_stack_impedance_eis_w64_stable.npz" --window-size 64 --stride-train 16 --stride-eval 64 --eis-seq-len 128 --split-mode segment --segment-block-seconds 240 --op-source stack --test-size 0.2 --val-size 0.25
```

```powershell
python scripts/run_official_ablation_experiments.py --data data/processed/official_self_stack_impedance_eis_w64_stable.npz --variants full_rbf no_rbf no_kan_fusion static_prototype no_transport_reg no_separation_reg no_eis_input no_condition_input stack_only eis_cond_only --epochs 80 --patience 10 --output-root results/official_ablation --data-root data/processed/official_ablation
```

```powershell
python scripts/run_official_noise_experiments.py --ratio 8_2 --models rbf no_rbf --noise-stds 0.0 0.01 0.03 0.05 0.10 --epochs 80 --patience 10 --val-size 0.25 --stride-eval 64 --segment-block-seconds 240 --output-root results/official_noise_experiments --data-root data/processed/official_noise_experiments
```

```powershell
python scripts/plot_official_results.py --baseline-summary results/official_baseline_comparison/summary.csv --ablation-summary results/official_ablation/summary.csv --noise-summary results/official_noise_experiments/summary.csv --output-dir figures/official_summaries
```

---

## 8. 如何判断训练成功

训练时终端会显示类似：

```text
开始训练 Official CAPT-UniShape
训练/验证/测试样本数: ... / ... / ...
Epoch 001/080 | train_loss=... | val_acc=... | val_macro_f1=...
测试集结果: acc=... macro_f1=...
训练完成。已保存:
```

看到 `训练完成。已保存:`，并且对应目录下生成 `metrics.json`、`summary.csv`、`confusion_matrix.csv`、`predictions.csv`，说明该模型训练完成。

---

## 9. 论文最终需要看的文件

表格结果：

```text
results/official_baseline_comparison/summary.csv
results/official_ablation/summary.csv
results/official_noise_experiments/summary.csv
```

图像结果：

```text
figures/official_summaries/official_baseline_comparison_summary.png
figures/official_summaries/official_ablation_summary.png
figures/official_summaries/official_noise_summary.png
```

论文建议汇报：

- Accuracy
- Macro-F1
- 混淆矩阵
- 单样本推理时间
- 参数量
- 本文模型与传统机器学习、深度学习、Transformer、iTransformer 的对比
- 8:2、7:3、6:4、5:5 四组比例下的稳定性
- 消融实验中各模块对性能的贡献
- 单一比例下不同噪声强度的鲁棒性曲线

---

## 10. 官方 UniShape 预训练权重说明

当前指导文件中的命令默认 **不加载官方预训练权重**，因为项目里还没有明确指定哪一个 checkpoint 是最终官方预训练权重。

如果后续确认了权重文件路径，再在单独训练主模型时追加：

```text
--op-pretrained 权重路径 --eis-pretrained 权重路径
```

注意：如果用于公平实验，所有需要加载 UniShape 权重的对照项必须加载同一份权重。

---

## 11. Kaggle 运行说明

Kaggle 不能像本地文件夹一样直接长期“替换文件”。推荐做法是：

1. 把本项目压缩成 zip。
2. 在 Kaggle 创建或更新一个 Dataset，把 zip 作为新版本上传。
3. 在 Notebook 中 Add Data 选择该 Dataset。
4. 每次代码有更新，都上传新的 Dataset version，Notebook 里重新选择或刷新版本。

Notebook 中解压示例：

```python
import zipfile
from pathlib import Path

zip_path = Path('/kaggle/input/capt-unishape-kaggle/CAPT-UniShape_kaggle_package.zip')
work_dir = Path('/kaggle/working/CAPT-UniShape')
work_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(work_dir)
```

然后在 Notebook 中运行：

```python
%cd /kaggle/working/CAPT-UniShape
```

如果缺少依赖，先执行：

```python
!pip install -q openpyxl pyyaml scikit-learn
```

之后按第 7 节的命令运行即可。Kaggle 的命令前面通常加 `!`，例如：

```python
!python scripts/run_official_baseline_experiments.py --ratios 8_2 7_3 6_4 5_5 --models proposed logreg svm random_forest mlp cnn1d lstm transformer itransformer --epochs 80 --patience 10 --val-size 0.25 --stride-eval 64 --segment-block-seconds 240 --output-root results/official_baseline_comparison --data-root data/processed/official_baseline_comparison
```
