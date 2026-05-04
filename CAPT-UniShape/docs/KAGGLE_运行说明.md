# CAPT-UniShape：Kaggle 运行说明

本说明用于把本地项目上传到 Kaggle，并在 Kaggle Notebook 中运行真实论文实验。

---

## 1. 本地需要上传的文件

本项目不长期保留生成好的 zip 包，避免把可再生成的压缩产物放在项目根目录。需要上传 Kaggle 时，在项目根目录重新生成：

```powershell
Compress-Archive -Path configs,data,external,models,scripts,src,docs,train.py,evaluate.py,README.md -DestinationPath CAPT-UniShape_kaggle_package.zip -Force
```

生成后的文件是：

```text
CAPT-UniShape_kaggle_package.zip
```

生成位置：

```text
D:\learn\论文所需材料\论文2\CAPT-UniShape\CAPT-UniShape_kaggle_package.zip
```

这个 zip 包应包含：

```text
configs/
data/raw/测试数据.xlsx
external/unishape/
models/
scripts/
src/
train.py
evaluate.py
README.md
docs/README_训练测试绘图.md
docs/KAGGLE_运行说明.md
```

---

## 2. 在 Kaggle 上传项目

Kaggle 不能像本地一样直接永久替换某个文件夹。推荐使用 **Kaggle Dataset** 管理代码包。

操作步骤：

1. 打开 Kaggle。
2. 点击右上角 **Create**。
3. 选择 **New Dataset**。
4. 上传 `CAPT-UniShape_kaggle_package.zip`。
5. Dataset 名字建议填：

```text
capt-unishape-kaggle
```

6. 创建 Dataset。

如果以后代码修改了，不要在 Notebook 里手动改很多文件，推荐重新生成 zip，然后在这个 Dataset 页面点击 **New Version** 上传新版本。

### 2.1 不想每次重新新建 Cell 的推荐做法

Kaggle Notebook 只需要创建一次。以后代码更新时，不要新建 Notebook，也不要重新写 Cell，按下面流程即可：

1. 本地重新生成 `CAPT-UniShape_kaggle_package.zip`。
2. 打开原来的 Kaggle Dataset 页面。
3. 点击 **New Version**，上传新的 zip。
4. 回到原来的 Notebook，右侧 Data 面板刷新/切换到 Dataset 最新版本。
5. 点击 Notebook 顶部 **Restart Session**，然后 **Run All**。

也就是说：**Notebook 的 Cell 固定不变，只更新 Dataset 版本**。如果路径或 Dataset 名字变化，建议使用下面第 4 节的“自动查找 zip”启动 Cell，避免硬编码路径。

---

## 3. 创建 Kaggle Notebook

1. 新建一个 Kaggle Notebook。
2. 在右侧点击 **Add Data**。
3. 选择刚才上传的 Dataset，例如：

```text
capt-unishape-kaggle
```

---

## 4. 解压项目

在 Kaggle Notebook 第一个代码单元运行下面这段即可。以后每次 Dataset 更新后，仍然复用这个 Cell，不需要重建：

```python
from pathlib import Path
import os
import shutil
import zipfile

work_dir = Path('/kaggle/working/CAPT-UniShape')
if work_dir.exists():
    shutil.rmtree(work_dir)

zip_candidates = list(Path('/kaggle/input').glob('**/CAPT-UniShape_kaggle_package.zip'))
if zip_candidates:
    with zipfile.ZipFile(zip_candidates[0], 'r') as zf:
        zf.extractall(work_dir)
else:
    project_candidates = [
        p for p in Path('/kaggle/input').glob('**/train.py')
        if (p.parent / 'scripts' / 'run_official_baseline_experiments.py').exists()
    ]
    if not project_candidates:
        raise FileNotFoundError('没有找到 CAPT-UniShape zip 或项目目录，请确认右侧已经 Add Data')
    shutil.copytree(project_candidates[0].parent, work_dir)

os.chdir(work_dir)
print('当前目录:', os.getcwd())
print('baseline脚本存在:', Path('scripts/run_official_baseline_experiments.py').exists())
```

然后进入项目目录：

```python
%cd /kaggle/working/CAPT-UniShape
```

---

## 5. 安装依赖

Kaggle 通常已经自带 `torch`、`numpy`、`pandas`、`matplotlib`、`scikit-learn`，但为了保险，运行：

```python
!pip install -q openpyxl pyyaml scikit-learn
```

检查 GPU：

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
```

如果输出 `True`，说明可以使用 GPU。

---

## 6. 推荐运行顺序

下面命令是在 Kaggle Notebook 里运行的，所以命令前面要加 `!`。

注意：下面所有默认命令都 **不启用** `--class-aware-train-stride`。该选项会改变训练窗口分布，前期实测可能让多数类覆盖不足、测试效果变差，因此只建议作为额外对照实验，不作为论文主结果默认设置。当前默认只用验证集选择 epoch，不默认 train+val 重训，以保持验证集和测试集独立；若要做额外对照，可手动追加 --refit-trainval。

### 6.1 四比例完整基准对比实验

这个实验会跑：

- proposed
- logreg
- svm
- random_forest
- mlp
- cnn1d
- lstm
- transformer
- itransformer

运行：

```python
from pathlib import Path
import shutil, os

# 源项目目录（只读）
SRC = Path("/kaggle/input/datasets/weiming1zeng/xiaolunwen2-4-29")

# 目标工作目录（可写）
DST = Path("/kaggle/working/CAPT-UniShape")

# 如果之前跑过，先删掉，保证是干净的
if DST.exists():
    shutil.rmtree(DST)

# 复制整个项目到 /kaggle/working
shutil.copytree(SRC, DST)

# 切换到工作目录
os.chdir(DST)
print("当前目录:", os.getcwd())
print("是否存在 scripts/run_official_baseline_experiments.py:",
      (DST / "scripts" / "run_official_baseline_experiments.py").exists())

# 运行官方 baseline 脚本
!python scripts/run_official_baseline_experiments.py \
  --ratios 8_2 7_3 6_4 5_5 \
  --models proposed logreg svm random_forest mlp cnn1d lstm transformer itransformer \
  --epochs 80 \
  --patience 10 \
  --val-size 0.25 \
  --stride-eval 64 \
  --segment-block-seconds 240 \
  --output-root results/official_baseline_comparison \
  --data-root data/processed/official_baseline_comparison
```

主要结果：

```text
results/official_baseline_comparison/summary.csv
results/official_baseline_comparison/test_summary.csv
```

其中 `test_summary.csv` 是最直接的对比模型测试集结果表，包含每个模型的 `test_accuracy` 和 `test_macro_f1`。Notebook 终端里也会逐个打印 `测试集效果 | model=... | test_acc=... | test_macro_f1=...`。

---

### 6.2 生成消融实验数据

运行：

```python
!python scripts/build_official_npz_from_self_excel.py --excel "data/raw/测试数据.xlsx" --output "data/processed/official_self_stack_impedance_eis_w64_stable.npz" --window-size 64 --stride-train 16 --stride-eval 64 --eis-seq-len 128 --split-mode segment --segment-block-seconds 240 --op-source stack --test-size 0.2 --val-size 0.25
```

主要输出：

```text
data/processed/official_self_stack_impedance_eis_w64_stable.npz
data/processed/official_self_stack_impedance_eis_w64_stable.summary.json
```

---

### 6.3 消融实验

运行：

```python
!python scripts/run_official_ablation_experiments.py --data data/processed/official_self_stack_impedance_eis_w64_stable.npz --variants full_rbf no_rbf no_kan_fusion static_prototype no_transport_reg no_separation_reg no_eis_input no_condition_input stack_only eis_cond_only --epochs 80 --patience 10 --output-root results/official_ablation --data-root data/processed/official_ablation
```

主要结果：

```text
results/official_ablation/summary.csv
```

---

### 6.4 噪声鲁棒性实验

默认跑 8:2，一次训练干净模型，再只给测试集加噪声：

```python
!python scripts/run_official_noise_experiments.py --ratio 8_2 --models rbf no_rbf --noise-stds 0.0 0.01 0.03 0.05 0.10 --epochs 80 --patience 10 --val-size 0.25 --stride-eval 64 --segment-block-seconds 240 --output-root results/official_noise_experiments --data-root data/processed/official_noise_experiments
```

主要结果：

```text
results/official_noise_experiments/summary.csv
```

---

## 7. 绘制结果图

三类实验都跑完后运行：

```python
!python scripts/plot_official_results.py --baseline-summary results/official_baseline_comparison/summary.csv --ablation-summary results/official_ablation/summary.csv --noise-summary results/official_noise_experiments/summary.csv --output-dir figures/official_summaries
```

会生成：

```text
figures/official_summaries/official_baseline_comparison_summary.png
figures/official_summaries/official_ablation_summary.png
figures/official_summaries/official_noise_summary.png
```

---

## 8. 下载结果

在 Kaggle Notebook 中，可以把结果打包下载：

```python
!zip -r CAPT-UniShape_results.zip results figures data/processed/*.summary.json
```

打包后文件在：

```text
/kaggle/working/CAPT-UniShape/CAPT-UniShape_results.zip
```

在右侧 Output 区域可以下载。

---

## 9. 如果 Kaggle 运行时间不够

可以分批跑。

例如先跑传统机器学习：

```python
!python scripts/run_official_baseline_experiments.py --ratios 8_2 7_3 6_4 5_5 --models logreg svm random_forest --val-size 0.25 --stride-eval 64 --segment-block-seconds 240 --output-root results/official_baseline_comparison --data-root data/processed/official_baseline_comparison
```

再跑深度学习和 Transformer：

```python
!python scripts/run_official_baseline_experiments.py --ratios 8_2 7_3 6_4 5_5 --models proposed mlp cnn1d lstm transformer itransformer --epochs 80 --patience 10 --val-size 0.25 --stride-eval 64 --segment-block-seconds 240 --output-root results/official_baseline_comparison --data-root data/processed/official_baseline_comparison
```

注意：新版脚本会重写 `results/official_baseline_comparison/summary.csv`，不再追加旧行。要完全重跑并清空旧模型目录时，可以先删除旧结果：

```python
!rm -rf results/official_baseline_comparison data/processed/official_baseline_comparison
```

---

## 10. 常见问题

### 10.1 找不到 zip 文件

确认 Notebook 右侧已经 Add Data，并且 Dataset 中包含：

```text
CAPT-UniShape_kaggle_package.zip
```

### 10.2 找不到 Excel 文件

确认解压后存在：

```text
data/raw/测试数据.xlsx
```

可以运行：

```python
!ls data/processed
```

### 10.3 想更新代码

不要在 Kaggle Notebook 里大量手动改文件。推荐：

1. 本地重新生成 `CAPT-UniShape_kaggle_package.zip`。
2. 到 Kaggle Dataset 页面点击 **New Version**。
3. 上传新 zip。
4. Notebook 中刷新 Dataset 版本后重新解压运行。
