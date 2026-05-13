# CAPT-UniShape 实验进度清单（截至 2026-05-11）

> 说明：本清单依据当前仓库里**已经落盘的结果文件/图表/汇总表**整理；有文件就记为已完成，没有对应产物就记为待完成或待确认。

## 一、已完成

### 1. 数据准备与数据诊断
- [x] 原始实验数据已整理：`D:/learn/论文所需材料/论文2/CAPT-UniShape/data/raw/测试数据.xlsx`
- [x] 数据集诊断与统计汇总已完成：`D:/learn/论文所需材料/论文2/CAPT-UniShape/outputs/ac_voltage_dataset_diagnosis/`
- [x] 已生成多套实验所需 `NPZ/processed` 数据：`D:/learn/论文所需材料/论文2/CAPT-UniShape/data/processed/`

### 2. 主实验与比例实验
- [x] 8:2 主结果显式测试复核已完成：`D:/learn/论文所需材料/论文2/CAPT-UniShape/results/codex_eval_proposed_macro98_seed44_8_2_bestckpt/metrics.json`
- [x] 8:2 基线对比结果已整理：`D:/learn/论文所需材料/论文2/CAPT-UniShape/results/codex_baseline_selected_self_seed44_8_2/test_summary.csv`
- [x] 7:3 / 6:4 / 5:5 独立比例重训练结果已完成：`D:/learn/论文所需材料/论文2/CAPT-UniShape/results/codex_ratio_comparison_independent_seed44_retrain_7_3_6_4_5_5/test_summary.csv`
- [x] 单 seed 五折模型对比已完成：`D:/learn/论文所需材料/论文2/CAPT-UniShape/results/codex_single_seed44_5fold_model_comparison/single_seed_fold_summary.csv`
- [x] 多 seed 稳定性结果已保留：
  - `D:/learn/论文所需材料/论文2/CAPT-UniShape/results/official_baseline_multiseed_proposed_eval32/ranked_multiseed_test_summary.csv`
  - `D:/learn/论文所需材料/论文2/CAPT-UniShape/results/official_baseline_multiseed_ml_eval32/ranked_multiseed_test_summary.csv`

### 3. 消融与噪声实验
- [x] 8:2 消融实验已完成：`D:/learn/论文所需材料/论文2/CAPT-UniShape/results/codex_ablation_improved_proposed_seed44_8_2/summary.csv`
- [x] 5:5 消融实验已完成：`D:/learn/论文所需材料/论文2/CAPT-UniShape/results/codex_ablation_improved_proposed_seed44_5_5/summary.csv`
- [x] SNR 口径噪声鲁棒性基线对比已完成：`D:/learn/论文所需材料/论文2/CAPT-UniShape/results/codex_snr_noise_baselines_proposed_seed44_8_2/summary.csv`

### 4. 结果图与结果数据整合
- [x] 已生成模型结构图：`D:/learn/论文所需材料/论文2/CAPT-UniShape/outputs/paper_figures/architecture_diagram.png`
- [x] 已生成混淆矩阵图：`D:/learn/论文所需材料/论文2/CAPT-UniShape/outputs/paper_figures/confusion_matrix_proposed_seed44_8_2.svg`
- [x] 已生成消融结果图：`D:/learn/论文所需材料/论文2/CAPT-UniShape/outputs/paper_figures/ablation_bar_chart.png`
- [x] 已生成 SNR 噪声鲁棒性图：`D:/learn/论文所需材料/论文2/CAPT-UniShape/outputs/paper_figures/noise_robustness_proposed_snr.svg`
- [x] 已整理实验总表 Excel：
  - `D:/learn/论文所需材料/论文2/CAPT-UniShape/outputs/results_summary/CAPT-UniShape_实验结果总表.xlsx`
  - `D:/learn/论文所需材料/论文2/CAPT-UniShape/outputs/results_summary/CAPT-UniShape_实验结果总表_已调整.xlsx`
  - `D:/learn/论文所需材料/论文2/CAPT-UniShape/outputs/results_summary/CAPT-UniShape_实验结果总表_不同测试比例重训练.xlsx`
- [x] 已整理结果说明文档：
  - `D:/learn/论文所需材料/论文2/CAPT-UniShape/outputs/results_summary/自测数据集基线对比结果.md`
  - `D:/learn/论文所需材料/论文2/CAPT-UniShape/outputs/results_summary/自测数据集不同比例重训练结果.md`
  - `D:/learn/论文所需材料/论文2/CAPT-UniShape/docs/CAPT-UniShape_论文实验资料整理.md`

## 二、还需要做

### 1. 结果图补齐与统一
- [ ] 补齐论文最终版结果图：当前 `outputs/paper_figures/` 只有 5 个图文件，**还没有看到**四比例主结果对比图/稳定性总图等最终论文常用总览图。
- [ ] 补齐图的导出格式：当前实际存在的是 `png` 与 `svg` 的部分文件，**未看到**文档里提到的完整 `png/pdf/svg` 三格式导出结果。
- [ ] 若论文要单独展示原型头机制图，需补生成 `prototype_head_diagram` 对应图片文件（当前目录中未看到该文件）。

### 2. 结果目录与实验口径收口
- [ ] 若要完全按 `official_baseline_comparison` 这一路径作为最终主结果目录，需要补齐该目录下缺失内容；当前**未看到**：
  - `D:/learn/论文所需材料/论文2/CAPT-UniShape/results/official_baseline_comparison/summary.csv`
  - `D:/learn/论文所需材料/论文2/CAPT-UniShape/results/official_baseline_comparison/5_5/`
- [ ] 若论文主表要求四个比例都使用**同一轮独立重训练协议**，还需补跑或单独整理 8:2 独立比例重训练结果；当前 7:3/6:4/5:5 是本轮重训练，8:2 主要沿用既有参考结果。
- [ ] 若还需要 `noise_std` 口径的非 SNR 噪声实验结果，需补齐或确认目录；当前仓库里**未看到** `D:/learn/论文所需材料/论文2/CAPT-UniShape/results/codex_noise_improved_proposed_seed44_8_2/`。

### 3. 结果整合的最终收尾
- [ ] 统一“最终引用版”总表：目前总表至少有 3 份（原版、已调整版、不同比例重训练版），需要确定论文正文最终引用哪一份。
- [ ] 统一类别名称：第 1 类/第 2 类物理含义在已有材料中仍有冲突，图表、总表、配置文件需要统一后再定稿。
- [ ] 若论文要正式报告推理时间，建议在同一脚本/同一环境下重新 benchmark 一次，避免不同结果文件里的 ms/sample 不可直接横向比较。

## 三、建议下一步顺序
1. 先确定论文最终采用哪一套主结果口径（8:2 参考结果 + 7:3/6:4/5:5 重训练，还是全部按同一协议重跑/重整）。
2. 统一总表版本，只保留一份最终引用表。
3. 补齐最终论文图（尤其是四比例主结果总览图、稳定性图、缺失格式导出图）。
4. 统一类别名称与图表标注。
5. 最后再把图、表、文字说明同步进论文正文和答辩/PPT材料。

## 四、当前判断
- **实验主体已经基本做完**：主结果、比例实验、五折、多 seed、消融、SNR 噪声、结果汇总表都已经有落盘产物。
- **目前主要剩下的是“收口工作”**：补图、统一口径、统一总表、统一类别名称、补缺失目录/格式，而不是从零开始做主实验。
