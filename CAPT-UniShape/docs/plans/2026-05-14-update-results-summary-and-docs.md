# 总表与结果文档更新 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将最新的对比测试结果和噪声测试结果同步到总表，并更新 `docs/模型与实验.md` 与 `docs/论文结论与结果汇总.md`。

**Architecture:** 先确认当前实际在用的总表文件和最新结果来源，再把 Excel 中对应 sheet 的表格内容按真实结果覆盖更新，最后同步修正文档中的结果表述、数据来源和结论口径，确保三处信息一致。

**Tech Stack:** Markdown、Excel（openpyxl）、PowerShell、CSV 汇总文件

---

### Task 1: 确认输入与目标文件

**Files:**
- Modify: `outputs/results_summary/CAPT-UniShape_实验结果总表_含自测公开训练结果汇总.xlsx`
- Modify: `docs/模型与实验.md`
- Modify: `docs/论文结论与结果汇总.md`
- Reference: `results/updated_dataset_baseline_ratio_comparison_20260513_seed44/test_summary.csv`
- Reference: `results/current_snr_noise_6_4_seed44_artifacts/summary.csv`

**Step 1: 检查总表 sheet 结构与文档现状**

Run: `python` / `openpyxl` 读取 Excel sheet 名和前几行；`Get-Content` 查看两份文档。

**Step 2: 确认最新结果来源**

Run: 检查 `results/updated_dataset_baseline_ratio_comparison_20260513_seed44/test_summary.csv` 与 `results/current_snr_noise_6_4_seed44_artifacts/summary.csv`。

**Step 3: 记录需要更新的位置**

明确 Excel 中“基线对比”“SNR噪声对比”等 sheet，和两份文档中涉及旧结果、旧路径、旧 SNR 档位的段落与表格。

### Task 2: 更新 Excel 总表

**Files:**
- Modify: `outputs/results_summary/CAPT-UniShape_实验结果总表_含自测公开训练结果汇总.xlsx`

**Step 1: 用脚本读取最新 CSV**

读取最新对比测试和噪声测试 CSV，按表头转换为 Excel 需要的显示格式。

**Step 2: 覆盖写入对应 sheet**

使用 `openpyxl` 定位目标 sheet，清空旧数据区并写入新表头、新数据和必要的更新时间/来源说明。

**Step 3: 保存并重新读取验证**

重新打开 Excel，检查 sheet 名、行数、关键单元格值是否已更新。

### Task 3: 更新两份说明文档

**Files:**
- Modify: `docs/模型与实验.md`
- Modify: `docs/论文结论与结果汇总.md`

**Step 1: 修正文档中的结果来源和实验口径**

把旧的基线对比与噪声结果来源替换为最新目录，确保比例、模型集合、SNR 档位一致。

**Step 2: 更新表格与关键结论**

按最新结果改写对比测试结果、噪声测试结果、结论段和风险说明，避免继续引用旧数值。

**Step 3: 保持文档内部一致**

核对同一文件内的表格、正文、脚本路径、结果来源是否一致。

### Task 4: 验证

**Files:**
- Test: `outputs/results_summary/CAPT-UniShape_实验结果总表_含自测公开训练结果汇总.xlsx`
- Test: `docs/模型与实验.md`
- Test: `docs/论文结论与结果汇总.md`

**Step 1: 重新读取 Excel 关键 sheet**

Run: `python` / `openpyxl` 打印关键 sheet 的前几行与关键值。

**Step 2: 搜索文档中的旧路径和旧结果**

Run: `rg` 搜索过期结果目录、过期 SNR 列表或旧实验口径，确认已替换。

**Step 3: 检查 git diff**

Run: `git diff -- outputs/results_summary/... docs/模型与实验.md docs/论文结论与结果汇总.md`

