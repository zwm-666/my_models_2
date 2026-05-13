# Public AC Voltage Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 清理公开 AC Voltage 数据集里因 `mixed_stratified` 行级随机切分导致失真的结果，并为后续无泄漏重跑准备统一口径。

**Architecture:** 先保留总表中的原始数值，但把公开数据集区块整体标记为失效，明确说明不能用于论文主结论。随后优先采用无泄漏协议重跑公开数据，其中首选 `old_to_new` / `new_to_old` 跨 MEA 域协议；若需要保持 train/test 比例展示，再补充分组切分方案。

**Tech Stack:** `openpyxl`, PowerShell, `D:\python3.9\python.exe`

---

### Task 1: 标记总表中失效的公开结果

**Files:**
- Modify: `outputs/results_summary/CAPT-UniShape_实验结果总表_含自测公开训练结果汇总.xlsx`

**Step 1: 读取公开数据集区块**

定位 `训练结果总汇总` sheet 中 `公开数据集(AC Voltage)` 的全部行。

**Step 2: 写入失效备注**

保留原始指标数值，将备注改为“结果失效：mixed_stratified 按行随机切分导致同源样本泄漏，禁止用于论文主表；待按无泄漏协议重跑”，并补充来源说明。

**Step 3: 保存并读回校验**

确认公开区块全部已带失效标记，且未影响自测数据集区块。

### Task 2: 形成后续无泄漏重跑口径

**Files:**
- Review: `scripts/run_public_ac_voltage_baselines.py`
- Review: `scripts/build_ac_voltage_npz.py`

**Step 1: 确认现有无泄漏协议**

核对 `old_to_new` / `new_to_old` 是否已可直接运行，并确认其结果字段能否回填总表。

**Step 2: 给出回填口径**

明确“比例”列在无泄漏协议下如何表示，例如写成 `old->new` 与 `new->old`。

**Step 3: 选择后续执行路径**

若用户确认跨域协议，则直接重跑并回填；若用户坚持比例展示，则新增分组切分协议后再跑。
