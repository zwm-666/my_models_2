# Proposed Accuracy Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve and run the proposed CAPT-UniShape model until a held-out test accuracy of at least 95% is reached.

**Architecture:** Keep the current official CAPT-UniShape trunk, but tighten evaluation and ablation semantics before tuning. Then run proposed-only experiments with fair split handling and training choices that can improve generalisation.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, scikit-learn, standard-library unittest.

---

### Task 1: Guard Evaluation and Ablation Semantics

**Files:**
- Create: `tests/test_evaluation_and_ablation.py`
- Modify: `evaluate.py`
- Modify: `models/modules/residual_kan_fusion.py`

**Steps:**
1. Write failing tests for split selection and disabled KAN regularisation.
2. Run the tests with `C:\Users\86191\.conda\envs\pytorch\python.exe -m unittest tests.test_evaluation_and_ablation -v`.
3. Add split-selection helpers and `--split` support to `evaluate.py`.
4. Make `ResidualKANFusion(use_residual_kan=False)` return zero KAN contribution and zero KAN regularisation.
5. Re-run the focused tests.

### Task 2: Add Proposed-Only Experiment Loop

**Files:**
- Create or modify: `scripts/run_proposed_accuracy_search.py`

**Steps:**
1. Add a script that builds NPZ data, trains only the proposed model, reads test accuracy from `metrics.json`, and stops once accuracy is at least 0.95.
2. Try fair configurations first: fixed test split, larger validation support, train+val refit, and class weighting variants.
3. Write each attempt under `results/codex_proposed_accuracy_search/`.
4. Report the winning command and metrics.

### Task 3: Run Experiments

**Commands:**
- `C:\Users\86191\.conda\envs\pytorch\python.exe scripts/run_proposed_accuracy_search.py`
- If the automatic loop is too slow on CPU, run the best single command directly with reduced search space.

**Success Criteria:**
- `metrics.json` for the proposed model contains `test.accuracy >= 0.95`.
- The result uses an explicit test split and records the exact data/training settings.
