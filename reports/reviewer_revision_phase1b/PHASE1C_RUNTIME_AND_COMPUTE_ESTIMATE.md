# PHASE 1C — COMPUTATIONAL PROFILING, RUNTIME ESTIMATE, AND STORAGE BUDGET
## Jha MDD EEG SPWVD Research Paper — Forensic Reviewer Revision

**Auditor / Lead Engineer:** Senior Biomedical Signal Processing & Machine Learning Reproducibility Auditor  
**Date:** November 7, 2026  
**Hardware Platform:** Apple Silicon M-Series (macOS Darwin 24.x, Unified Memory Architecture, 8–16 CPU Cores, Metal Performance Shaders MPS)  
**Deliverable Directory:** `reports/reviewer_revision_phase1b/`  
**Execution Context:** Pre-execution computational profiling for Phase 1C experimental modules.

---

## 1. Benchmarked Runtime Profile Across Phase 1C Modules

The table below provides realistic wall-clock execution estimates for all six proposed Phase 1C revision modules, derived from local hardware profiling in Phase 1A and Phase 1B:

| Module # | Experimental Module Name | Computational Operations & Complexity | Estimated Runtime (Apple Silicon) | Peak RAM Footprint | Hardware Acceleration Strategy |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **M1** | **Full 4-Backbone $\times$ 8-Classifier Matrix** | 32 models $\times$ 10 folds evaluated on pre-cached `.npy` feature arrays ($512\text{-D}$ to $4,096\text{-D}$) | **~1.5 minutes** | ~1.5 GB | Scikit-Learn multi-threading (`n_jobs=-1`) on existing cached embeddings |
| **M2** | **Complete-Pipeline Nested Group CV** | Outer 10-fold $\times$ Inner 5-fold ($50\text{ inner folds}$) evaluating 4 representations $\times$ 3 classifiers | **~3.5 minutes** | ~2.0 GB | Parallel inner grid evaluations over memory-resident feature matrices |
| **M3** | **Repeated Group 10-Fold CV (100 Runs)** | 10 independent partition seeds $\times$ 10 folds = 100 fold iterations on 512-D ResNet-18 embeddings | **~2.0 minutes** | ~1.2 GB | Linear and RBF SVM solvers running across multi-core CPU |
| **M4** | **End-to-End ResNet-18 Fine-Tuning** | 10 folds $\times$ 15 epochs backpropagation on layer 4 convolutions (870/930 images, batch size 32) | **~8.5 minutes** | ~4.2 GB | PyTorch Metal Performance Shaders (`torch.device("mps")`) with mixed precision |
| **M5** | **Fine Gaussian SVM Rectification** | Exact mathematical kernel scaling comparison on 512-D space | **~15 seconds** | ~500 MB | Fast analytical Gram matrix computation |
| **M6** | **Artifact-Controlled Pipeline Reruns** | 181 continuous FastICA decompositions + SPWVD/STFT/CWT feature re-extraction + 10-fold CV | **~7.5 minutes** | ~3.5 GB | Vectorized FastICA unmixing + PyTorch batch feature extraction |
| **Total** | **Complete Phase 1C Experimental Suite** | **All 6 revision modules executed sequentially** | **~23.0 minutes** | **Peak: ~4.2 GB** | **100% executable locally without external cloud GPU dependencies** |

---

## 2. Disk Storage Budget for Versioned Phase 1C Outputs

All Phase 1C outputs will be written to `results/revision_experiments/phase1c/` without altering historical baseline files:

| Target Subdirectory | Contents & Generated Files | Estimated Disk Footprint |
| :--- | :--- | :---: |
| `phase1c/backbone_matrix/` | 32-model segment and participant OOF prediction CSVs, summary tables | ~35 MB |
| `phase1c/nested_cv/` | Nested outer fold prediction CSVs, parameter selection logs per fold | ~15 MB |
| `phase1c/repeated_cv/` | 100-run partition metrics CSV, stability distribution tables | ~20 MB |
| `phase1c/finetune/` | Fine-tuned PyTorch model state checkpoints (best epoch per fold) and predictions | ~195 MB |
| `phase1c/artifact_controlled/` | Cleaned feature caches, OOF predictions for clean SPWVD/Welch/STFT/CWT | ~120 MB |
| **Total Storage Budget** | **Complete Phase 1C Data Footprint** | **~385 MB** |

---

## 3. Computational Safety and Memory Management Protocol

To ensure seamless execution without thermal throttling, swap thrashing, or memory leaks:
1. **Garbage Collection Between Modules:** Each script will explicitly invoke `gc.collect()` and `torch.mps.empty_cache()` between folds to prevent unified memory accumulation during fine-tuning.
2. **Intermediate Checkpointing:** Outer fold predictions will be saved to disk incrementally after each fold completes, ensuring that a process interruption does not require restarting from fold 0.
3. **Execution Safety Boundary:** Scripts will log start timestamps, elapsed time, peak Resident Set Size (RSS), and Python/library versions into a companion `phase1c_execution_log.yaml`.
