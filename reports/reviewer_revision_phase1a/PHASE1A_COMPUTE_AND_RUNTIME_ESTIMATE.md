# PHASE 1A — COMPUTE, STORAGE, AND RUNTIME ESTIMATES
## Jha MDD EEG SPWVD Research Paper — Forensic Reviewer Revision

**Auditor / Computational Lead:** Senior Biomedical Signal Processing & Machine Learning Reproducibility Auditor  
**Date:** November 7, 2026  
**Hardware Platform:** Apple Silicon M-Series (macOS Darwin 24.x, 8-16 Performance/Efficiency Cores, Unified Memory Architecture)  
**Execution Runtimes:** Python 3.12.9 (`.venv`), PyTorch 2.6.0 (Apple MPS / Accelerate), Scikit-Learn 1.6.1 (OpenMP multi-threaded)

---

## 1. Computational Benchmark Estimates by Revision Experiment

The table below details the estimated runtime, CPU/GPU utilization, and memory footprint for each proposed revision experiment, based on local hardware profiling:

| Revision Module | Task Description | Operations & Scaling | Estimated Runtime (Apple Silicon) | Memory Footprint (RAM) | Primary Compute Bottleneck |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Artifact Diagnostic** | Peak-to-peak amplitude screening across 181 EDF files | 181 EDF reads, 19 channels $\times$ 512 samples/epoch | **~45 seconds** | ~1.5 GB | Disk I/O (EDF reads) |
| **Welch PSD Baseline** | 5 physiological band-power features across 19 leads | SciPy Welch FFT on 1,800 segments + 10-fold CV | **~30 seconds** | ~500 MB | CPU vectorized FFT |
| **STFT + ResNet-18** | Hann-windowed STFT on 1,800 segments + ResNet-18 forward pass | 1,800 2D spectrograms resized to $224 \times 224$ + PyTorch feature extraction | **~2.5 minutes** | ~2.0 GB | PyTorch Conv forward pass |
| **CWT + ResNet-18** | Morlet continuous wavelet transform on 1,800 segments + ResNet-18 | Complex convolution across 64 scales $\times$ 1,800 segments | **~4.0 minutes** | ~3.0 GB | CWT convolution matrix |
| **Full Backbone Matrix** | Evaluate all 8 classifiers on existing cached features (4 CNNs) | $4 \times 8 = 32$ models under 10-fold CV on pre-cached `.npy` arrays | **~1.5 minutes** | ~1.2 GB | Scikit-Learn SVM/Ensemble solvers |
| **Nested Group CV** | Outer 10-Fold $\times$ Inner 5-Fold grid search on top 3 classifiers | 50 folds $\times$ 18 parameter combinations per classifier | **~3.0 minutes** | ~1.5 GB | Inner-CV SVM optimization loops |
| **Repeated Group CV** | 10 independent partition seeds $\times$ 10 folds ($100\text{ runs}$) for top 3 classifiers | 100 fold iterations on cached 512-D ResNet-18 embeddings | **~2.0 minutes** | ~1.0 GB | Linear solver repetitions |
| **ResNet-18 Fine-Tuning** | End-to-end backpropagation on unfreezed layer 4 (10 folds) | 10 folds $\times$ 15 epochs $\times$ 870 images, batch size 32 | **~8.0 minutes** | ~4.0 GB | PyTorch backpropagation & autograd |
| **Cumulative Estimate** | **Full Revision Experimental Suite** | **All 8 revision modules combined** | **~22.0 minutes** | **Peak: ~4.0 GB** | **Well within local laptop capability** |

---

## 2. Disk Storage Requirements for Versioned Experiment Outputs

All new revision outputs will be written to `results/revision_experiments/` without modifying historical baseline files:

| Target Directory | Content & Files | Estimated Disk Space |
| :--- | :--- | :---: |
| `results/revision_experiments/artifacts/` | Peak-to-peak distribution tables, artifact rejection CSVs | ~5 MB |
| `results/revision_experiments/baselines/` | Welch PSD features, STFT & CWT `.npy` embeddings, baseline metric summaries | ~60 MB |
| `results/revision_experiments/backbone_matrix/` | 32-model full evaluation tables and out-of-fold prediction CSVs | ~25 MB |
| `results/revision_experiments/nested_cv/` | Best hyperparameter logs and outer test fold predictions | ~10 MB |
| `results/revision_experiments/repeated_cv/` | 100-run iteration metrics, partition stability CSVs | ~15 MB |
| `results/revision_experiments/finetune/` | Fine-tuned checkpoint weights (best epoch) and out-of-fold predictions | ~180 MB |
| **Total Revision Footprint** | **All versioned experimental outputs** | **~295 MB** |

---

## 3. Acceleration and Parallelization Strategies

1. **PyTorch Apple Silicon Metal Performance Shaders (MPS)**:
   - ResNet-18 forward passes for STFT/CWT representations and end-to-end fine-tuning will leverage `torch.device("mps")` or multi-threaded CPU Accelerate framework.
2. **In-Memory Cache Reuse**:
   - The full backbone matrix ($4 \times 8 = 32$ evaluations) operates directly on existing cached `.npy` files (`features_resnet18_*.npy`, `features_resnet50_*.npy`, `features_alexnet_*.npy`, `features_vgg16_*.npy`), avoiding any redundant feature extraction.
3. **Multi-Threaded Cross-Validation**:
   - Scikit-learn estimators will utilize `n_jobs=-1` (leveraging all available performance cores) during grid searches and repeated CV runs.
4. **Conclusion**:
   - The total execution time for all proposed reviewer experiments is approximately **22 minutes**, making complete experimental revision on the local machine entirely feasible without requiring external cloud GPU infrastructure.

