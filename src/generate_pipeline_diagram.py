"""
Script: generate_pipeline_diagram.py
Purpose: Generates a publication-grade, professional architecture schematic for
         the Jha et al. MDD EEG classification pipeline (Figure 1).
Layout: 3x2 symmetric grid with clean, non-overlapping serpentine flow:
        Row 1 (left-to-right): Box 1 -> Box 2 -> Box 3
        Right transition: Box 3 -> Box 4 (straight downward vertical arrow)
        Row 2 (right-to-left): Box 4 -> Box 5 -> Box 6 (straight leftward horizontal arrows)
Output: results/pipeline_architecture_diagram.png (300 DPI)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_diagram(output_path="results/pipeline_architecture_diagram.png"):
    fig, ax = plt.subplots(figsize=(13.2, 6.6), dpi=300)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    boxes = [
        {
            "xy": (0.6, 3.65), "w": 3.5, "h": 2.35,
            "title": "1. Multi-Channel EEG",
            "lines": [
                "HUSM Clinical Cohort (N = 64)",
                "19 Monopolar 10–20 Leads",
                "Sampling Frequency: 256 Hz",
                "Resting EC & EO States"
            ],
            "color": "#EBF3FB", "ec": "#1E6091"
        },
        {
            "xy": (4.85, 3.65), "w": 3.5, "h": 2.35,
            "title": "2. Preprocessing & Slicing",
            "lines": [
                "0.5 – 45.0 Hz Bandpass Filter",
                "Zero-Phase IIR Filtering",
                "Epoch Window: T = 2.0 s (N = 512)",
                "15 Non-overlapping Epochs / Rec"
            ],
            "color": "#EDF7ED", "ec": "#2D6A4F"
        },
        {
            "xy": (9.1, 3.65), "w": 3.5, "h": 2.35,
            "title": "3. Joint SPWVD Spectral Maps",
            "lines": [
                "Analytic Signal via Hilbert",
                "Kaiser Lag Window (β = 5.0, L = 255)",
                "Hamming Time Window (L = 31)",
                "224 × 224 × 3 RGB Standardization"
            ],
            "color": "#FEF3E2", "ec": "#D97706"
        },
        {
            "xy": (9.1, 0.6), "w": 3.5, "h": 2.35,
            "title": "4. Deep Transfer Learning",
            "lines": [
                "Pretrained ResNet-18 (ImageNet)",
                "Frozen Convolutional Weights",
                "Global Average Pooling (avgpool)",
                "512-D Feature Representation"
            ],
            "color": "#F3E8FF", "ec": "#7C3AED"
        },
        {
            "xy": (4.85, 0.6), "w": 3.5, "h": 2.35,
            "title": "5. Group 10-Fold Cross-Validation",
            "lines": [
                "Stratified Group 10-Fold Isolation",
                "Zero Participant Leakage (Train/Test)",
                "sub-HS15 Multi-run Isolated to Fold 9",
                "StandardScaler Fit on Train Folds Only"
            ],
            "color": "#EDE9FE", "ec": "#4C1D95"
        },
        {
            "xy": (0.6, 0.6), "w": 3.5, "h": 2.35,
            "title": "6. Classification & Aggregation",
            "lines": [
                "Downstream Estimators (SVM, NN, KNN)",
                "Segment Posterior p(MDD | x)",
                "Mean Posterior Aggregation per Subject",
                "Participant Diagnostic: P̂_i ≥ 0.5"
            ],
            "color": "#FCE7F3", "ec": "#BE185D"
        }
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch(
            b["xy"], b["w"], b["h"], boxstyle="round,pad=0.08",
            facecolor=b["color"], edgecolor=b["ec"], linewidth=2.0
        )
        ax.add_patch(rect)
        
        # Title Banner
        ax.text(
            b["xy"][0] + b["w"] / 2.0, b["xy"][1] + b["h"] - 0.35,
            b["title"], ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=b["ec"], fontfamily="sans-serif"
        )
        
        # Subtle horizontal divider line beneath title
        ax.plot(
            [b["xy"][0] + 0.25, b["xy"][0] + b["w"] - 0.25],
            [b["xy"][1] + b["h"] - 0.62, b["xy"][1] + b["h"] - 0.62],
            color=b["ec"], alpha=0.35, lw=1.0
        )
        
        # Bullet text lines
        for idx, l in enumerate(b["lines"]):
            ax.text(
                b["xy"][0] + b["w"] / 2.0, b["xy"][1] + b["h"] - 0.98 - idx * 0.36,
                l, ha="center", va="center", fontsize=9.0, color="#1F2937", fontfamily="sans-serif"
            )

    # Clean, crisp arrows with zero overlapping
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#374151")

    # 1 -> 2 (Top row: left to center)
    ax.annotate("", xy=(4.80, 4.82), xytext=(4.15, 4.82), arrowprops=arrow_kw)

    # 2 -> 3 (Top row: center to right)
    ax.annotate("", xy=(9.05, 4.82), xytext=(8.40, 4.82), arrowprops=arrow_kw)

    # 3 -> 4 (Right side: clean straight vertical drop)
    ax.annotate("", xy=(10.85, 3.00), xytext=(10.85, 3.60), arrowprops=arrow_kw)

    # 4 -> 5 (Bottom row: right to center)
    ax.annotate("", xy=(8.40, 1.77), xytext=(9.05, 1.77), arrowprops=arrow_kw)

    # 5 -> 6 (Bottom row: center to left)
    ax.annotate("", xy=(4.15, 1.77), xytext=(4.80, 1.77), arrowprops=arrow_kw)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated publication pipeline diagram: {output_path}")

if __name__ == "__main__":
    generate_diagram()
