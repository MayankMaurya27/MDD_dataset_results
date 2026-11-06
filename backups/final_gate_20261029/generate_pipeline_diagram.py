import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis("off")

# Define stages
boxes = [
    {
        "xy": (0.4, 3.4), "w": 2.2, "h": 1.8,
        "title": "1. Multi-Channel EEG",
        "lines": ["HUSM Cohort (N=64)", "19 Monopolar 10-20 Leads", "Sampling Rate: 256 Hz", "Resting EC & EO States"],
        "color": "#E3F2FD", "ec": "#1565C0"
    },
    {
        "xy": (3.2, 3.4), "w": 2.4, "h": 1.8,
        "title": "2. Preprocessing & Slicing",
        "lines": ["0.5 – 45.0 Hz Bandpass", "Zero-Phase IIR Filtering", "Window: T = 2.0s (N=512)", "15 Non-overlapping Epochs"],
        "color": "#E8F5E9", "ec": "#2E7D32"
    },
    {
        "xy": (6.2, 3.4), "w": 2.5, "h": 1.8,
        "title": "3. Joint SPWVD Spectral Maps",
        "lines": ["Analytic Signal via Hilbert", "Kaiser Lag Smoothing (β=5.0)", "Hamming Time Window (L=31)", "224×224×3 RGB Standardization"],
        "color": "#FFF3E0", "ec": "#E65100"
    },
    {
        "xy": (9.3, 3.4), "w": 2.3, "h": 1.8,
        "title": "4. Deep Transfer Learning",
        "lines": ["Pretrained ResNet-18 (ImageNet)", "Frozen Convolutional Weights", "Global Avg Pooling (avgpool)", "512-D Feature Representation"],
        "color": "#F3E5F5", "ec": "#6A1B9A"
    },
    {
        "xy": (2.0, 0.6), "w": 3.6, "h": 1.8,
        "title": "5. Group 10-Fold Cross-Validation",
        "lines": ["Stratified Group 10-Fold Isolation", "Zero Participant Overlap (Train/Test)", "sub-HS15 Multi-run Isolated to Fold 9", "StandardScaler Fit on Train Folds Only"],
        "color": "#EDE7F6", "ec": "#4527A0"
    },
    {
        "xy": (6.6, 0.6), "w": 3.6, "h": 1.8,
        "title": "6. Classification & Aggregation",
        "lines": ["Downstream Estimators (Linear SVM, RBF)", "Segment Probability p(MDD | x)", "Mean Posterior Aggregation per Subject", "Participant Diagnostic: P̂_i ≥ 0.5"],
        "color": "#FCE4EC", "ec": "#AD1457"
    }
]

for b in boxes:
    rect = patches.FancyBboxPatch(
        b["xy"], b["w"], b["h"], boxstyle="round,pad=0.1",
        facecolor=b["color"], edgecolor=b["ec"], linewidth=2.0
    )
    ax.add_patch(rect)
    # Title
    ax.text(
        b["xy"][0] + b["w"]/2.0, b["xy"][1] + b["h"] - 0.28,
        b["title"], ha="center", va="center",
        fontsize=10.5, fontweight="bold", color=b["ec"]
    )
    # Lines
    for idx, l in enumerate(b["lines"]):
        ax.text(
            b["xy"][0] + b["w"]/2.0, b["xy"][1] + b["h"] - 0.65 - idx*0.28,
            l, ha="center", va="center", fontsize=8.5, color="#212121"
        )

# Arrows
arrow_props = dict(facecolor="#37474F", edgecolor="#37474F", width=1.5, headwidth=7, shrink=0.05)

# 1 -> 2
ax.annotate("", xy=(3.2, 4.3), xytext=(2.6, 4.3), arrowprops=arrow_props)
# 2 -> 3
ax.annotate("", xy=(6.2, 4.3), xytext=(5.6, 4.3), arrowprops=arrow_props)
# 3 -> 4
ax.annotate("", xy=(9.3, 4.3), xytext=(8.7, 4.3), arrowprops=arrow_props)

# 4 -> 5 (turn around down to row 2)
ax.annotate("", xy=(5.6, 1.5), xytext=(10.4, 3.4),
            arrowprops=dict(facecolor="#37474F", edgecolor="#37474F", width=1.5, headwidth=7, shrink=0.05,
                            connectionstyle="arc3,rad=-0.25"))

# 5 -> 6
ax.annotate("", xy=(6.6, 1.5), xytext=(5.6, 1.5), arrowprops=arrow_props)

plt.tight_layout()
out_fig = "results/pipeline_architecture_diagram.png"
plt.savefig(out_fig, bbox_inches="tight", dpi=300)
print(f"Generated genuine pipeline diagram: {out_fig}")
