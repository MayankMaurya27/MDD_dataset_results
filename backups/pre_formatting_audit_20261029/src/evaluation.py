"""Evaluation and Metrics calculation module.

Computes:
- Accuracy
- Balanced Accuracy
- Precision
- Sensitivity / Recall
- Specificity
- F1-Score
- ROC-AUC
- Confusion Matrix

Supports both participant-level (primary) and segment-level (secondary) evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Compute full classification diagnostic metrics.
    
    Parameters
    ----------
    y_true : 1D array of int
    y_pred : 1D array of int
    y_prob : 1D array of float, optional
    
    Returns
    -------
    dict of metric name -> value (percentage or decimal)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    sens = recall_score(y_true, y_pred, zero_division=0) * 100.0 # Sensitivity / Recall
    spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0 # Specificity
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    
    auc = np.nan
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_prob) * 100.0
        except Exception:
            pass
            
    return {
        "accuracy": round(acc, 2),
        "balanced_accuracy": round(bal_acc, 2),
        "precision": round(prec, 2),
        "sensitivity": round(sens, 2),
        "specificity": round(spec, 2),
        "f1_score": round(f1, 2),
        "roc_auc": round(auc, 2) if not np.isnan(auc) else None,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

def aggregate_participant_predictions(segment_df):
    """
    Aggregate segment-level predictions to participant-level predictions.
    
    Parameters
    ----------
    segment_df : pd.DataFrame containing:
        - 'participant_id'
        - 'y_true'
        - 'y_prob' (predicted probability of class 1 / MDD)
        
    Returns
    -------
    participant_df : pd.DataFrame containing:
        - 'participant_id'
        - 'y_true'
        - 'mean_prob'
        - 'y_pred'
    """
    grouped = segment_df.groupby("participant_id").agg(
        y_true=("y_true", "first"),
        mean_prob=("y_prob", "mean"),
        segment_count=("y_prob", "count")
    ).reset_index()
    
    grouped["y_pred"] = (grouped["mean_prob"] >= 0.5).astype(int)
    return grouped
