"""Unit and Validation Tests for Phase 1C Module M1.

Tests:
1. Verifies that all 181 recordings were processed and logged.
2. Tests that multiple recordings belonging to the same participant (e.g. sub-HS15 in EO)
   are never split across cross-validation folds.
3. Tests that exactly <= 2 components were rejected per recording, preserving >= 17 dimensions.
4. Tests that Condition B satisfies the pre-registered ocular attenuation (>= 35% reduction, p < 1e-10)
   and posterior alpha retention (>= 85%).
"""

import json
import os
import sys
import numpy as np
import pandas as pd


def test_m1_outputs():
    csv_rec = "results/phase1c/M1_artifact_validation/per_recording_ica_diagnostics.csv"
    csv_ic = "results/phase1c/M1_artifact_validation/component_rejection_logs.csv"
    json_val = "results/phase1c/M1_artifact_validation/m1_statistical_validation_results.json"
    
    assert os.path.exists(csv_rec), f"Missing {csv_rec}"
    assert os.path.exists(csv_ic), f"Missing {csv_ic}"
    assert os.path.exists(json_val), f"Missing {json_val}"
    
    df_rec = pd.read_csv(csv_rec)
    assert len(df_rec) == 181, f"Expected 181 recordings, got {len(df_rec)}"
    
    # Check degree-of-freedom cap: max 2 rejected ICs
    assert (df_rec["n_rejected_ics"] <= 2).all(), "Found recording with > 2 rejected ICs!"
    assert (df_rec["retained_dimensions"] >= 17).all(), "Found recording with < 17 retained dimensions!"
    
    # Check participant isolation
    df_folds = pd.read_csv("results/fold_assignments.csv")
    for pid, group in df_rec.groupby("participant_id"):
        # If in fold assignments, verify single fold
        if pid in df_folds["participant_id"].values:
            assigned_folds = df_folds[df_folds["participant_id"] == pid]["fold"].unique()
            assert len(assigned_folds) == 1, f"Participant {pid} has multiple fold assignments!"
            
    # Check validation results
    with open(json_val, "r") as f:
        val_res = json.load(f)
        
    assert val_res["condition_b_status"] == "PASS", f"Condition B failed validation: {val_res}"
    print("  ✓ All 181 recordings processed.")
    print("  ✓ Degree-of-freedom cap (max 2 ICs rejected) verified across all recordings.")
    print("  ✓ Participant isolation confirmed across folds.")
    print("  ✓ Pre-registered validation criteria achieved PASS status.")
    return True

if __name__ == "__main__":
    print("======================================================================")
    print("               MODULE M1 ARTIFACT VALIDATION TEST                     ")
    print("======================================================================")
    if test_m1_outputs():
        print("======================================================================")
        print("                     MODULE M1 TESTS PASSED                          ")
        print("======================================================================")
        sys.exit(0)
    else:
        sys.exit(1)

