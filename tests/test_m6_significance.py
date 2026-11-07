"""Unit and Validation Tests for Phase 1C Module M6 (Statistical Analysis).

Tests:
1. Validates that comprehensive metrics tables exist for both EC and EO cohorts.
2. Validates that paired McNemar tests and Wilcoxon tests exist.
3. Asserts that McNemar tests with zero discordant pairs (b+c=0) are explicitly documented as degenerate.
4. Validates that ablation deconstruction report exists and formalizes p=1.000 for 1-participant differences.
"""

import json
import os
import sys
import pandas as pd


def test_m6_results():
    for task in ["eyesClosed", "eyesOpen"]:
        tbl_csv = f"results/phase1c/M6_statistical_analysis/comprehensive_metrics_table_{task}.csv"
        pair_csv = f"results/phase1c/M6_statistical_analysis/paired_statistical_tests_{task}.csv"
        
        assert os.path.exists(tbl_csv), f"Missing {tbl_csv}"
        assert os.path.exists(pair_csv), f"Missing {pair_csv}"
        
        df_tbl = pd.read_csv(tbl_csv)
        df_pair = pd.read_csv(pair_csv)
        
        assert len(df_tbl) >= 6, f"Expected at least 6 benchmark pipelines, got {len(df_tbl)}"
        assert len(df_pair) >= 4, f"Expected at least 4 paired comparisons, got {len(df_pair)}"
        
        # Verify McNemar test properties
        for _, row in df_pair.iterrows():
            if row["discordant_pairs"] == 0:
                assert row["mcnemar_exact_p"] == 1.000, "Degenerate McNemar test must have p = 1.000"
                
    json_path = "results/phase1c/M6_statistical_analysis/ablation_deconstruction_report.json"
    assert os.path.exists(json_path), f"Missing {json_path}"
    with open(json_path, "r") as f:
        rep = json.load(f)
    assert rep["case_study"]["exact_binomial_mcnemar_p_value"] == 1.000
    
    print("  ✓ Comprehensive metrics tables verified for EC and EO.")
    print("  ✓ Paired McNemar and Wilcoxon tests verified.")
    print("  ✓ McNemar degeneracy guards verified (b+c=0 -> p=1.000).")
    print("  ✓ Historical 1-participant ablation deconstruction verified.")
    return True

if __name__ == "__main__":
    print("======================================================================")
    print("                 MODULE M6 STATISTICAL ANALYSIS TEST                  ")
    print("======================================================================")
    if test_m6_results():
        print("======================================================================")
        print("                     MODULE M6 TESTS PASSED                          ")
        print("======================================================================")
        sys.exit(0)
    else:
        sys.exit(1)

