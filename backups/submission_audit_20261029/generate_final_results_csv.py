import pandas as pd
import numpy as np

# Load original verified ledger
df = pd.read_csv('reports/verified_evidence_results_ledger.csv')

# Calculate ddof=1 fold standard deviation from fold_std_ddof0_pct (10 folds: std_ddof1 = std_ddof0 * sqrt(10/9))
# where fold_std_ddof0_pct is present
def calc_ddof1(val):
    if pd.isna(val):
        return np.nan
    return round(val * np.sqrt(10.0 / 9.0), 2)

df['fold_std_ddof1_pct'] = df['fold_std_ddof0_pct'].apply(calc_ddof1)

# Refine scientific_claim_type to be fully nuanced and defensible
def refine_claim_type(row):
    group = str(row['experiment_group'])
    claim = str(row['scientific_claim_type'])
    if 'Nested CV' in group or 'Nested' in claim:
        return 'Nested Hyperparameter Tuning on Pre-Selected Backbone'
    elif 'Exploratory' in claim or 'Ablation' in claim:
        return 'Exploratory'
    elif 'Confirmatory' in claim:
        return 'Confirmatory Baseline'
    return claim

df['scientific_claim_type'] = df.apply(refine_claim_type, axis=1)

# Ensure required column ordering
desired_cols = [
    'condition', 'experiment_group', 'classifier', 'feature_extractor', 'evaluation_level',
    'n_participants', 'n_hc', 'n_mdd', 'tn', 'fp', 'fn', 'tp',
    'accuracy_pct', 'sensitivity_pct', 'specificity_pct', 'precision_pct', 'f1_pct', 'roc_auc_pct',
    'fold_mean_acc_pct', 'fold_std_ddof0_pct', 'fold_std_ddof1_pct',
    'artifact_source_path', 'verification_status', 'scientific_claim_type'
]

# Check missing columns
for c in desired_cols:
    if c not in df.columns:
        print(f"Adding missing column: {c}")
        df[c] = np.nan

df_out = df[desired_cols]

out_path = 'reports/FINAL_EXPERIMENTAL_RESULTS.csv'
df_out.to_csv(out_path, index=False)
print(f"Wrote {len(df_out)} rows to {out_path}")
