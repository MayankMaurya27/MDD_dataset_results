"""Generate deterministic stratified group 10-fold cross-validation fold assignments.

Participants:
- 30 Healthy Controls (sub-HS1 to sub-HS30)
- 34 MDD Patients (sub-MDDS1 to sub-MDDS34)
Total = 64 participants.

Fold configuration:
- 10 folds
- Exactly 3 HC per fold (3 * 10 = 30)
- 4 folds with 4 MDD, 6 folds with 3 MDD (4*4 + 6*3 = 34)
- Folds 0-3: 7 participants (3 HC, 4 MDD)
- Folds 4-9: 6 participants (3 HC, 3 MDD)
"""

import os
import pandas as pd
import numpy as np

def generate_folds(seed=42):
    rng = np.random.RandomState(seed)
    
    inventory_path = "reports/dataset_inventory.csv"
    if not os.path.exists(inventory_path):
        raise FileNotFoundError(f"{inventory_path} not found. Run audit first.")
        
    df_inv = pd.read_csv(inventory_path)
    
    # Get unique participants and their labels
    participants = df_inv[['participant_id', 'group_label', 'numeric_label']].drop_duplicates().sort_values('participant_id').reset_index(drop=True)
    
    hc_subs = participants[participants['numeric_label'] == 0]['participant_id'].tolist()
    mdd_subs = participants[participants['numeric_label'] == 1]['participant_id'].tolist()
    
    assert len(hc_subs) == 30, f"Expected 30 HC, got {len(hc_subs)}"
    assert len(mdd_subs) == 34, f"Expected 34 MDD, got {len(mdd_subs)}"
    
    # Shuffle with fixed seed
    shuffled_hc = list(rng.permutation(hc_subs))
    shuffled_mdd = list(rng.permutation(mdd_subs))
    
    fold_assignments = {}
    
    # Assign 3 HC to each of the 10 folds
    for f in range(10):
        fold_hc = shuffled_hc[f*3 : (f+1)*3]
        for s in fold_hc:
            fold_assignments[s] = f
            
    # Assign MDD: 4 MDD to folds 0-3, 3 MDD to folds 4-9
    mdd_idx = 0
    for f in range(10):
        n_mdd = 4 if f < 4 else 3
        fold_mdd = shuffled_mdd[mdd_idx : mdd_idx + n_mdd]
        mdd_idx += n_mdd
        for s in fold_mdd:
            fold_assignments[s] = f
            
    assert mdd_idx == 34, f"Assigned {mdd_idx} MDD, expected 34"
    
    participants['fold'] = participants['participant_id'].map(fold_assignments)
    participants['group'] = participants['group_label'].map({'HC': 'Healthy Control', 'MDD': 'Major Depressive Disorder'})
    
    # Add scan counts and conditions from inventory
    scan_counts = df_inv.groupby('participant_id').size().to_dict()
    participants['num_recordings'] = participants['participant_id'].map(scan_counts)
    
    # Sort by fold then participant_id
    participants.sort_values(by=['fold', 'numeric_label', 'participant_id'], inplace=True)
    
    os.makedirs('results', exist_ok=True)
    out_csv = 'results/fold_assignments.csv'
    participants.to_csv(out_csv, index=False)
    print(f"Generated {out_csv} with {len(participants)} participants.")
    
    # Print fold summary
    summary = participants.groupby('fold').agg(
        total_participants=('participant_id', 'count'),
        hc_count=('numeric_label', lambda x: (x == 0).sum()),
        mdd_count=('numeric_label', lambda x: (x == 1).sum()),
        total_scans=('num_recordings', 'sum')
    )
    print("\nFold Allocation Summary:")
    print(summary)
    
    return participants

if __name__ == '__main__':
    generate_folds(seed=42)
