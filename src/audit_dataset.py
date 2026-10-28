"""Audit script for the Mumtaz 2016 / nemar nm000114 BIDS dataset.

Generates:
- reports/dataset_inventory.csv
- reports/dataset_audit.md
- reports/data_quality_report.md
"""

import os
import glob
import json
import pandas as pd
import numpy as np

BIDS_DIR = os.path.abspath("nm000114_v1.0.0")

def parse_edf_header(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(256)
        if len(header) < 256:
            return None
        num_records = int(header[236:244].decode('latin1').strip())
        record_duration = float(header[244:252].decode('latin1').strip())
        num_channels = int(header[252:256].decode('latin1').strip())
        
        ch_labels = [f.read(16).decode('latin1').strip() for _ in range(num_channels)]
        ch_transducer = [f.read(80).decode('latin1').strip() for _ in range(num_channels)]
        ch_dim = [f.read(8).decode('latin1').strip() for _ in range(num_channels)]
        ch_phys_min = [float(f.read(8).decode('latin1').strip()) for _ in range(num_channels)]
        ch_phys_max = [float(f.read(8).decode('latin1').strip()) for _ in range(num_channels)]
        ch_dig_min = [int(f.read(8).decode('latin1').strip()) for _ in range(num_channels)]
        ch_dig_max = [int(f.read(8).decode('latin1').strip()) for _ in range(num_channels)]
        ch_prefilt = [f.read(80).decode('latin1').strip() for _ in range(num_channels)]
        ch_samples = [int(f.read(8).decode('latin1').strip()) for _ in range(num_channels)]
        ch_reserved = [f.read(32).decode('latin1').strip() for _ in range(num_channels)]
        
        total_duration = num_records * record_duration
        fs = ch_samples[0] / record_duration if record_duration > 0 else 0
        
        return {
            'num_channels': num_channels,
            'channels': ch_labels,
            'duration_sec': total_duration,
            'num_records': num_records,
            'record_duration': record_duration,
            'fs': fs
        }

def run_audit():
    subs = sorted([d for d in os.listdir(BIDS_DIR) if d.startswith('sub-')])
    
    rows = []
    sub_summary = {}
    
    for s in subs:
        is_hc = s.startswith('sub-HS')
        group_label = 'HC' if is_hc else 'MDD'
        numeric_label = 0 if is_hc else 1
        
        scans_file = os.path.join(BIDS_DIR, s, f'{s}_scans.tsv')
        if not os.path.exists(scans_file):
            continue
            
        df_scan = pd.read_csv(scans_file, sep='\t')
        sub_summary[s] = {
            'group': group_label,
            'scan_count': len(df_scan),
            'tasks': []
        }
        
        for _, r in df_scan.iterrows():
            rel_fname = r['filename']
            full_path = os.path.join(BIDS_DIR, s, rel_fname)
            
            task = rel_fname.split('_task-')[1].split('_')[0] if '_task-' in rel_fname else 'unknown'
            run = rel_fname.split('_run-')[1].split('_')[0] if '_run-' in rel_fname else '1'
            
            sub_summary[s]['tasks'].append(task)
            
            info = parse_edf_header(full_path)
            if info is None:
                is_corrupt = True
                dur = 0
                fs = 0
                num_ch = 0
                chs = []
            else:
                is_corrupt = False
                dur = info['duration_sec']
                fs = info['fs']
                num_ch = info['num_channels']
                chs = info['channels']
                
            standard_1020_chs = [c for c in chs if any(c.startswith(f"EEG {lead}-") for lead in 
                ['Fp1','F3','C3','P3','O1','F7','T3','T5','Fz','Fp2','F4','C4','P4','O2','F8','T4','T6','Cz','Pz'])]
            
            aux_chs = [c for c in chs if c not in standard_1020_chs and 'Annotation' not in c]
            
            notes = []
            if s == 'sub-HS15' and run == '2':
                notes.append('Repeated eyesOpen run (run-2)')
            if is_corrupt:
                notes.append('Corrupt file')
                
            rows.append({
                'participant_id': s,
                'group': 'Healthy Control' if is_hc else 'Major Depressive Disorder',
                'group_label': group_label,
                'numeric_label': numeric_label,
                'task': task,
                'run': run,
                'filename': os.path.relpath(full_path, BIDS_DIR),
                'file_size_bytes': os.path.getsize(full_path) if os.path.exists(full_path) else 0,
                'duration_seconds': round(dur, 2),
                'duration_minutes': round(dur / 60.0, 2),
                'sampling_rate_hz': fs,
                'total_channels': num_ch,
                'standard_1020_channels': len(standard_1020_chs),
                'aux_channels': len(aux_chs),
                'has_annotations': any('Annotation' in c for c in chs),
                'is_corrupt': is_corrupt,
                'notes': '; '.join(notes) if notes else 'OK'
            })
            
    df_inventory = pd.DataFrame(rows)
    df_inventory.sort_values(by=['group_label', 'participant_id', 'task', 'run'], inplace=True)
    df_inventory.to_csv('reports/dataset_inventory.csv', index=False)
    print(f"Generated reports/dataset_inventory.csv with {len(df_inventory)} records.")
    
    return df_inventory, sub_summary

if __name__ == '__main__':
    run_audit()
