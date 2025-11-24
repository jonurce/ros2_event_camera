# Inspect the structure of the dataset

import h5py
import numpy as np
from pathlib import Path
import re

def load_labels_robust(txt_path):
    """Load label.txt even if it has quotes, spaces, or mixed types"""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        # Remove quotes, split by space/comma, filter empty
        nums = re.findall(r'"?(\d+)"?', line)
        if len(nums) >= 3:
            x, y, state = map(int, nums[:3])
            data.append([x, y, state])
    
    return np.array(data, dtype=np.int32)

def inspect_3et_with_blink(root_dir):
    root = Path(root_dir)
    print(f"3ET+ CVPR 2025 FULL ANALYSIS (WITH BLINK LABEL)\n" + "="*80)
    
    stats = {
        'total_folders': 0,
        'total_events': 0,
        'total_frames': 0,
        'open_frames': 0,
        'closed_frames': 0,
        'resolutions': set(),
        'subjects': set()
    }
    
    for split in ['train', 'test']:
        split_path = root / split
        if not split_path.exists(): continue
            
        print(f"\n[{split.upper()}]")
        print("-" * 70)
        
        for folder in sorted(split_path.iterdir()):
            if not folder.is_dir(): continue
                
            h5_file = folder / f"{folder.name}.h5"
            txt_file = folder / "label.txt"
            if not h5_file.exists() or not txt_file.exists(): 
                continue
                
            # Events
            with h5py.File(h5_file, 'r') as f:
                events = f['events'][()]
            num_events = len(events)
            
            # Labels + blink
            labels = load_labels_robust(txt_file)
            if len(labels) == 0:
                print(f"  FAILED to load labels: {txt_file}")
                continue
            x, y, state = labels[:, 0], labels[:, 1], labels[:, 2]
            open_count = np.sum(state == 0)
            closed_count = np.sum(state == 1)
            
            # Resolution
            w = events['x'].max() + 1
            h = events['y'].max() + 1
            
            stats['total_folders'] += 1
            stats['total_events'] += num_events
            stats['total_frames'] += len(labels)
            stats['open_frames'] += open_count
            stats['closed_frames'] += closed_count
            stats['resolutions'].add((w, h))
            stats['subjects'].add(folder.name)
            
            print(f"{folder.name:6} | {num_events:8,} ev | {len(labels):5} frames | "
                  f"Open: {open_count:5} ({open_count/len(labels)*100:5.1f}%) | "
                  f"Closed: {closed_count:4} ({closed_count/len(labels)*100:4.1f}%) | {w}x{h}")
    
    # FINAL REPORT
    print("\n" + "="*80)
    print("3ET+ BLINK+Gaze DATASET — FULL REPORT")
    print("="*80)
    print(f"Folders (recordings)    : {stats['total_folders']}")
    print(f"Total raw events        : {stats['total_events']:,}")
    print(f"Total labeled frames    : {stats['total_frames']:,}")
    print(f"Eye OPEN frames         : {stats['open_frames']:,} ({stats['open_frames']/stats['total_frames']*100:5.2f}%)")
    print(f"Eye CLOSED frames       : {stats['closed_frames']:,} ({stats['closed_frames']/stats['total_frames']*100:5.2f}%)")
    print(f"Avg events/frame        : {stats['total_events']/stats['total_frames']:,.1f}")
    print(f"Resolutions             : {stats['resolutions']}")
    print(f"Subjects                : {len(stats['subjects'])}")
    print("\nNEXT STEP → Preprocess to (B, C=2, T, H, W) + labels (x,y,state)")

if __name__ == "__main__":
    # path_to_event_data = '/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/event-based-eye-tracking-cvpr-2025/3ET+ dataset/event_data/'
    path_to_event_data = "/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/event-based-eye-tracking-cvpr-2025/3ET+ dataset/event_data"
    inspect_3et_with_blink(path_to_event_data)