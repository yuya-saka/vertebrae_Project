import pandas as pd
from pathlib import Path

data_dir = Path('/mnt/nfs1/home/yamamoto-hiroto/research/vertebrae_saka/data/slice_train/axial')
csv_files = list(data_dir.glob('inp*/fracture_labels_inp*.csv'))

total = 0
fracture = 0

print("Sample class distribution:")
for csv_file in sorted(csv_files)[:5]:
    df = pd.read_csv(csv_file)
    total += len(df)
    frac_count = (df['Fracture_Label'] == 1).sum()
    fracture += frac_count
    print(f'{csv_file.parent.name}: Total={len(df):4d}, Fracture={frac_count:3d} ({frac_count/len(df)*100:.2f}%)')

print(f'\nOverall (5 patients): Fracture={fracture}/{total} ({fracture/total*100:.2f}%)')
