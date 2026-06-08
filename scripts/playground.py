import pandas as pd
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PATH = "results/representations/resnet50_class_metrics.parquet"

df = pd.read_parquet(PATH)
print(df.head())
print(len(df))
print(set(df['corruption']))
print(set(df['severity']))
print('synsets', len(df['synset']))
print('synsets uni', len(set(df['synset'])))

print("metrics: ", set(df['metric']))


print(df[df['metric'] == 'tangential_fraction_iqr']['value'])

print(df[df['metric'] == 'relative_shift_iqr']['value'])


print(df.columns)

from model import get_model 

print(list(get_model("convnext_base")[0].classifier.children())[0])