import pandas as pd

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
