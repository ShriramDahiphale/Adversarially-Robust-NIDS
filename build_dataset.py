import glob
import pandas as pd

folder = r"D:\DL Project\data\MachineLearningCVE"
files = glob.glob(folder + r"\*.csv")
print(f"Found {len(files)} files")

df = pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)
df.columns = [c.strip() for c in df.columns]
print(df["Label"].value_counts())

# Keep 20% of each label so rare attack types stay represented
df = df.groupby("Label", group_keys=False).sample(frac=0.2, random_state=42)
print(f"Sampled rows: {len(df)}")

df.to_csv(r"D:\DL Project\data\combined_sample.csv", index=False)