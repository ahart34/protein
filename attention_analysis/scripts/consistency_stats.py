import pandas as pd

# Load CSV
df = pd.read_csv(
    "attention_analysis/results/consistency.csv",
    header=None,
    usecols=[0, 1, 2],
    names=["model", "result_file", "mean_correlation"]
)

# Group by model and compute mean and std of mean_correlation
summary = df.groupby("model")["mean_correlation"].agg(["mean", "std"]).reset_index()
out_path = "attention_analysis/results/consistency_summary.csv"
summary.to_csv(out_path, index=False)

print(f"Wrote summary to {out_path}")
print(summary)