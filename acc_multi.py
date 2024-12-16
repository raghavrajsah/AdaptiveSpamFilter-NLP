import pandas as pd

file1_path = "good_test.tsv"
file2_path = "good_preds.tsv" 
output_path = "metrics_by_language1.tsv"

file1 = pd.read_csv(file1_path, sep="\t", encoding="utf-8")
file2 = pd.read_csv(file2_path, sep="\t", encoding="utf-8")

file1.columns = file1.columns.str.strip().str.lower()
file2.columns = file2.columns.str.strip().str.lower()

merged = pd.merge(file1, file2, on="textid", how="inner", suffixes=("_file1", "_file2"))

merged["target"] = merged["target_file1"]

merged = merged.drop(columns=["target_file2"])

merged["predicted_numeric"] = merged["predicted"].str.split("_").str[-1].astype(int)

merged["correct"] = (merged["target"] == merged["predicted_numeric"]).astype(int)
merged["true_positive"] = ((merged["target"] == 1) & (merged["predicted_numeric"] == 1)).astype(int)
merged["false_positive"] = ((merged["target"] == 0) & (merged["predicted_numeric"] == 1)).astype(int)
merged["false_negative"] = ((merged["target"] == 1) & (merged["predicted_numeric"] == 0)).astype(int)

metrics_by_language = (
    merged.groupby("condition")
    .agg(
        total=("correct", "size"),
        correct=("correct", "sum"),
        true_positive=("true_positive", "sum"),
        false_positive=("false_positive", "sum"),
        false_negative=("false_negative", "sum"),
    )
    .reset_index()
)

metrics_by_language["accuracy"] = metrics_by_language["correct"] / metrics_by_language["total"]
metrics_by_language["precision"] = metrics_by_language["true_positive"] / (
    metrics_by_language["true_positive"] + metrics_by_language["false_positive"]
)
metrics_by_language["recall"] = metrics_by_language["true_positive"] / (
    metrics_by_language["true_positive"] + metrics_by_language["false_negative"]
)

overall = pd.DataFrame({
    "condition": ["total"],
    "total": [merged["correct"].size],
    "correct": [merged["correct"].sum()],
    "true_positive": [merged["true_positive"].sum()],
    "false_positive": [merged["false_positive"].sum()],
    "false_negative": [merged["false_negative"].sum()],
})
overall["accuracy"] = overall["correct"] / overall["total"]
overall["precision"] = overall["true_positive"] / (
    overall["true_positive"] + overall["false_positive"]
)
overall["recall"] = overall["true_positive"] / (
    overall["true_positive"] + overall["false_negative"]
)

result = pd.concat([metrics_by_language, overall], ignore_index=True)

result.to_csv(output_path, sep="\t", index=False)

