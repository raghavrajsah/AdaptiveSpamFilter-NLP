import pandas as pd

file_path = "hindi_preds.tsv"
output_metrics_path = "metricsHi.tsv"
output_incorrect_path = "incorrect_predictionsHI.tsv"

df = pd.read_csv(file_path, sep="\t", encoding="utf-8")
df.columns = df.columns.str.strip().str.lower()

df["predicted_numeric"] = df["predicted"].str.split("_").str[-1].astype(int)
df["correct"] = (df["target"] == df["predicted_numeric"]).astype(int)
df["true_positive"] = ((df["target"] == 1) & (df["predicted_numeric"] == 1)).astype(int)
df["false_positive"] = ((df["target"] == 0) & (df["predicted_numeric"] == 1)).astype(int)
df["false_negative"] = ((df["target"] == 1) & (df["predicted_numeric"] == 0)).astype(int)

total = len(df)
correct = df["correct"].sum()
true_positive = df["true_positive"].sum()
false_positive = df["false_positive"].sum()
false_negative = df["false_negative"].sum()

accuracy = correct / total
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

metrics = pd.DataFrame([{
    "total": total,
    "correct": correct,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall
}])

metrics.to_csv(output_metrics_path, sep="\t", index=False)

incorrect_predictions = df[df["correct"] == 0]
incorrect_predictions.to_csv(output_incorrect_path, sep="\t", index=False)

print(f"Metrics saved to {output_metrics_path}")
print(f"Incorrect predictions saved to {output_incorrect_path}")


