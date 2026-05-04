from datasets import load_dataset
import pandas as pd

dataset = load_dataset("glue", "mrpc")

pairs = []

for data in dataset["train"]:
    if data["label"] == 1:
        pairs.append({
            "input": "paraphrase: " + data["sentence1"],
            "output": data["sentence2"]
        })

df = pd.DataFrame(pairs)
df.to_csv("paraphrase_dataset.csv", index=False)

print("Dataset saved successfully")