"""
Evaluation Script — Smart Paraphrasing Tool

Runs the full pipeline on the test split of the dataset and computes:
  - BLEU  (sacrebleu)
  - ROUGE-1, ROUGE-2, ROUGE-L (rouge-score)
"""

import pandas as pd
from tqdm import tqdm
import evaluate as hf_eval

from pipeline import paraphrase_paragraph, preprocess_text

# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_PATH  = "paraphrase_dataset.csv"
EVAL_FRACTION = 0.10   # use 10% of data for evaluation
MAX_SAMPLES   = 100    # cap for speed

# ─── Load Data ────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Use last 10% as test split (same split as training)
split_idx = int(len(df) * 0.9)
test_df   = df.iloc[split_idx:].reset_index(drop=True)

# Cap samples
if len(test_df) > MAX_SAMPLES:
    test_df = test_df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

print(f"Evaluating on {len(test_df)} samples...")

# ─── Run Pipeline ─────────────────────────────────────────────────────────────
predictions = []
references  = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating"):
    # Strip the 'paraphrase: ' prefix if present
    input_text = row["input"]
    if input_text.startswith("paraphrase:"):
        input_text = input_text[len("paraphrase:"):].strip()

    result = paraphrase_paragraph(input_text, num_beams=4, num_return_sequences=4)
    predictions.append(result["paraphrased_text"])
    references.append(row["output"])

# ─── Compute Metrics ──────────────────────────────────────────────────────────
print("\nComputing metrics...")

rouge  = hf_eval.load("rouge")
sacrebleu = hf_eval.load("sacrebleu")

rouge_result = rouge.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True
)
bleu_result = sacrebleu.compute(
    predictions=predictions,
    references=[[r] for r in references]
)

# ─── Print Results ────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  EVALUATION RESULTS")
print("=" * 50)
print(f"  BLEU     : {bleu_result['score']:.2f}")
print(f"  ROUGE-1  : {rouge_result['rouge1']:.4f}")
print(f"  ROUGE-2  : {rouge_result['rouge2']:.4f}")
print(f"  ROUGE-L  : {rouge_result['rougeL']:.4f}")
print("=" * 50)

# ─── Sample Outputs ───────────────────────────────────────────────────────────
print("\nSample Predictions:")
for i in range(min(5, len(predictions))):
    print(f"\n[{i+1}] Original  : {references[i]}")
    print(f"    Predicted : {predictions[i]}")
