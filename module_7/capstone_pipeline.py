import pandas as pd
from transformers import pipeline

# Load data (your NLP processed CSV from Module 4)
input_file = "../module_4/nlp_processed_data.csv"
df = pd.read_csv(input_file)

print("\n===== LOADED DATA =====")
print(df.head())

# Use Hugging Face Transformer model
sentiment_model = pipeline("sentiment-analysis")

# Apply advanced sentiment
results = sentiment_model(df["clean_text"].tolist())

df["transformer_label"] = [r["label"] for r in results]
df["transformer_score"] = [r["score"] for r in results]

print("\n===== ADVANCED SENTIMENT SAMPLE =====")
print(df[["clean_text", "transformer_label", "transformer_score"]].head())

# Save results
output_file = "../data/results/capstone_sentiment_output.csv"
df.to_csv(output_file, index=False)

print(f"\nSaved capstone results to: {output_file}")

