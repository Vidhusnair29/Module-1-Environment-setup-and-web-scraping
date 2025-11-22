from transformers import pipeline

# Sample AI-related sentences for Module 6
texts = [
    "AI will change the future.",
    "Deep learning models are getting better every year.",
    "We must ensure safety and ethics in AI systems.",
    "I am not sure if AI will be beneficial to humanity.",
    "Transformers are incredibly powerful for NLP tasks."
]

# Load the HuggingFace sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis")

print("\n===== MODULE 6: TRANSFORMER SENTIMENT OUTPUT =====\n")

# Run sentiment on each text
for text in texts:
    result = sentiment_model(text)[0]
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}  (Score: {result['score']:.4f})\n")
