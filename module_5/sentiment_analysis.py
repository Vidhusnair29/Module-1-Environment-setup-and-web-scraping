import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download resources
nltk.download('vader_lexicon')

# Load dataset from Module 4
df = pd.read_csv("../module_4/nlp_processed_data.csv")

print("\n===== ORIGINAL CLEAN TEXT =====")
print(df['clean_text'].head())

# ============================================================
# 1️⃣ VADER SENTIMENT ANALYSIS
# ============================================================
vader = SentimentIntensityAnalyzer()

def vader_score(text):
    return vader.polarity_scores(text)["compound"]

df["vader_compound"] = df["clean_text"].apply(vader_score)


# ============================================================
# 2️⃣ TEXTBLOB SENTIMENT
# ============================================================
def textblob_polarity(text):
    return TextBlob(text).sentiment.polarity

df["textblob_polarity"] = df["clean_text"].apply(textblob_polarity)


# ============================================================
# 3️⃣ FINAL SENTIMENT LABEL (using VADER)
# ============================================================
def label_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment_label"] = df["vader_compound"].apply(label_sentiment)


print("\n===== SENTIMENT RESULTS SAMPLE =====")
print(df[["clean_text", "vader_compound", "textblob_polarity", "sentiment_label"]].head())


# ============================================================
# 4️⃣ VISUALIZATIONS
# ============================================================

# — Sentiment Label Count —
plt.figure(figsize=(6,4))
df["sentiment_label"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# — Distribution Plot —
plt.figure(figsize=(7,5))
sns.histplot(df["vader_compound"], bins=10, kde=True)
plt.title("VADER Sentiment Score Distribution")
plt.show()


# ============================================================
# 5️⃣ SAVE FINAL OUTPUT
# ============================================================
df.to_csv("sentiment_analysis_results.csv", index=False)
print("\nSaved sentiment results as: sentiment_analysis_results.csv")
