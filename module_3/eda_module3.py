import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("social_media_dataset.csv")

print("===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA INFO =====")
print(df.info())

print("\n===== SUMMARY STATISTICS =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Likes distribution
plt.figure(figsize=(8, 5))
plt.hist(df["likes"], bins=10)
plt.title("Likes Distribution")
plt.xlabel("Likes")
plt.ylabel("Frequency")
plt.show()

# Retweets vs Likes
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="likes", y="retweets")
plt.title("Likes vs Retweets")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df[['likes','retweets']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,5))
df['username'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Tweets per User")
plt.xlabel("User")
plt.ylabel("Tweet Count")
plt.show()

from wordcloud import WordCloud

text = " ".join(df["tweet"].astype(str))
wc = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("Tweet WordCloud")
plt.show()

df.to_csv("processed_social_media.csv", index=False)
print("Cleaned dataset saved as: processed_social_media.csv")


