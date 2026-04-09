from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

analyzer = SentimentIntensityAnalyzer()

def add_sentiment():
    df = pd.read_csv("data/clean_text.csv")

    # Create sentiment score
    df["sentiment"] = df["text"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"]
    )

    df.to_csv("data/text_with_sentiment.csv", index=False)

    print("✅ Sentiment added!")

if __name__ == "__main__":
    add_sentiment()