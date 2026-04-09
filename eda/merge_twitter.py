import pandas as pd

df1 = pd.read_csv(
    "data/raw/tweets_labelled_09042020_16072020.csv",
    engine="python",
    on_bad_lines="skip",
    encoding="latin1",
    header=None
)

df2 = pd.read_csv(
    "data/raw/tweets_remaining_09042020_16072020.csv",
    engine="python",
    on_bad_lines="skip",
    encoding="latin1",
    header=None
)

df = pd.concat([df1, df2], ignore_index=True)

# keep only non-empty text rows
df = df.dropna()

# keep only first column (actual tweet text)
df = df[[0]]
df.columns = ["text"]

df.to_csv("data/raw/kaggle_twitter.csv", index=False)

print("✅ Clean Twitter dataset ready!")