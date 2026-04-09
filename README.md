This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.



## EDA & Feature Engineering 

Overview

This project predicts stock market closing prices using a combination of market data (OHLCV) and social sentiment from Twitter, Reddit, and financial news.

We combine quantitative signals (price, volume, indicators) with qualitative signals (public sentiment) to model market behavior.

# Data Sources
Stock Market Data
Source: Yahoo Finance (via yfinance)
Indices used:
NIFTY 50 (^NSEI)
S&P 500 (^GSPC)
Features:
Open, High, Low, Close, Volume (OHLCV)
 Sentiment Data (Kaggle)
Twitter dataset (stock tweets)
Reddit dataset (financial discussions)
Financial news dataset (headlines/articles)

All datasets are merged and cleaned into a unified format.

# Data Processing Pipeline
Text Cleaning
Standardized text, timestamps, and sources
File: clean_text.py
Sentiment Analysis
VADER sentiment scoring
Output: sentiment score per post
File: sentiment.py
Daily Aggregation (S_t)
Weighted sentiment using:
likes/upvotes
source credibility

Formula:
S_t = weighted average sentiment per day
Feature Engineering
Lagged sentiment: S_t-1 to S_t-6
Technical indicators:
RSI, MACD, Bollinger Bands
Moving averages
File: merge_sequence.py

#Time Alignment
All data is grouped into daily buckets
bucket column acts as the time index
# Model Input (Sliding Window)

We use a 6-day sliding window:

Input (X):
Past 6 days of:
OHLCV
Sentiment (S_t)
Technical indicators
Output (y):
Next day closing price
# Final Dataset

Stored as NumPy arrays:

X_nifty50.npy, y_nifty50.npy
X_sp500.npy, y_sp500.npy

Shape:

X → (samples, 6, features)
y → (samples,)

#Key Idea

The model learns patterns such as:

Rising sentiment → potential price increase
Sudden negative sentiment → possible drop
 
 #Visualizations: 
Generated plots include:

Price trends
Sentiment trends
Sentiment vs price
RSI analysis
Correlation heatmaps
