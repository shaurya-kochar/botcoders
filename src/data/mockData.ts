/**
 * Mock data that mirrors the exact shape of backend API responses.
 * Replace each `getMock*` call with a real fetch() to the corresponding endpoint:
 *
 *   /get_sentiment   → getSentimentData()
 *   /get_market_mood_index → getMMIData()
 *   /get_stock_data  → getStockData()
 *   /get_prediction  → getPrediction()
 */

import type {
  SentimentDataPoint,
  SentimentSummary,
  MMIDataPoint,
  StockDataPoint,
  PredictionResult,
  StockOption,
} from "@/types";

// ─── Stock List ───────────────────────────────────────────────────────────────

export const STOCK_OPTIONS: StockOption[] = [
  { ticker: "AAPL", name: "Apple Inc." },
  { ticker: "TSLA", name: "Tesla Inc." },
  { ticker: "NVDA", name: "NVIDIA Corporation" },
  { ticker: "AMZN", name: "Amazon.com Inc." },
  { ticker: "MSFT", name: "Microsoft Corporation" },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────

function dateRange(startDate: string, days: number): string[] {
  const dates: string[] = [];
  const start = new Date(startDate);
  for (let i = 0; i < days; i++) {
    const d = new Date(start);
    d.setDate(start.getDate() + i);
    dates.push(d.toISOString().split("T")[0]);
  }
  return dates;
}

function jitter(base: number, range: number): number {
  return parseFloat((base + (Math.random() - 0.5) * range).toFixed(4));
}

// ─── Sentiment Data ───────────────────────────────────────────────────────────

const SENTIMENT_SEEDS: Record<string, { score: number; base: number }> = {
  AAPL: { score: 0.35, base: 175 },
  TSLA: { score: -0.1, base: 245 },
  NVDA: { score: 0.55, base: 820 },
  AMZN: { score: 0.2, base: 182 },
  MSFT: { score: 0.4, base: 415 },
};

export function getMockSentimentData(ticker: string): SentimentDataPoint[] {
  const seed = SENTIMENT_SEEDS[ticker] ?? { score: 0.1, base: 100 };
  const dates = dateRange("2025-01-01", 60);

  let score = seed.score;
  return dates.map((date, i) => {
    score = Math.max(-1, Math.min(1, score + (Math.random() - 0.5) * 0.12));
    const pos = parseFloat(((score + 1) / 2 * 0.7 + Math.random() * 0.1).toFixed(3));
    const neg = parseFloat(((1 - (score + 1) / 2) * 0.6 + Math.random() * 0.1).toFixed(3));
    const neu = parseFloat((1 - pos - neg).toFixed(3));
    const sources = ["twitter", "reddit", "instagram", "newsletter"];
    return {
      date,
      sentimentScore: parseFloat(score.toFixed(4)),
      positive: Math.min(pos, 0.95),
      negative: Math.min(neg, 0.95),
      neutral: Math.max(neu, 0.02),
      source: sources[i % sources.length],
    };
  });
}

export function getMockSentimentSummary(ticker: string): SentimentSummary {
  const data = getMockSentimentData(ticker);
  const avg = (key: keyof SentimentDataPoint) =>
    parseFloat(
      (
        (data.reduce((s, d) => s + (d[key] as number), 0) / data.length) *
        100
      ).toFixed(1)
    );
  return {
    positive: avg("positive"),
    negative: avg("negative"),
    neutral: avg("neutral"),
  };
}

// ─── Market Mood Index ────────────────────────────────────────────────────────

function mmiLabel(val: number): string {
  if (val < 20) return "Extreme Fear";
  if (val < 40) return "Fear";
  if (val < 60) return "Neutral";
  if (val < 80) return "Greed";
  return "Extreme Greed";
}

export function getMockMMIData(ticker: string): MMIDataPoint[] {
  const seed = SENTIMENT_SEEDS[ticker] ?? { score: 0.1, base: 100 };
  const dates = dateRange("2025-01-01", 60);
  let mmi = 50 + seed.score * 30;
  return dates.map((date) => {
    mmi = Math.max(5, Math.min(95, mmi + (Math.random() - 0.5) * 8));
    const val = parseFloat(mmi.toFixed(1));
    return { date, mmi: val, label: mmiLabel(val) };
  });
}

// ─── Stock Data ───────────────────────────────────────────────────────────────

export function getMockStockData(ticker: string): StockDataPoint[] {
  const seed = SENTIMENT_SEEDS[ticker] ?? { score: 0.1, base: 100 };
  const dates = dateRange("2025-01-01", 60);
  const sentiment = getMockSentimentData(ticker);

  let close = seed.base;
  return dates.map((date, i) => {
    const dailyChange = (Math.random() - 0.48) * close * 0.025;
    close = parseFloat((close + dailyChange).toFixed(2));
    const open = parseFloat((close - jitter(0, close * 0.01)).toFixed(2));
    const high = parseFloat((Math.max(open, close) + jitter(0, close * 0.008)).toFixed(2));
    const low = parseFloat((Math.min(open, close) - jitter(0, close * 0.008)).toFixed(2));
    const volume = Math.floor(Math.random() * 50_000_000 + 10_000_000);
    return {
      date,
      open,
      high,
      low,
      close,
      volume,
      sentimentOverlay: sentiment[i].sentimentScore,
    };
  });
}

// ─── Prediction ───────────────────────────────────────────────────────────────

export function getMockPrediction(ticker: string): PredictionResult {
  const stock = getMockStockData(ticker);
  const last = stock[stock.length - 1];
  const seed = SENTIMENT_SEEDS[ticker] ?? { score: 0, base: 100 };

  const confidence = parseFloat((0.55 + Math.abs(seed.score) * 0.35).toFixed(2));
  const direction = seed.score > 0.05 ? "Upward" : seed.score < -0.05 ? "Downward" : "Neutral";
  const delta = direction === "Upward" ? 1.03 : direction === "Downward" ? 0.97 : 1.0;

  return {
    ticker,
    predictedMovement: direction,
    confidence,
    predictedPrice: parseFloat((last.close * delta).toFixed(2)),
    currentPrice: last.close,
    mae: parseFloat((last.close * 0.012).toFixed(2)),
    rmse: parseFloat((last.close * 0.018).toFixed(2)),
    mape: parseFloat((2.1 + Math.random() * 1.5).toFixed(2)),
    generatedAt: new Date().toISOString(),
  };
}
