/**
 * Mock data — mirrors exact backend API response shapes.
 *
 * When connecting to the real backend, replace each getMock* call with fetch():
 *   getMockSentimentData(symbol)  →  GET /get_sentiment?asset=AAPL&type=stock
 *   getMockMMIData(symbol)        →  GET /get_market_mood_index?asset=NIFTY50&type=index
 *   getMockPriceData(symbol)      →  GET /get_stock_data | /get_market_index_data
 *   getMockPrediction(symbol)     →  GET /get_prediction?asset=AAPL&type=stock
 *
 * All generators are keyed by `symbol` (e.g. "AAPL", "NIFTY50") and work
 * identically for both stocks and indices — the page decides which list to show.
 */

import type {
  AssetOption,
  AssetType,
  SentimentDataPoint,
  SentimentSummary,
  MMIDataPoint,
  PriceDataPoint,
  PredictionResult,
  AssetPayload,
} from "@/types";

// ─── Asset Lists ──────────────────────────────────────────────────────────────

export const STOCK_OPTIONS: AssetOption[] = [
  { symbol: "AAPL",  name: "Apple Inc.",          assetType: "stock", region: "US" },
  { symbol: "TSLA",  name: "Tesla Inc.",           assetType: "stock", region: "US" },
  { symbol: "NVDA",  name: "NVIDIA Corporation",   assetType: "stock", region: "US" },
  { symbol: "AMZN",  name: "Amazon.com Inc.",      assetType: "stock", region: "US" },
  { symbol: "MSFT",  name: "Microsoft Corporation",assetType: "stock", region: "US" },
];

export const INDEX_OPTIONS: AssetOption[] = [
  { symbol: "NIFTY50", name: "NIFTY 50",   assetType: "index", region: "IN" },
  { symbol: "SENSEX",  name: "SENSEX",     assetType: "index", region: "IN" },
  { symbol: "SP500",   name: "S&P 500",    assetType: "index", region: "US" },
  { symbol: "NASDAQ",  name: "NASDAQ Composite", assetType: "index", region: "US" },
];

/** Returns the correct options list for the given asset type */
export function getOptions(assetType: AssetType): AssetOption[] {
  return assetType === "stock" ? STOCK_OPTIONS : INDEX_OPTIONS;
}

/** Returns the default symbol for each asset type */
export function getDefaultSymbol(assetType: AssetType): string {
  return assetType === "stock" ? "AAPL" : "NIFTY50";
}

// ─── Seed data ────────────────────────────────────────────────────────────────
// base = realistic starting price; score = sentiment bias (-1 to 1)

interface Seed { score: number; base: number; volatility: number }

const SEEDS: Record<string, Seed> = {
  // Stocks
  AAPL:  { score:  0.35, base:   175,    volatility: 0.020 },
  TSLA:  { score: -0.10, base:   245,    volatility: 0.035 },
  NVDA:  { score:  0.55, base:   820,    volatility: 0.030 },
  AMZN:  { score:  0.20, base:   182,    volatility: 0.022 },
  MSFT:  { score:  0.40, base:   415,    volatility: 0.018 },
  // Indices
  NIFTY50: { score:  0.25, base: 22_400,  volatility: 0.012 },
  SENSEX:  { score:  0.22, base: 73_800,  volatility: 0.012 },
  SP500:   { score:  0.30, base:  4_950,  volatility: 0.013 },
  NASDAQ:  { score:  0.28, base: 15_400,  volatility: 0.016 },
};

const DEFAULT_SEED: Seed = { score: 0.1, base: 100, volatility: 0.02 };

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

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}

function fmt(v: number, decimals = 4) {
  return parseFloat(v.toFixed(decimals));
}

// ─── Sentiment ────────────────────────────────────────────────────────────────

export function getMockSentimentData(symbol: string): SentimentDataPoint[] {
  const seed = SEEDS[symbol] ?? DEFAULT_SEED;
  const dates = dateRange("2025-01-01", 60);
  const sources = ["twitter", "reddit", "instagram", "newsletter"];

  let score = seed.score;
  return dates.map((date, i) => {
    score = clamp(score + (Math.random() - 0.5) * 0.12, -1, 1);
    const pos = fmt(clamp((score + 1) / 2 * 0.7 + Math.random() * 0.1, 0.02, 0.95), 3);
    const neg = fmt(clamp((1 - (score + 1) / 2) * 0.6 + Math.random() * 0.1, 0.02, 0.95), 3);
    const neu = fmt(clamp(1 - pos - neg, 0.02, 0.95), 3);
    return {
      date,
      sentimentScore: fmt(score),
      positive: pos,
      negative: neg,
      neutral: neu,
      source: sources[i % sources.length],
    };
  });
}

export function getMockSentimentSummary(symbol: string): SentimentSummary {
  const data = getMockSentimentData(symbol);
  const len = data.length;
  const avg = (key: keyof SentimentDataPoint) =>
    fmt((data.reduce((s, d) => s + (d[key] as number), 0) / len) * 100, 1);
  return { positive: avg("positive"), negative: avg("negative"), neutral: avg("neutral") };
}

// ─── Market Mood Index ────────────────────────────────────────────────────────

function mmiLabel(v: number): string {
  if (v < 20) return "Extreme Fear";
  if (v < 40) return "Fear";
  if (v < 60) return "Neutral";
  if (v < 80) return "Greed";
  return "Extreme Greed";
}

export function getMockMMIData(symbol: string): MMIDataPoint[] {
  const seed = SEEDS[symbol] ?? DEFAULT_SEED;
  const dates = dateRange("2025-01-01", 60);
  let mmi = clamp(50 + seed.score * 30, 5, 95);
  return dates.map((date) => {
    mmi = clamp(mmi + (Math.random() - 0.5) * 8, 5, 95);
    const val = fmt(mmi, 1);
    return { date, mmi: val, label: mmiLabel(val) };
  });
}

// ─── Price Data (works for both stocks and indices) ───────────────────────────

export function getMockPriceData(symbol: string): PriceDataPoint[] {
  const seed = SEEDS[symbol] ?? DEFAULT_SEED;
  const dates = dateRange("2025-01-01", 60);
  const sentiment = getMockSentimentData(symbol);

  // Indices have higher nominal volumes (turnover in crores / billions)
  const isIndex = symbol in { NIFTY50: 1, SENSEX: 1, SP500: 1, NASDAQ: 1 };
  const volBase  = isIndex ? 5_000_000_000 : 10_000_000;
  const volRange = isIndex ? 3_000_000_000 : 40_000_000;

  let close = seed.base;
  return dates.map((date, i) => {
    const change = (Math.random() - 0.48) * close * seed.volatility;
    close = fmt(close + change, 2);
    const open  = fmt(close - (Math.random() - 0.5) * close * (seed.volatility * 0.4), 2);
    const high  = fmt(Math.max(open, close) + Math.random() * close * (seed.volatility * 0.3), 2);
    const low   = fmt(Math.min(open, close) - Math.random() * close * (seed.volatility * 0.3), 2);
    const volume = Math.floor(Math.random() * volRange + volBase);
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

export function getMockPrediction(symbol: string, assetType: AssetType): PredictionResult {
  const prices = getMockPriceData(symbol);
  const last   = prices[prices.length - 1];
  const seed   = SEEDS[symbol] ?? DEFAULT_SEED;

  const confidence = fmt(clamp(0.55 + Math.abs(seed.score) * 0.35, 0, 1), 2);
  const direction: PredictionResult["predictedMovement"] =
    seed.score > 0.05 ? "Upward" : seed.score < -0.05 ? "Downward" : "Neutral";
  const delta = direction === "Upward" ? 1.03 : direction === "Downward" ? 0.97 : 1.0;

  return {
    ticker: symbol,
    assetType,
    predictedMovement: direction,
    confidence,
    predictedPrice: fmt(last.close * delta, 2),
    currentPrice: last.close,
    mae:  fmt(last.close * 0.012, 2),
    rmse: fmt(last.close * 0.018, 2),
    mape: fmt(2.1 + Math.random() * 1.5, 2),
    generatedAt: new Date().toISOString(),
  };
}

// ─── Unified asset payload (matches structured backend format) ────────────────

export function getMockAssetPayload(symbol: string, assetType: AssetType): AssetPayload {
  const sentimentData = getMockSentimentData(symbol);
  const mmiData       = getMockMMIData(symbol);
  const priceData     = getMockPriceData(symbol);
  const pred          = getMockPrediction(symbol, assetType);

  return {
    assetType,
    asset: symbol,
    data: sentimentData.map((s, i) => ({
      date:       s.date,
      sentiment:  s.sentimentScore,
      mmi:        mmiData[i]?.mmi ?? 50,
      price:      priceData[i]?.close ?? 0,
      prediction: pred.predictedMovement.toLowerCase() as "upward" | "downward" | "neutral",
    })),
  };
}

// ─── Back-compat aliases (used in existing chart components) ──────────────────
/** @deprecated Use getMockPriceData instead */
export const getMockStockData = getMockPriceData;
export type { PriceDataPoint as StockDataPoint };
