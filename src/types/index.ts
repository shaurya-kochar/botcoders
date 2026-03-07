// ─── Asset Type ───────────────────────────────────────────────────────────────

/** Top-level switch: are we looking at a single stock or a market index? */
export type AssetType = "stock" | "index";

// ─── Sentiment ───────────────────────────────────────────────────────────────

export interface SentimentDataPoint {
  date: string;           // "YYYY-MM-DD"
  sentimentScore: number; // -1.0 to 1.0
  positive: number;       // 0.0 to 1.0
  negative: number;       // 0.0 to 1.0
  neutral: number;        // 0.0 to 1.0
  source: string;         // "twitter" | "reddit" | "instagram" | "newsletter"
}

export interface SentimentSummary {
  positive: number; // percentage
  negative: number;
  neutral: number;
}

// ─── Market Mood Index ────────────────────────────────────────────────────────

export interface MMIDataPoint {
  date: string;
  mmi: number; // 0 to 100 (like Fear & Greed index)
  label: string; // "Extreme Fear" | "Fear" | "Neutral" | "Greed" | "Extreme Greed"
}

// ─── Price Data (covers both stocks and indices) ──────────────────────────────

export interface PriceDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;            // for indices this is total traded volume / turnover
  sentimentOverlay?: number; // optional overlay from -1 to 1
}

// ─── Prediction ───────────────────────────────────────────────────────────────

export interface PredictionResult {
  /** Symbol — stock ticker (AAPL) or index symbol (NIFTY50) */
  ticker: string;
  assetType: AssetType;
  predictedMovement: "Upward" | "Downward" | "Neutral";
  confidence: number;    // 0.0 to 1.0
  predictedPrice: number;
  currentPrice: number;
  mae: number;
  rmse: number;
  mape: number;
  generatedAt: string;   // ISO timestamp
}

// ─── Asset Option (used by both StockSelector and IndexSelector) ──────────────

export interface AssetOption {
  /** machine-readable symbol, e.g. "AAPL" or "NIFTY50" */
  symbol: string;
  /** human-readable label, e.g. "Apple Inc." or "NIFTY 50" */
  name: string;
  /** which category this option belongs to */
  assetType: AssetType;
  /** optional flag/region tag shown in the selector */
  region?: string;
}

// ─── API Response Wrappers (matches backend format) ──────────────────────────

export interface ApiResponse<T> {
  status: "success" | "error";
  data: T;
  message?: string;
  timestamp: string;
}

/**
 * Unified asset payload — matches the backend response shape for both
 * /get_stock_data and /get_market_index_data
 *
 * {
 *   assetType: "stock",
 *   asset: "AAPL",
 *   data: [ { date, sentiment, mmi, price, prediction }, ... ]
 * }
 */
export interface AssetPayload {
  assetType: AssetType;
  asset: string;
  data: Array<{
    date: string;
    sentiment: number;
    mmi: number;
    price: number;
    prediction: "upward" | "downward" | "neutral";
  }>;
}
