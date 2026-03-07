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

// ─── Stock Data ───────────────────────────────────────────────────────────────

export interface StockDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sentimentOverlay?: number; // optional overlay from -1 to 1
}

// ─── Prediction ───────────────────────────────────────────────────────────────

export interface PredictionResult {
  ticker: string;
  predictedMovement: "Upward" | "Downward" | "Neutral";
  confidence: number;       // 0.0 to 1.0
  predictedPrice: number;
  currentPrice: number;
  mae: number;
  rmse: number;
  mape: number;
  generatedAt: string;      // ISO timestamp
}

// ─── Stock Options ────────────────────────────────────────────────────────────

export interface StockOption {
  ticker: string;
  name: string;
}

// ─── API Response Wrappers (matches backend format) ──────────────────────────

export interface ApiResponse<T> {
  status: "success" | "error";
  data: T;
  message?: string;
  timestamp: string;
}
