/**
 * Main Dashboard Page
 *
 * Currently uses mock data from src/data/mockData.ts.
 * To connect to a real backend, replace each getMock*() call with a fetch()
 * to the corresponding endpoint:
 *
 *   getMockSentimentData(ticker)  →  GET /get_sentiment?ticker=AAPL
 *   getMockSentimentSummary(...)  →  derived from sentiment or separate endpoint
 *   getMockMMIData(ticker)        →  GET /get_market_mood_index?ticker=AAPL
 *   getMockStockData(ticker)      →  GET /get_stock_data?ticker=AAPL
 *   getMockPrediction(ticker)     →  GET /get_prediction?ticker=AAPL
 */

"use client";

import { useState, useMemo, useCallback } from "react";
import { RefreshCw, TrendingUp, Activity, BarChart2, Brain } from "lucide-react";

import StockSelector    from "@/components/StockSelector";
import SentimentChart   from "@/components/SentimentChart";
import MMIChart         from "@/components/MMIChart";
import StockChart       from "@/components/StockChart";
import PredictionPanel  from "@/components/PredictionPanel";
import SentimentPieChart from "@/components/SentimentPieChart";

import {
  STOCK_OPTIONS,
  getMockSentimentData,
  getMockSentimentSummary,
  getMockMMIData,
  getMockStockData,
  getMockPrediction,
} from "@/data/mockData";

// ─── Small KPI stat card ──────────────────────────────────────────────────────

function StatCard({
  icon: Icon,
  label,
  value,
  delta,
  deltaPositive,
}: {
  icon: typeof TrendingUp;
  label: string;
  value: string;
  delta?: string;
  deltaPositive?: boolean;
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-center gap-4">
      <div className="p-2.5 rounded-lg bg-indigo-600/15 shrink-0">
        <Icon size={20} className="text-indigo-400" />
      </div>
      <div className="min-w-0">
        <p className="text-xs text-gray-500 truncate">{label}</p>
        <p className="text-lg font-bold text-white leading-tight">{value}</p>
        {delta && (
          <p className={`text-xs mt-0.5 font-medium ${deltaPositive ? "text-green-400" : "text-red-400"}`}>
            {delta}
          </p>
        )}
      </div>
    </div>
  );
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [ticker, setTicker] = useState<string>("AAPL");
  const [refreshKey, setRefreshKey] = useState(0);

  /* ── Data (swap these useMemo calls with async useEffect + fetch for real API) */
  const sentimentData    = useMemo(() => getMockSentimentData(ticker),    [ticker, refreshKey]);
  const sentimentSummary = useMemo(() => getMockSentimentSummary(ticker), [ticker, refreshKey]);
  const mmiData          = useMemo(() => getMockMMIData(ticker),          [ticker, refreshKey]);
  const stockData        = useMemo(() => getMockStockData(ticker),        [ticker, refreshKey]);
  const prediction       = useMemo(() => getMockPrediction(ticker),       [ticker, refreshKey]);

  const refresh = useCallback(() => setRefreshKey((k) => k + 1), []);

  /* ── KPI strip values */
  const latestSentiment = sentimentData[sentimentData.length - 1]?.sentimentScore ?? 0;
  const latestMMI       = mmiData[mmiData.length - 1]?.mmi ?? 50;
  const latestClose     = stockData[stockData.length - 1]?.close ?? 0;
  const prevClose       = stockData[stockData.length - 2]?.close ?? latestClose;
  const closeChange     = latestClose - prevClose;
  const closePct        = ((closeChange / prevClose) * 100).toFixed(2);

  return (
    <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">Market Sentiment Dashboard</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            NLP-driven sentiment analysis × stock price correlation
          </p>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <StockSelector options={STOCK_OPTIONS} selected={ticker} onChange={setTicker} />
          <button
            onClick={refresh}
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-white bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg px-3 py-2.5 transition-colors"
          >
            <RefreshCw size={13} /> Refresh
          </button>
        </div>
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Activity}
          label={`${ticker} Sentiment Score`}
          value={`${latestSentiment >= 0 ? "+" : ""}${latestSentiment.toFixed(3)}`}
          delta={latestSentiment > 0 ? "Positive bias" : latestSentiment < 0 ? "Negative bias" : "Neutral"}
          deltaPositive={latestSentiment >= 0}
        />
        <StatCard
          icon={Brain}
          label="Market Mood Index"
          value={`${latestMMI.toFixed(1)}`}
          delta={mmiData[mmiData.length - 1]?.label}
          deltaPositive={latestMMI >= 50}
        />
        <StatCard
          icon={BarChart2}
          label={`${ticker} Close Price`}
          value={`$${latestClose.toFixed(2)}`}
          delta={`${closeChange >= 0 ? "+" : ""}${closeChange.toFixed(2)} (${closePct}%)`}
          deltaPositive={closeChange >= 0}
        />
        <StatCard
          icon={TrendingUp}
          label="LSTM Prediction"
          value={prediction.predictedMovement}
          delta={`${(prediction.confidence * 100).toFixed(0)}% confidence`}
          deltaPositive={prediction.predictedMovement === "Upward"}
        />
      </div>

      {/* Row 1: Sentiment Timeline + MMI */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <SentimentChart data={sentimentData} />
        <MMIChart data={mmiData} />
      </div>

      {/* Row 2: Stock chart full width */}
      <StockChart data={stockData} ticker={ticker} />

      {/* Row 3: Prediction panel + Pie chart */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <PredictionPanel data={prediction} />
        <SentimentPieChart data={sentimentSummary} />
      </div>

      {/* Data sources */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
        <p className="text-xs font-medium text-gray-400 mb-2">Data Sources Considered in Analysis</p>
        <div className="flex flex-wrap gap-2">
          {["Twitter / X", "Reddit (r/stocks, r/investing)", "Instagram", "WhatsApp (consented)", "Financial Newsletters"].map(
            (src) => (
              <span key={src} className="text-xs px-2.5 py-1 rounded-full bg-gray-800 border border-gray-700 text-gray-400">
                {src}
              </span>
            )
          )}
        </div>
      </div>

      {/* Pipeline explanation */}
      <div id="about" className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h2 className="text-sm font-semibold text-white mb-3">System Pipeline</h2>
        <div className="flex flex-wrap items-center gap-2 text-xs text-gray-400">
          {[
            "Raw Social Text", "Preprocessing", "FinBERT Sentiment",
            "Feature Engineering", "Market Mood Index",
            "OHLCV Stock Data", "LSTM / Regression", "Dashboard Visualisation",
          ].map((step, i, arr) => (
            <span key={step} className="flex items-center gap-2">
              <span className="bg-gray-800 border border-gray-700 rounded-md px-2.5 py-1.5 font-medium text-gray-300">
                {step}
              </span>
              {i < arr.length - 1 && <span className="text-gray-600">→</span>}
            </span>
          ))}
        </div>
      </div>

      {/* Footer */}
      <footer className="text-center text-xs text-gray-600 pb-4">
        SentimentStock · iNLP University Project · {new Date().getFullYear()} ·
        All visualisations use mock data pending backend integration.
      </footer>

    </div>
  );
}
