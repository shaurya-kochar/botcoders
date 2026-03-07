/**
 * Main Dashboard Page
 *
 * Asset type toggle (Stock | Market Index) drives which selector and data is shown.
 *
 * To connect to the real backend, replace each getMock* call with fetch():
 *   GET /get_sentiment?asset=AAPL&type=stock
 *   GET /get_market_mood_index?asset=NIFTY50&type=index
 *   GET /get_stock_data?asset=AAPL        (stocks)
 *   GET /get_market_index_data?asset=NIFTY50  (indices)
 *   GET /get_prediction?asset=AAPL&type=stock
 */

"use client";

import { useState, useMemo, useCallback } from "react";
import { RefreshCw, TrendingUp, Activity, BarChart2, Brain } from "lucide-react";
import type { AssetType } from "@/types";

import AssetTypeSelector from "@/components/AssetTypeSelector";
import AssetSelector     from "@/components/AssetSelector";
import SentimentChart    from "@/components/SentimentChart";
import MMIChart          from "@/components/MMIChart";
import StockChart        from "@/components/StockChart";
import PredictionPanel   from "@/components/PredictionPanel";
import SentimentPieChart from "@/components/SentimentPieChart";

import {
  getOptions,
  getDefaultSymbol,
  getMockSentimentData,
  getMockSentimentSummary,
  getMockMMIData,
  getMockPriceData,
  getMockPrediction,
} from "@/data/mockData";

// ─── Small KPI card ───────────────────────────────────────────────────────────

function StatCard({
  icon: Icon,
  label,
  value,
  delta,
  deltaPositive,
  accent = "indigo",
}: {
  icon: typeof TrendingUp;
  label: string;
  value: string;
  delta?: string;
  deltaPositive?: boolean;
  accent?: "indigo" | "amber";
}) {
  const iconBg  = accent === "amber" ? "bg-amber-500/15" : "bg-indigo-600/15";
  const iconCol = accent === "amber" ? "text-amber-400"  : "text-indigo-400";
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-center gap-4">
      <div className={`p-2.5 rounded-lg ${iconBg} shrink-0`}>
        <Icon size={20} className={iconCol} />
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
  // ── Asset type toggle (Stock ↔ Index)
  const [assetType, setAssetType] = useState<AssetType>("stock");

  // ── Selected symbol — auto-resets when assetType changes
  const [symbol, setSymbol] = useState<string>(getDefaultSymbol("stock"));

  // ── Refresh counter forces useMemo to re-run (simulates re-fetch)
  const [refreshKey, setRefreshKey] = useState(0);

  // Available options for the current asset type
  const options = useMemo(() => getOptions(assetType), [assetType]);

  // ── When the user switches asset type, pick the default for that type
  const handleAssetTypeChange = useCallback((newType: AssetType) => {
    setAssetType(newType);
    setSymbol(getDefaultSymbol(newType));
  }, []);

  // ── Derived data
  // Replace these with fetch() calls when the backend is ready.
  const sentimentData    = useMemo(() => getMockSentimentData(symbol),    [symbol, refreshKey]);
  const sentimentSummary = useMemo(() => getMockSentimentSummary(symbol), [symbol, refreshKey]);
  const mmiData          = useMemo(() => getMockMMIData(symbol),          [symbol, refreshKey]);
  const priceData        = useMemo(() => getMockPriceData(symbol),        [symbol, refreshKey]);
  const prediction       = useMemo(
    () => getMockPrediction(symbol, assetType),
    [symbol, assetType, refreshKey]
  );

  const refresh = useCallback(() => setRefreshKey((k) => k + 1), []);

  // ── KPI strip values
  const latestSentiment = sentimentData[sentimentData.length - 1]?.sentimentScore ?? 0;
  const latestMMI       = mmiData[mmiData.length - 1]?.mmi ?? 50;
  const latestClose     = priceData[priceData.length - 1]?.close ?? 0;
  const prevClose       = priceData[priceData.length - 2]?.close ?? latestClose;
  const closeChange     = latestClose - prevClose;
  const closePct        = prevClose > 0 ? ((closeChange / prevClose) * 100).toFixed(2) : "0.00";

  // Label copy changes depending on asset type
  const priceLabel      = assetType === "index" ? "Index Level" : "Close Price";
  const priceValue      = assetType === "index"
    ? latestClose.toLocaleString("en-IN")  // NIFTY / SENSEX use lakh-crore format
    : `$${latestClose.toFixed(2)}`;

  return (
    <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

      {/* ── Page header ───────────────────────────────────────────────────── */}
      <div className="flex flex-col gap-4">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
          <div>
            <h1 className="text-xl font-bold text-white">Market Sentiment Dashboard</h1>
            <p className="text-sm text-gray-500 mt-0.5">
              NLP-driven sentiment analysis ×{" "}
              {assetType === "stock" ? "individual stock" : "market index"} price correlation
            </p>
          </div>
          <button
            onClick={refresh}
            className="self-start flex items-center gap-1.5 text-xs text-gray-400 hover:text-white bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg px-3 py-2.5 transition-colors"
          >
            <RefreshCw size={13} /> Refresh data
          </button>
        </div>

        {/* ── Asset type + asset selector in one row ── */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 flex-wrap">
          {/* Step 1: choose the asset type */}
          <AssetTypeSelector value={assetType} onChange={handleAssetTypeChange} />

          {/* Visual divider */}
          <span className="hidden sm:block w-px h-8 bg-gray-700" />

          {/* Step 2: choose specific stock or index */}
          <AssetSelector
            options={options}
            selected={symbol}
            assetType={assetType}
            onChange={setSymbol}
          />
        </div>
      </div>

      {/* ── Asset type context banner ─────────────────────────────────────── */}
      <div
        className={`
          rounded-xl border px-4 py-3 flex items-center gap-3 text-sm
          ${assetType === "index"
            ? "bg-amber-500/5 border-amber-500/20"
            : "bg-indigo-500/5 border-indigo-500/20"}
        `}
      >
        <span className="text-lg">
          {assetType === "index" ? "📊" : "📈"}
        </span>
        <span className={assetType === "index" ? "text-amber-300" : "text-indigo-300"}>
          <strong>{options.find((o) => o.symbol === symbol)?.name ?? symbol}</strong>
          {assetType === "index"
            ? " — Market Index · Sentiment aggregated from all tracked platforms"
            : " — Individual Stock · Sentiment from Twitter, Reddit, Instagram, newsletters"}
        </span>
      </div>

      {/* ── KPI Strip ─────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Activity}
          label={`${symbol} Sentiment Score`}
          value={`${latestSentiment >= 0 ? "+" : ""}${latestSentiment.toFixed(3)}`}
          delta={latestSentiment > 0 ? "Positive bias" : latestSentiment < 0 ? "Negative bias" : "Neutral"}
          deltaPositive={latestSentiment >= 0}
          accent={assetType === "index" ? "amber" : "indigo"}
        />
        <StatCard
          icon={Brain}
          label="Market Mood Index"
          value={latestMMI.toFixed(1)}
          delta={mmiData[mmiData.length - 1]?.label}
          deltaPositive={latestMMI >= 50}
          accent={assetType === "index" ? "amber" : "indigo"}
        />
        <StatCard
          icon={BarChart2}
          label={`${symbol} ${priceLabel}`}
          value={priceValue}
          delta={`${closeChange >= 0 ? "+" : ""}${
            assetType === "index"
              ? closeChange.toFixed(0)
              : closeChange.toFixed(2)
          } (${closePct}%)`}
          deltaPositive={closeChange >= 0}
          accent={assetType === "index" ? "amber" : "indigo"}
        />
        <StatCard
          icon={TrendingUp}
          label="LSTM Prediction"
          value={prediction.predictedMovement}
          delta={`${(prediction.confidence * 100).toFixed(0)}% confidence`}
          deltaPositive={prediction.predictedMovement === "Upward"}
          accent={assetType === "index" ? "amber" : "indigo"}
        />
      </div>

      {/* ── Row 1: Sentiment Timeline + MMI ──────────────────────────────── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <SentimentChart data={sentimentData} />
        <MMIChart data={mmiData} />
      </div>

      {/* ── Row 2: Price chart (full width) ──────────────────────────────── */}
      <StockChart data={priceData} ticker={symbol} assetType={assetType} />

      {/* ── Row 3: Prediction + Pie ───────────────────────────────────────── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <PredictionPanel data={prediction} />
        <SentimentPieChart data={sentimentSummary} />
      </div>

      {/* ── Data sources ─────────────────────────────────────────────────── */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
        <p className="text-xs font-medium text-gray-400 mb-2">
          Data Sources Considered in Analysis
        </p>
        <div className="flex flex-wrap gap-2">
          {[
            "Twitter / X",
            "Reddit (r/stocks, r/investing, r/IndiaInvestments)",
            "Instagram",
            "WhatsApp (consented)",
            "Financial Newsletters",
          ].map((src) => (
            <span
              key={src}
              className="text-xs px-2.5 py-1 rounded-full bg-gray-800 border border-gray-700 text-gray-400"
            >
              {src}
            </span>
          ))}
        </div>
      </div>

      {/* ── Pipeline ─────────────────────────────────────────────────────── */}
      <div id="about" className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h2 className="text-sm font-semibold text-white mb-3">System Pipeline</h2>
        <div className="flex flex-wrap items-center gap-2 text-xs text-gray-400">
          {[
            "Raw Social Text",
            "Preprocessing",
            "FinBERT Sentiment",
            "Feature Engineering",
            "Market Mood Index",
            assetType === "index" ? "Index OHLCV Data" : "Stock OHLCV Data",
            "LSTM / Regression",
            "Dashboard Visualisation",
          ].map((step, i, arr) => (
            <span key={step} className="flex items-center gap-2">
              <span className="bg-gray-800 border border-gray-700 rounded-md px-2.5 py-1.5 font-medium text-gray-300">
                {step}
              </span>
              {i < arr.length - 1 && (
                <span className="text-gray-600">→</span>
              )}
            </span>
          ))}
        </div>
      </div>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="text-center text-xs text-gray-600 pb-4">
        SentimentStock · iNLP University Project · {new Date().getFullYear()} ·
        All visualisations use mock data pending backend integration.
      </footer>
    </div>
  );
}
