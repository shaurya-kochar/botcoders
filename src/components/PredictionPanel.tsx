"use client";

import { TrendingUp, TrendingDown, Minus, AlertCircle } from "lucide-react";
import type { PredictionResult } from "@/types";

interface Props {
  data: PredictionResult;
}

type Movement = PredictionResult["predictedMovement"];

const MOVEMENT_CONFIG: Record<
  Movement,
  { icon: typeof TrendingUp; color: string; bg: string; label: string }
> = {
  Upward: {
    icon: TrendingUp,
    color: "text-green-400",
    bg: "bg-green-500/10 border-green-500/20",
    label: "Bullish — expected upward movement",
  },
  Downward: {
    icon: TrendingDown,
    color: "text-red-400",
    bg: "bg-red-500/10 border-red-500/20",
    label: "Bearish — expected downward movement",
  },
  Neutral: {
    icon: Minus,
    color: "text-yellow-400",
    bg: "bg-yellow-500/10 border-yellow-500/20",
    label: "Sideways — no strong directional signal",
  },
};

function MetricCard({
  label,
  value,
  unit,
  hint,
}: {
  label: string;
  value: string | number;
  unit?: string;
  hint?: string;
}) {
  return (
    <div className="bg-gray-800 rounded-lg p-3 flex flex-col gap-1">
      <p className="text-xs text-gray-500 flex items-center gap-1">
        {label}
        {hint && (
          <span title={hint} className="cursor-help">
            <AlertCircle size={11} className="text-gray-600" />
          </span>
        )}
      </p>
      <p className="text-lg font-bold text-white">
        {value}
        {unit && <span className="text-xs text-gray-400 ml-0.5">{unit}</span>}
      </p>
    </div>
  );
}

export default function PredictionPanel({ data }: Props) {
  const cfg = MOVEMENT_CONFIG[data.predictedMovement];
  const Icon = cfg.icon;
  const priceDiff = data.predictedPrice - data.currentPrice;
  const pricePct = ((priceDiff / data.currentPrice) * 100).toFixed(2);
  const confidencePct = (data.confidence * 100).toFixed(0);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-5 flex flex-col gap-5">
      <div>
        <h2 className="text-base font-semibold text-white">Prediction Panel</h2>
        <p className="text-xs text-gray-500 mt-0.5">
          LSTM model output for next trading session
        </p>
      </div>

      {/* Direction card */}
      <div className={`rounded-xl border p-4 flex items-center gap-4 ${cfg.bg}`}>
        <div className={`p-3 rounded-full bg-gray-900/50`}>
          <Icon size={28} className={cfg.color} />
        </div>
        <div>
          <p className={`text-xl font-bold ${cfg.color}`}>
            {data.predictedMovement}
          </p>
          <p className="text-xs text-gray-400 mt-0.5">{cfg.label}</p>
        </div>
        {/* Confidence ring */}
        <div className="ml-auto text-right">
          <p className={`text-3xl font-extrabold ${cfg.color}`}>
            {confidencePct}%
          </p>
          <p className="text-xs text-gray-500">confidence</p>
        </div>
      </div>

      {/* Price grid */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          label="Current Price"
          value={`$${data.currentPrice.toFixed(2)}`}
        />
        <MetricCard
          label="Predicted Price"
          value={`$${data.predictedPrice.toFixed(2)}`}
        />
        <MetricCard
          label="Expected Change"
          value={`${priceDiff >= 0 ? "+" : ""}${priceDiff.toFixed(2)}`}
          unit={` (${pricePct}%)`}
        />
        <MetricCard label="Ticker" value={data.ticker} />
      </div>

      {/* Evaluation metrics */}
      <div>
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
          Model Evaluation Metrics
        </p>
        <div className="grid grid-cols-3 gap-3">
          <MetricCard
            label="MAE"
            value={`$${data.mae}`}
            hint="Mean Absolute Error"
          />
          <MetricCard
            label="RMSE"
            value={`$${data.rmse}`}
            hint="Root Mean Square Error"
          />
          <MetricCard
            label="MAPE"
            value={`${data.mape}%`}
            hint="Mean Absolute Percentage Error"
          />
        </div>
      </div>

      <p className="text-[10px] text-gray-600 text-right">
        Generated: {new Date(data.generatedAt).toLocaleString()}
      </p>
    </div>
  );
}
