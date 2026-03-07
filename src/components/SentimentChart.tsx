"use client";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend,
} from "recharts";
import type { SentimentDataPoint } from "@/types";

interface Props {
  data: SentimentDataPoint[];
}

// ─── Custom Tooltip ───────────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const score: number = payload[0]?.value ?? 0;
  const color = score > 0 ? "#22c55e" : score < 0 ? "#ef4444" : "#94a3b8";
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-gray-400 mb-1">{label}</p>
      <p className="font-semibold" style={{ color }}>
        Sentiment: {score >= 0 ? "+" : ""}
        {score.toFixed(4)}
      </p>
      <p className="text-gray-400 mt-0.5">
        Source: {payload[0]?.payload?.source}
      </p>
    </div>
  );
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function SentimentChart({ data }: Props) {
  // Show every 7th date label to avoid crowding
  const ticks = data
    .filter((_, i) => i % 7 === 0)
    .map((d) => d.date);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
      <div className="mb-4">
        <h2 className="text-base font-semibold text-white">
          Sentiment Timeline
        </h2>
        <p className="text-xs text-gray-500 mt-0.5">
          Daily aggregated sentiment score (−1 negative → +1 positive)
        </p>
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: -10, bottom: 0 }}>
          <defs>
            <linearGradient id="sentGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="date"
            ticks={ticks}
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#374151" }}
            tickFormatter={(v) => v.slice(5)} // "MM-DD"
          />
          <YAxis
            domain={[-1, 1]}
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#374151" }}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0} stroke="#4b5563" strokeDasharray="4 4" />
          <Line
            type="monotone"
            dataKey="sentimentScore"
            stroke="#6366f1"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "#6366f1" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
