"use client";

import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { SentimentSummary } from "@/types";

interface Props {
  data: SentimentSummary;
}

const SLICES = [
  { key: "positive" as const, label: "Positive", color: "#22c55e" },
  { key: "negative" as const, label: "Negative", color: "#ef4444" },
  { key: "neutral"  as const, label: "Neutral",  color: "#94a3b8" },
];

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const { name, value } = payload[0];
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="font-semibold text-white">
        {name}: {value.toFixed(1)}%
      </p>
    </div>
  );
}

function renderCustomLabel({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) {
  if (percent < 0.05) return null;
  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.55;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);
  return (
    <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={12} fontWeight={600}>
      {`${(percent * 100).toFixed(1)}%`}
    </text>
  );
}

export default function SentimentPieChart({ data }: Props) {
  const chartData = SLICES.map((s) => ({
    name: s.label,
    value: data[s.key],
    color: s.color,
  }));

  const dominant = chartData.reduce((a, b) => (a.value > b.value ? a : b));

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
      <div className="mb-4">
        <h2 className="text-base font-semibold text-white">
          Sentiment Distribution
        </h2>
        <p className="text-xs text-gray-500 mt-0.5">
          Breakdown of positive / negative / neutral posts
        </p>
      </div>

      <div className="flex flex-col items-center">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              innerRadius={55}
              outerRadius={90}
              paddingAngle={3}
              dataKey="value"
              labelLine={false}
              label={renderCustomLabel}
            >
              {chartData.map((entry, index) => (
                <Cell key={index} fill={entry.color} strokeWidth={0} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>

        {/* Centre label */}
        <div className="-mt-2 text-center">
          <p className="text-xs text-gray-500 mb-3">Dominant Sentiment</p>
          <span
            className="text-sm font-bold px-3 py-1 rounded-full"
            style={{ backgroundColor: dominant.color + "22", color: dominant.color }}
          >
            {dominant.name}
          </span>
        </div>

        {/* Legend */}
        <div className="flex gap-5 mt-5">
          {SLICES.map((s) => (
            <div key={s.key} className="flex items-center gap-1.5 text-xs text-gray-400">
              <span
                className="w-3 h-3 rounded-full inline-block"
                style={{ backgroundColor: s.color }}
              />
              {s.label}
              <span className="text-white font-semibold">{data[s.key].toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
