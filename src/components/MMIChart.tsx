"use client";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import type { MMIDataPoint } from "@/types";

interface Props {
  data: MMIDataPoint[];
}

function mmiColor(val: number): string {
  if (val < 20) return "#ef4444"; // red – Extreme Fear
  if (val < 40) return "#f97316"; // orange – Fear
  if (val < 60) return "#eab308"; // yellow – Neutral
  if (val < 80) return "#84cc16"; // light-green – Greed
  return "#22c55e";               // green – Extreme Greed
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const val: number = payload[0]?.value ?? 0;
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-gray-400 mb-1">{label}</p>
      <p className="font-semibold" style={{ color: mmiColor(val) }}>
        MMI: {val.toFixed(1)}
      </p>
      <p className="text-gray-400 mt-0.5">{payload[0]?.payload?.label}</p>
    </div>
  );
}

export default function MMIChart({ data }: Props) {
  const latest = data[data.length - 1];
  const latestColor = mmiColor(latest?.mmi ?? 50);
  const ticks = data.filter((_, i) => i % 7 === 0).map((d) => d.date);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h2 className="text-base font-semibold text-white">
            Market Mood Index
          </h2>
          <p className="text-xs text-gray-500 mt-0.5">
            Aggregated sentiment indicator (0 = Extreme Fear → 100 = Extreme Greed)
          </p>
        </div>
        {latest && (
          <div className="text-right">
            <p className="text-2xl font-bold" style={{ color: latestColor }}>
              {latest.mmi}
            </p>
            <p className="text-xs mt-0.5" style={{ color: latestColor }}>
              {latest.label}
            </p>
          </div>
        )}
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <AreaChart data={data} margin={{ top: 5, right: 20, left: -10, bottom: 0 }}>
          <defs>
            <linearGradient id="mmiGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="date"
            ticks={ticks}
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#374151" }}
            tickFormatter={(v) => v.slice(5)}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#374151" }}
          />
          <Tooltip content={<CustomTooltip />} />
          {/* Zone lines */}
          <ReferenceLine y={20} stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.4} />
          <ReferenceLine y={40} stroke="#f97316" strokeDasharray="3 3" strokeOpacity={0.4} />
          <ReferenceLine y={60} stroke="#84cc16" strokeDasharray="3 3" strokeOpacity={0.4} />
          <ReferenceLine y={80} stroke="#22c55e" strokeDasharray="3 3" strokeOpacity={0.4} />
          <Area
            type="monotone"
            dataKey="mmi"
            stroke="#f59e0b"
            strokeWidth={2}
            fill="url(#mmiGrad)"
            dot={false}
            activeDot={{ r: 4, fill: "#f59e0b" }}
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* Zone legend */}
      <div className="flex flex-wrap gap-3 mt-3">
        {[
          { label: "Extreme Fear", color: "#ef4444" },
          { label: "Fear", color: "#f97316" },
          { label: "Neutral", color: "#eab308" },
          { label: "Greed", color: "#84cc16" },
          { label: "Extreme Greed", color: "#22c55e" },
        ].map((z) => (
          <span key={z.label} className="flex items-center gap-1 text-xs text-gray-400">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: z.color }}
            />
            {z.label}
          </span>
        ))}
      </div>
    </div>
  );
}
