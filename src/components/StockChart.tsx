"use client";

import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import type { PriceDataPoint, AssetType } from "@/types";

interface Props {
  data: PriceDataPoint[];
  /** Symbol displayed in the heading: stock ticker or index symbol */
  ticker: string;
  assetType?: AssetType;
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as PriceDataPoint;
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2.5 text-xs shadow-xl min-w-[160px]">
      <p className="text-gray-400 mb-2 font-medium">{label}</p>
      <div className="space-y-0.5">
        <p className="text-gray-300">
          Open: <span className="text-white font-medium">${d.open.toFixed(2)}</span>
        </p>
        <p className="text-gray-300">
          High: <span className="text-green-400 font-medium">${d.high.toFixed(2)}</span>
        </p>
        <p className="text-gray-300">
          Low: <span className="text-red-400 font-medium">${d.low.toFixed(2)}</span>
        </p>
        <p className="text-gray-300">
          Close: <span className="text-white font-medium">${d.close.toFixed(2)}</span>
        </p>
        <p className="text-gray-300">
          Volume:{" "}
          <span className="text-indigo-300 font-medium">
            {(d.volume / 1_000_000).toFixed(1)}M
          </span>
        </p>
        {d.sentimentOverlay !== undefined && (
          <p className="text-gray-300 border-t border-gray-700 pt-1 mt-1">
            Sentiment:{" "}
            <span
              className="font-medium"
              style={{ color: d.sentimentOverlay > 0 ? "#22c55e" : "#ef4444" }}
            >
              {d.sentimentOverlay >= 0 ? "+" : ""}
              {d.sentimentOverlay.toFixed(3)}
            </span>
          </p>
        )}
      </div>
    </div>
  );
}

export default function StockChart({ data, ticker, assetType = "stock" }: Props) {
  const ticks = data.filter((_, i) => i % 7 === 0).map((d) => d.date);
  const prices = data.map((d) => d.close);
  const minP = Math.floor(Math.min(...prices) * 0.99);
  const maxP = Math.ceil(Math.max(...prices) * 1.01);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
      <div className="mb-4">
        <h2 className="text-base font-semibold text-white">
          {ticker} — {assetType === "index" ? "Index Level" : "Stock Price"}
        </h2>
        <p className="text-xs text-gray-500 mt-0.5">
          {assetType === "index"
            ? "Index close level with traded volume and sentiment overlay"
            : "Close price with volume bars and sentiment colour overlay"}
        </p>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="date"
            ticks={ticks}
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#374151" }}
            tickFormatter={(v) => v.slice(5)}
          />
          {/* Left axis: price */}
          <YAxis
            yAxisId="price"
            orientation="left"
            domain={[minP, maxP]}
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#374151" }}
            tickFormatter={(v) => `$${v}`}
          />
          {/* Right axis: volume */}
          <YAxis
            yAxisId="volume"
            orientation="right"
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `${(v / 1_000_000).toFixed(0)}M`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: "12px", color: "#9ca3af", paddingTop: "12px" }}
          />
          <Bar
            yAxisId="volume"
            dataKey="volume"
            name="Volume"
            fill="#312e81"
            opacity={0.5}
            radius={[2, 2, 0, 0]}
          />
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="close"
            name="Close Price"
            stroke="#6366f1"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "#6366f1" }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
