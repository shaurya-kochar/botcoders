"use client";

import type { AssetType } from "@/types";

interface Props {
  value: AssetType;
  onChange: (type: AssetType) => void;
}

const OPTIONS: { value: AssetType; label: string; description: string }[] = [
  {
    value: "stock",
    label: "Individual Stock",
    description: "AAPL · TSLA · NVDA · AMZN",
  },
  {
    value: "index",
    label: "Market Index",
    description: "NIFTY · SENSEX · S&P 500 · NASDAQ",
  },
];

export default function AssetTypeSelector({ value, onChange }: Props) {
  return (
    <div className="flex items-center gap-1 bg-gray-800/60 border border-gray-700/60 rounded-xl p-1">
      {OPTIONS.map((opt) => {
        const active = value === opt.value;
        return (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className={`
              relative flex flex-col items-start px-4 py-2 rounded-lg text-left
              transition-all duration-200 min-w-[140px]
              ${
                active
                  ? "bg-indigo-600 shadow-lg shadow-indigo-900/40"
                  : "hover:bg-gray-700/60 text-gray-400 hover:text-gray-200"
              }
            `}
          >
            <span
              className={`text-sm font-semibold leading-none ${
                active ? "text-white" : ""
              }`}
            >
              {opt.label}
            </span>
            <span
              className={`text-[10px] mt-1 leading-none tracking-wide ${
                active ? "text-indigo-200" : "text-gray-600"
              }`}
            >
              {opt.description}
            </span>
          </button>
        );
      })}
    </div>
  );
}
