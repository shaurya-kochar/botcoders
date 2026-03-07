"use client";

import { ChevronDown } from "lucide-react";
import type { AssetOption, AssetType } from "@/types";

interface Props {
  options: AssetOption[];
  selected: string;          // the symbol
  assetType: AssetType;      // drives label copy
  onChange: (symbol: string) => void;
}

// Region flag emoji — keeps things visually rich without extra deps
const REGION_FLAGS: Record<string, string> = {
  US: "🇺🇸",
  IN: "🇮🇳",
};

export default function AssetSelector({ options, selected, assetType, onChange }: Props) {
  const current = options.find((o) => o.symbol === selected);
  const label = assetType === "stock" ? "Stock" : "Index";

  return (
    <div className="flex items-center gap-3">
      <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
        {label}
      </span>
      <div className="relative">
        <select
          value={selected}
          onChange={(e) => onChange(e.target.value)}
          className="
            appearance-none bg-gray-800 border border-gray-700 text-white
            text-sm font-semibold rounded-lg pl-4 pr-10 py-2.5
            focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
            hover:border-gray-500 cursor-pointer transition-colors
          "
        >
          {options.map((opt) => (
            <option key={opt.symbol} value={opt.symbol}>
              {opt.region ? `${REGION_FLAGS[opt.region] ?? ""} ` : ""}
              {opt.symbol} — {opt.name}
            </option>
          ))}
        </select>
        <ChevronDown
          size={16}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
        />
      </div>
      {current?.region && (
        <span className="hidden sm:inline text-lg leading-none" aria-label={current.region}>
          {REGION_FLAGS[current.region]}
        </span>
      )}
    </div>
  );
}
