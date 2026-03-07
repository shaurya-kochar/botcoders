"use client";

import { ChevronDown } from "lucide-react";
import type { StockOption } from "@/types";

interface Props {
  options: StockOption[];
  selected: string;
  onChange: (ticker: string) => void;
}

export default function StockSelector({ options, selected, onChange }: Props) {
  const current = options.find((o) => o.ticker === selected);

  return (
    <div className="flex items-center gap-3">
      <span className="text-sm font-medium text-gray-400 uppercase tracking-wider">
        Stock
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
            <option key={opt.ticker} value={opt.ticker}>
              {opt.ticker} — {opt.name}
            </option>
          ))}
        </select>
        <ChevronDown
          size={16}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
        />
      </div>
      {current && (
        <span className="hidden sm:inline text-xs text-gray-500">
          {current.name}
        </span>
      )}
    </div>
  );
}
