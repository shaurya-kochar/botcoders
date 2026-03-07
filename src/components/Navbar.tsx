"use client";

import { BarChart2, Github } from "lucide-react";

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 border-b border-gray-800 bg-gray-950/80 backdrop-blur-md">
      <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-indigo-600">
            <BarChart2 size={18} className="text-white" />
          </div>
          <div>
            <span className="font-bold text-white text-sm leading-none">SentimentStock</span>
            <span className="block text-[10px] text-gray-500 leading-none mt-0.5">
              NLP × Market Intelligence
            </span>
          </div>
        </div>

        {/* Nav links */}
        <nav className="hidden md:flex items-center gap-6 text-sm text-gray-400">
          <a href="#dashboard" className="hover:text-white transition-colors">Dashboard</a>
          <a href="#about" className="hover:text-white transition-colors">About</a>
        </nav>

        {/* Right actions */}
        <div className="flex items-center gap-3">
          <span className="hidden sm:inline-flex items-center gap-1.5 text-xs bg-green-500/10 text-green-400 border border-green-500/20 px-2.5 py-1 rounded-full">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            Mock Data
          </span>
          <a
            href="https://github.com/shaurya-kochar/botcoders"
            target="_blank"
            rel="noreferrer"
            className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-white transition-colors"
          >
            <Github size={18} />
          </a>
        </div>
      </div>
    </header>
  );
}
