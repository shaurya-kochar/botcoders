import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "SentimentStock — NLP Market Intelligence",
  description:
    "Evaluating the Impact of Social Media on Stock Prices using FinBERT sentiment analysis and LSTM prediction models.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-gray-950 text-gray-100 antialiased">
        <Navbar />
        <main id="dashboard">{children}</main>
      </body>
    </html>
  );
}
