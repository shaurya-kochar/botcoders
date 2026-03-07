# Data Engineering

## What This Folder Contains
| File | Description |
|------|-------------|
| `final_data.ipynb` | Complete data pipeline notebook |
| `FINAL_clean_dataset.csv` | Final dataset — 429,453 rows, 26 columns |
| `ALL_charts_final.png` | Dataset overview charts |
| `before_after_chart.png` | Price before/after news analysis |

## Dataset Summary
- **Rows:** 429,453
- **Columns:** 26  
- **Size:** 324 MB
- **Date Range:** January 2018 to February 2026
- **Stocks:** AAPL, TSLA, GOOGL, AMZN, MSFT, CRM

## Data Sources
| Source | URL |
|--------|-----|
| Reuters Financial News | huggingface.co/datasets/ashraq/financial-news-articles |
| Yahoo Finance API | finance.yahoo.com via yfinance |
| Salesforce Case Study | Real CEO controversy Feb 2026 |

## All 26 Columns
| Column | Description |
|--------|-------------|
| date | Real Reuters publication date |
| company | Stock symbol |
| tweet | News article text |
| source | reuters |
| news_type | specific or general |
| sentiment_label | positive / negative / neutral |
| positiveness | Score 0-1 |
| negativeness | Score 0-1 |
| open | Opening price on news day |
| high | Highest price on news day |
| low | Lowest price on news day |
| close | Closing price on news day |
| volume | Shares traded on news day |
| running_max | All-time high so far |
| running_min | All-time low so far |
| open_day_before | Opening price day before news |
| high_day_before | Highest price day before news |
| low_day_before | Lowest price day before news |
| close_day_before | Closing price day before news |
| open_day_after | Opening price day after news |
| high_day_after | Highest price day after news |
| low_day_after | Lowest price day after news |
| close_day_after | Closing price day after news |
| price_change_after | How much price moved after news |
| direction | UP / DOWN / FLAT |
| y | Target variable = close price |

## Pipeline Steps
1. Load 306,242 Reuters articles from HuggingFace
2. Extract real dates using regex
3. Clean text and calculate positiveness/negativeness scores
4. Classify articles — specific (one stock) or general (all stocks)  
5. Download OHLCV prices from Yahoo Finance 2018-2026
6. Merge news with stock prices by date + company
7. Add prices from day before and day after each article
8. Calculate direction UP/DOWN/FLAT
9. Add Salesforce 2026 case study
