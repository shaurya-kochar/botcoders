# Model Evaluation Report

> Auto-generated on **2026-03-08 07:08:26**

## Overview

This report compares **Linear Regression**, **Random Forest**, and **LSTM** on a stock
close-price prediction task, evaluated with and without Twitter sentiment features.

**Features without sentiment (6):** open, high, low, volume, running_max, running_min  
**Additional sentiment features (2):** positiveness, negativeness  
**Target:** close price  

---

## Training Set Metrics

| Model | Sentiment | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---|---|---|
| Linear Regression | Without | 0.2351 | 0.1814 | 0.9989 |
| Linear Regression | With | 0.2351 | 0.1814 | 0.9989 |
| Random Forest | Without | 0.0116 | 0.0003 | 1.0000 |
| Random Forest | With | 0.0120 | 0.0004 | 1.0000 |
| LSTM | Without | 6.4472 | 1.6797 | 0.1854 |
| LSTM | With | 6.2656 | 1.5689 | 0.2307 |

---

## Test Set Metrics (Validation)

| Model | Sentiment | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---|---|---|
| Linear Regression | Without | 0.4439 | 0.1448 | 0.9989 |
| Linear Regression | With | 0.4455 | 0.1449 | 0.9989 |
| Random Forest | Without | 7.9131 | 0.7281 | 0.6546 |
| Random Forest | With | 7.9230 | 0.7308 | 0.6538 |
| LSTM | Without | 12.8530 | 2.1541 | 0.0888 |
| LSTM | With | 12.7274 | 2.1988 | 0.1065 |

---

## Sentiment Feature Impact (Test Set)

Δ = metric_with_sentiment − metric_without_sentiment  
For RMSE/MAE: negative Δ = improvement. For R²: positive Δ = improvement.

| Model | ΔRMSE | ΔMAE | ΔR² |
|---|---|---|---|
| Linear Regression | +0.0015 ❌ | +0.0001 ❌ | -0.0000 ❌ |
| Random Forest | +0.0099 ❌ | +0.0028 ❌ | -0.0009 ❌ |
| LSTM | -0.1256 ✅ | +0.0446 ❌ | +0.0177 ✅ |

---

## LSTM Loss History (final epoch)

| Variant | Final Train Loss | Final Val Loss |
|---|---|---|
| Without Sentiment | 42.691293 | 168.227892 |
| With Sentiment | 40.508198 | 164.977533 |

---

## Plots

### Training Metrics Comparison

![Training Metrics Comparison](plots/metrics_comparison_train.png)

### Test Metrics Comparison

![Test Metrics Comparison](plots/metrics_comparison_test.png)

### Actual vs Predicted (Test Set)

![Actual vs Predicted (Test Set)](plots/actual_vs_predicted.png)

### LSTM Training & Validation Loss

![LSTM Training & Validation Loss](plots/lstm_loss_curves.png)

### Sentiment Feature Impact

![Sentiment Feature Impact](plots/sentiment_impact.png)

