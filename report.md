# Model Evaluation Report

> Auto-generated on **2026-04-09 11:23:28**

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
| Linear Regression | Without | 0.0048 | 0.0035 | 0.9988 |
| Linear Regression | With | 0.0048 | 0.0035 | 0.9988 |
| Random Forest | Without | 0.0019 | 0.0013 | 0.9998 |
| Random Forest | With | 0.0019 | 0.0013 | 0.9998 |
| LSTM | Without | 0.0083 | 0.0063 | 0.9965 |
| LSTM | With | 0.0077 | 0.0053 | 0.9970 |

---

## Test Set Metrics (Validation)

| Model | Sentiment | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---|---|---|
| Linear Regression | Without | 0.0066 | 0.0054 | 0.9968 |
| Linear Regression | With | 0.0066 | 0.0054 | 0.9968 |
| Random Forest | Without | 0.0161 | 0.0103 | 0.9809 |
| Random Forest | With | 0.0158 | 0.0100 | 0.9817 |
| LSTM | Without | 0.0114 | 0.0095 | 0.9904 |
| LSTM | With | 0.0132 | 0.0096 | 0.9871 |

---

## Sentiment Feature Impact (Test Set)

Δ = metric_with_sentiment − metric_without_sentiment  
For RMSE/MAE: negative Δ = improvement. For R²: positive Δ = improvement.

| Model | ΔRMSE | ΔMAE | ΔR² |
|---|---|---|---|
| Linear Regression | +0.0000 ❌ | +0.0000 ❌ | -0.0000 ❌ |
| Random Forest | -0.0004 ✅ | -0.0003 ✅ | +0.0009 ✅ |
| LSTM | +0.0018 ❌ | +0.0002 ❌ | -0.0033 ❌ |

---

## LSTM Loss History (final epoch)

| Variant | Final Train Loss | Final Val Loss |
|---|---|---|
| Without Sentiment | 0.000104 | 0.000126 |
| With Sentiment | 0.000107 | 0.000187 |

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

