# Model Evaluation Report

> Auto-generated on **2026-03-07 04:50:09**

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
| Linear Regression | Without | 0.9964 | 0.7896 | 0.9188 |
| Linear Regression | With | 0.6577 | 0.5630 | 0.9646 |
| Random Forest | Without | 0.8642 | 0.8088 | 0.9389 |
| Random Forest | With | 0.7851 | 0.6745 | 0.9496 |
| LSTM | Without | 128.7524 | 128.7052 | -1354.4702 |
| LSTM | With | 129.0127 | 128.9656 | -1359.9554 |

---

## Test Set Metrics (Validation)

| Model | Sentiment | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---|---|---|
| Linear Regression | Without | 1.1061 | 1.0763 | 0.1635 |
| Linear Regression | With | 1.2410 | 1.1575 | -0.0531 |
| Random Forest | Without | 4.8412 | 4.7613 | -15.0255 |
| Random Forest | With | 5.3473 | 5.1970 | -18.5510 |
| LSTM | Without | 138.4350 | 138.4298 | -13102.7578 |
| LSTM | With | 138.7139 | 138.7087 | -13155.6055 |

---

## Sentiment Feature Impact (Test Set)

Δ = metric_with_sentiment − metric_without_sentiment  
For RMSE/MAE: negative Δ = improvement. For R²: positive Δ = improvement.

| Model | ΔRMSE | ΔMAE | ΔR² |
|---|---|---|---|
| Linear Regression | +0.1350 ❌ | +0.0812 ❌ | -0.2166 ❌ |
| Random Forest | +0.5061 ❌ | +0.4357 ❌ | -3.5255 ❌ |
| LSTM | +0.2789 ❌ | +0.2789 ❌ | -52.8477 ❌ |

---

## LSTM Loss History (final epoch)

| Variant | Final Train Loss | Final Val Loss |
|---|---|---|
| Without Sentiment | 16616.241943 | 19164.255859 |
| With Sentiment | 16683.417480 | 19241.548828 |

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

