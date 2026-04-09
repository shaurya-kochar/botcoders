"""
api.py — FastAPI inference server for stock close-price prediction.

Loads all six model checkpoints from ./checkpoints/ at startup and exposes
prediction endpoints for each model × sentiment variant.

Run:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /predict/linear-regression
    POST /predict/random-forest
    POST /predict/lstm
    GET  /health
    GET  /models

Input note
----------
Features should be provided in **normalised** form (StandardScaler-scaled),
matching exactly what the models were trained on.

Feature order
    without sentiment : [open, high, low, volume]           (4 values)
    with    sentiment : [open, high, low, volume, avg_sentiment]  (5 values)

For the LSTM endpoint features must be given as a 5-row × 4/5-column list
(window_size=5 consecutive trading days, oldest first).
"""
import os
from contextlib import asynccontextmanager
from typing import Annotated

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from models.lstm import LSTMRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CKPT_DIR = os.path.join(ROOT, "checkpoints")

_CKPT = {
    "lr_no":   os.path.join(CKPT_DIR, "linear_regression_no_sentiment.joblib"),
    "lr_with": os.path.join(CKPT_DIR, "linear_regression_with_sentiment.joblib"),
    "rf_no":   os.path.join(CKPT_DIR, "random_forest_no_sentiment.joblib"),
    "rf_with": os.path.join(CKPT_DIR, "random_forest_with_sentiment.joblib"),
    "lstm_no":   os.path.join(CKPT_DIR, "lstm_window5_no_sentiment.pt"),
    "lstm_with": os.path.join(CKPT_DIR, "lstm_window5_with_sentiment.pt"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model registry (populated at startup)
# ---------------------------------------------------------------------------

_models: dict = {}


def _load_lstm(path: str) -> LSTMRegressor:
    meta = torch.load(path, map_location=DEVICE, weights_only=False)
    m = LSTMRegressor(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
    ).to(DEVICE)
    m.load_state_dict(meta["model_state_dict"])
    m.eval()
    return m


@asynccontextmanager
async def lifespan(app: FastAPI):
    _models["lr_no"]   = joblib.load(_CKPT["lr_no"])
    _models["lr_with"] = joblib.load(_CKPT["lr_with"])
    _models["rf_no"]   = joblib.load(_CKPT["rf_no"])
    _models["rf_with"] = joblib.load(_CKPT["rf_with"])
    _models["lstm_no"]   = _load_lstm(_CKPT["lstm_no"])
    _models["lstm_with"] = _load_lstm(_CKPT["lstm_with"])
    print(f"All checkpoints loaded from {CKPT_DIR}")
    yield
    _models.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Stock Close-Price Predictor",
    description=(
        "Predicts the normalised closing price of NIFTY50 / SP500 using "
        "Linear Regression, Random Forest, or LSTM (window=5).  "
        "**Input features must be pre-normalised** (StandardScaler) to match "
        "training distribution."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

FEAT_NO   = ["open", "high", "low", "volume"]
FEAT_WITH = ["open", "high", "low", "volume", "avg_sentiment"]
WINDOW    = 5


class FlatInput(BaseModel):
    """Single time-step input for Linear Regression and Random Forest."""
    open:          Annotated[float, Field(description="Normalised open price")]
    high:          Annotated[float, Field(description="Normalised high price")]
    low:           Annotated[float, Field(description="Normalised low price")]
    volume:        Annotated[float, Field(description="Normalised trading volume")]
    avg_sentiment: Annotated[float | None, Field(default=None,
                   description="Normalised average sentiment (omit for no-sentiment model)")]


class SequenceInput(BaseModel):
    """
    Five consecutive trading days for the LSTM model.

    Provide `window` as a list of 5 feature rows, oldest day first.
    Each row: [open, high, low, volume] without sentiment,
               [open, high, low, volume, avg_sentiment] with sentiment.
    """
    window: Annotated[
        list[list[float]],
        Field(
            description=(
                f"List of {WINDOW} rows, each row = 4 feature values "
                f"(open, high, low, volume) or 5 values "
                f"(+ avg_sentiment).  Oldest day first."
            )
        ),
    ]

    @model_validator(mode="after")
    def check_shape(self):
        if len(self.window) != WINDOW:
            raise ValueError(f"window must have exactly {WINDOW} rows, got {len(self.window)}")
        n_cols = len(self.window[0])
        if n_cols not in (4, 5):
            raise ValueError("Each row must have 4 (no sentiment) or 5 (with sentiment) values")
        if any(len(r) != n_cols for r in self.window):
            raise ValueError("All rows in window must have the same number of columns")
        return self


class PredictionResponse(BaseModel):
    model:         str
    with_sentiment: bool
    predicted_close: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_features(body: FlatInput, with_sentiment: bool) -> np.ndarray:
    row = [body.open, body.high, body.low, body.volume]
    if with_sentiment:
        if body.avg_sentiment is None:
            raise HTTPException(
                status_code=422,
                detail="avg_sentiment is required when using the with-sentiment model",
            )
        row.append(body.avg_sentiment)
    return np.array(row, dtype=np.float32).reshape(1, -1)


def _seq_features(body: SequenceInput, with_sentiment: bool) -> torch.Tensor:
    n_cols = len(body.window[0])
    expected = 5 if with_sentiment else 4
    if n_cols != expected:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{'With' if with_sentiment else 'Without'}-sentiment model expects "
                f"{expected} columns per row, got {n_cols}"
            ),
        )
    arr = np.array(body.window, dtype=np.float32)          # (5, n_cols)
    return torch.tensor(arr).unsqueeze(0).to(DEVICE)       # (1, 5, n_cols)


# ---------------------------------------------------------------------------
# Routes — Linear Regression
# ---------------------------------------------------------------------------

@app.post("/predict/linear-regression", response_model=PredictionResponse,
          summary="Linear Regression — single time-step prediction")
def predict_lr(body: FlatInput, sentiment: bool = False):
    """
    Predict closing price with the Linear Regression model.

    - **sentiment=false** (default): uses 4 features [open, high, low, volume]
    - **sentiment=true**: uses 5 features (adds avg_sentiment)
    """
    key = "lr_with" if sentiment else "lr_no"
    X = _flat_features(body, sentiment)
    pred = float(_models[key].predict(X)[0])
    return PredictionResponse(model="linear_regression", with_sentiment=sentiment,
                              predicted_close=pred)


# ---------------------------------------------------------------------------
# Routes — Random Forest
# ---------------------------------------------------------------------------

@app.post("/predict/random-forest", response_model=PredictionResponse,
          summary="Random Forest — single time-step prediction")
def predict_rf(body: FlatInput, sentiment: bool = False):
    """
    Predict closing price with the Random Forest model.

    - **sentiment=false** (default): uses 4 features
    - **sentiment=true**: uses 5 features (adds avg_sentiment)
    """
    key = "rf_with" if sentiment else "rf_no"
    X = _flat_features(body, sentiment)
    pred = float(_models[key].predict(X)[0])
    return PredictionResponse(model="random_forest", with_sentiment=sentiment,
                              predicted_close=pred)


# ---------------------------------------------------------------------------
# Routes — LSTM
# ---------------------------------------------------------------------------

@app.post("/predict/lstm", response_model=PredictionResponse,
          summary="LSTM (window=5) — 5-day sequence prediction")
def predict_lstm(body: SequenceInput, sentiment: bool = False):
    """
    Predict closing price with the LSTM model using a 5-day sliding window.

    Provide `window` as a list of **5 rows** (oldest → newest).

    - **sentiment=false** (default): each row = [open, high, low, volume]
    - **sentiment=true**: each row = [open, high, low, volume, avg_sentiment]
    """
    key = "lstm_with" if sentiment else "lstm_no"
    x = _seq_features(body, sentiment)
    with torch.no_grad():
        pred = float(_models[key](x).cpu().item())
    return PredictionResponse(model="lstm", with_sentiment=sentiment,
                              predicted_close=pred)


# ---------------------------------------------------------------------------
# Utility routes
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "models_loaded": list(_models.keys()),
            "device": str(DEVICE)}


@app.get("/models", summary="List loaded models and their feature schemas")
def list_models():
    return {
        "linear_regression": {
            "endpoint": "/predict/linear-regression",
            "input_type": "single row",
            "features_no_sentiment":   FEAT_NO,
            "features_with_sentiment": FEAT_WITH,
        },
        "random_forest": {
            "endpoint": "/predict/random-forest",
            "input_type": "single row",
            "features_no_sentiment":   FEAT_NO,
            "features_with_sentiment": FEAT_WITH,
        },
        "lstm": {
            "endpoint": "/predict/lstm",
            "input_type": f"sequence of {WINDOW} rows (oldest first)",
            "features_no_sentiment":   FEAT_NO,
            "features_with_sentiment": FEAT_WITH,
            "window_size": WINDOW,
        },
    }
