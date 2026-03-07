import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets.dataloader import get_dataloaders


def evaluate(model, loader, label=""):
    X_all, y_all = [], []
    for X_batch, y_batch in loader:
        X_all.append(X_batch.numpy())
        y_all.append(y_batch.numpy())
    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all).ravel()

    y_pred = model.predict(X_all)
    rmse = np.sqrt(mean_squared_error(y_all, y_pred))
    mae = mean_absolute_error(y_all, y_pred)
    r2 = r2_score(y_all, y_pred)

    print(f"{label}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    return y_pred, y_all, {"rmse": rmse, "mae": mae, "r2": r2}


def run(csv_path: str, with_sentiment: bool = True):
    tag = "With Sentiment (8 features)" if with_sentiment else "Without Sentiment (6 features)"
    print(f"\n=== Linear Regression — {tag} ===")

    train_loader, test_loader, _ = get_dataloaders(
        csv_path, test_size=0.2, batch_size=32, with_sentiment=with_sentiment
    )

    # Collect all training data
    X_train, y_train = [], []
    for X_batch, y_batch in train_loader:
        X_train.append(X_batch.numpy())
        y_train.append(y_batch.numpy())
    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train).ravel()

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"  Coefficients : {model.coef_}")
    print(f"  Intercept    : {model.intercept_:.4f}")

    tr_preds, tr_targets, tr_metrics = evaluate(model, train_loader, label="  [Train]")
    te_preds, te_targets, te_metrics = evaluate(model, test_loader,  label="  [Test] ")

    results = {
        "train": {**tr_metrics, "preds": tr_preds, "targets": tr_targets},
        "test":  {**te_metrics, "preds": te_preds, "targets": te_targets},
    }
    return model, results


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "../samples/dataset.csv")
    run(csv_path, with_sentiment=False)
    run(csv_path, with_sentiment=True)
