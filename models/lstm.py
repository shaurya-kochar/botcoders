import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets.dataloader import get_dataloaders, get_sequence_dataloaders


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMRegressor(nn.Module):
    """
    LSTM regressor.

    Accepts input of shape (batch, seq_len, input_size).
    When a 2-D tensor (batch, features) is passed the sequence length is
    assumed to be 1 (single time-step, backward-compatible path).
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept both (batch, features) and (batch, seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)            # (batch, 1, features)
        out, _ = self.lstm(x)             # (batch, seq_len, hidden_size)
        out = out[:, -1, :]               # last time-step: (batch, hidden_size)
        return self.fc(out)               # (batch, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader, device: torch.device, label: str = ""):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            preds.append(y_pred)
            targets.append(y_batch.numpy())

    preds = np.vstack(preds).ravel()
    targets = np.vstack(targets).ravel()

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    print(f"  {label}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    R²   : {r2:.4f}")
    return preds, targets, {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    csv_path: str,
    with_sentiment: bool = True,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 32,
    window_size: int = 5,
    checkpoint_dir: str = None,
):
    """
    Train an LSTM regressor.

    When window_size > 1 (default 5), uses get_sequence_dataloaders which
    reads combined_clean.csv and builds proper sliding-window sequences.
    Feature set:
        window_size > 1, no  sentiment: open, high, low, volume          (4)
        window_size > 1, with sentiment: open, high, low, volume, avg_sentiment (5)
        window_size == 1, no  sentiment: original 6-feature set
        window_size == 1, with sentiment: original 8-feature set

    Args:
        csv_path:       Path to the dataset CSV.
        with_sentiment: Whether to include sentiment features.
        hidden_size:    LSTM hidden state size.
        num_layers:     Number of LSTM layers.
        dropout:        Dropout between LSTM layers (only if num_layers > 1).
        lr:             Learning rate.
        epochs:         Training epochs.
        batch_size:     Samples per batch.
        window_size:    Sliding-window length in trading days (default 5).
    """
    tag = "With Sentiment" if with_sentiment else "Without Sentiment"
    print(f"\n=== LSTM (window={window_size}) — {tag} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    if window_size > 1:
        train_loader, test_loader, _ = get_sequence_dataloaders(
            csv_path,
            window=window_size,
            test_size=0.2,
            batch_size=batch_size,
            with_sentiment=with_sentiment,
        )
        input_size = 5 if with_sentiment else 4
    else:
        train_loader, test_loader, _ = get_dataloaders(
            csv_path, test_size=0.2, batch_size=batch_size,
            with_sentiment=with_sentiment,
        )
        input_size = 8 if with_sentiment else 6

    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    train_loss_history = []
    val_loss_history   = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        avg_val_loss = val_loss / len(test_loader)
        val_loss_history.append(avg_val_loss)

        if epoch % 10 == 0:
            print(f"  Epoch [{epoch:>4}/{epochs}]  "
                  f"Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")

    print("\n  --- Final Evaluation ---")
    tr_preds, tr_targets, tr_metrics = evaluate(model, train_loader, device, label="[Train]")
    te_preds, te_targets, te_metrics = evaluate(model, test_loader,  device, label="[Test] ")

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        suffix = "with_sentiment" if with_sentiment else "no_sentiment"
        ckpt_path = os.path.join(checkpoint_dir, f"lstm_window{window_size}_{suffix}.pt")
        torch.save({"model_state_dict": model.state_dict(),
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "window_size": window_size,
                    "with_sentiment": with_sentiment}, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    results = {
        "train": {**tr_metrics, "preds": tr_preds, "targets": tr_targets},
        "test":  {**te_metrics, "preds": te_preds, "targets": te_targets},
        "train_loss_history": train_loss_history,
        "val_loss_history":   val_loss_history,
    }
    return model, results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = os.path.expanduser(
        "~/dataset/inlp_project/final_sub/combined_clean.csv"
    )
    train(csv_path, with_sentiment=False, epochs=50, window_size=5)
    train(csv_path, with_sentiment=True,  epochs=50, window_size=5)
