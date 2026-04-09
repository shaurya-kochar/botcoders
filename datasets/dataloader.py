import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Original dataset feature columns ─────────────────────────────────────────
FEATURE_COLS = [
    "open", "high", "low", "volume",
    "running_max", "running_min",
    "positiveness", "negativeness",
]
TARGET_COL = "close"

# ── combined_clean.csv feature columns ───────────────────────────────────────
COMBINED_BASE_COLS      = ["open", "high", "low", "volume"]
COMBINED_SENTIMENT_COLS = ["open", "high", "low", "volume", "avg_sentiment"]


class StockDataset(Dataset):
    """PyTorch Dataset for stock price prediction (single time-step per sample)."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset):
    """Sliding-window sequence dataset for LSTM (window, features) → close."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        # sequences: (N, window, n_features)
        # targets:   (N,)
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(targets,   dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(
    csv_path: str,
    test_size: float = 0.2,
    batch_size: int = 4,
    shuffle_train: bool = True,
    random_state: int = 42,
    with_sentiment: bool = True,
):
    """
    Load the dataset CSV, split into train/test, scale features,
    and return PyTorch DataLoader objects.

    Args:
        csv_path:       Path to the dataset CSV file.
        test_size:      Fraction of data to reserve for test (default 0.2).
        batch_size:     Number of samples per batch (default 4).
        shuffle_train:  Whether to shuffle the training set (default True).
        random_state:   Seed for reproducibility (default 42).
        with_sentiment: If True, include positiveness & negativeness features.
                        If False, use only the 6 base stock features.

    Returns:
        train_loader:   DataLoader for the training set.
        test_loader:    DataLoader for the test set.
        scaler:         Fitted StandardScaler (for inverse-transforming predictions).
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = FEATURE_COLS if with_sentiment else FEATURE_COLS[:6]

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler


# ── Sequence DataLoader for combined_clean.csv (sliding window) ───────────────

def _build_sequences(df: pd.DataFrame, feature_cols: list, window: int):
    """
    Build (window, features) → close sliding-window samples from a sorted group.

    For each position i >= window-1, the input is rows [i-window+1 .. i] and
    the target is close[i] (the close of the last day in the window).
    """
    X_arr = df[feature_cols].values.astype(np.float32)
    y_arr = df["close"].values.astype(np.float32)
    seqs, targets = [], []
    for i in range(window - 1, len(df)):
        seqs.append(X_arr[i - window + 1 : i + 1])  # (window, n_features)
        targets.append(y_arr[i])
    return np.array(seqs, dtype=np.float32), np.array(targets, dtype=np.float32)


def get_sequence_dataloaders(
    csv_path: str,
    window: int = 5,
    test_size: float = 0.2,
    batch_size: int = 32,
    with_sentiment: bool = True,
):
    """
    Build sliding-window sequence DataLoaders from combined_clean.csv.

    Each sample is a tensor of shape (window, n_features) and the target is
    the normalised close price of the last day in the window.

    Sequences are built per index (NIFTY50 / SP500) so windows never cross
    index boundaries. Each index is split 80/20 (train/test) chronologically,
    then the splits are concatenated.

    Args:
        csv_path:       Path to combined_clean.csv.
        window:         Number of consecutive trading days per sample (default 5).
        test_size:      Fraction of each index's samples reserved for test.
        batch_size:     Samples per batch.
        with_sentiment: If True, include avg_sentiment as a feature.

    Returns:
        train_loader, test_loader, scaler
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values(["index", "date"]).reset_index(drop=True)

    feature_cols = COMBINED_SENTIMENT_COLS if with_sentiment else COMBINED_BASE_COLS
    n_features   = len(feature_cols)

    train_seqs,  train_targets  = [], []
    test_seqs,   test_targets   = [], []

    for _, grp in df.groupby("index", sort=True):
        grp = grp.reset_index(drop=True)
        seqs, targets = _build_sequences(grp, feature_cols, window)
        if len(seqs) == 0:
            continue
        split = max(1, int(len(seqs) * (1 - test_size)))
        train_seqs.append(seqs[:split]);   train_targets.append(targets[:split])
        test_seqs.append(seqs[split:]);    test_targets.append(targets[split:])

    X_train = np.concatenate(train_seqs,   axis=0)
    y_train = np.concatenate(train_targets, axis=0)
    X_test  = np.concatenate(test_seqs,    axis=0)
    y_test  = np.concatenate(test_targets,  axis=0)

    # Fit scaler on flattened train features, apply to both splits
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, n_features)
    ).reshape(X_train.shape)
    X_test  = scaler.transform(
        X_test.reshape(-1, n_features)
    ).reshape(X_test.shape)

    train_loader = DataLoader(
        SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        SequenceDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    n_feat_str = f"{n_features} features ({'w/' if with_sentiment else 'w/o'} sentiment)"
    print(f"  Sequence DataLoader | window={window} | {n_feat_str}")
    print(f"  Train samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")

    return train_loader, test_loader, scaler


def get_flat_dataloaders(
    csv_path: str,
    test_size: float = 0.2,
    batch_size: int = 32,
    shuffle_train: bool = True,
    with_sentiment: bool = True,
):
    """
    Load combined_clean.csv, apply an 80/20 temporal split per index,
    scale features, and return DataLoader objects for LR / RF / non-LSTM models.

    feature set (no sentiment): open, high, low, volume          (4 cols)
    feature set (with sentiment): open, high, low, volume, avg_sentiment (5 cols)
    target: close
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values(["index", "date"]).reset_index(drop=True)

    feature_cols = COMBINED_SENTIMENT_COLS if with_sentiment else COMBINED_BASE_COLS

    train_X, train_y, test_X, test_y = [], [], [], []
    for _, grp in df.groupby("index", sort=True):
        grp = grp.reset_index(drop=True)
        X = grp[feature_cols].values.astype(np.float32)
        y = grp["close"].values.astype(np.float32)
        split = max(1, int(len(X) * (1 - test_size)))
        train_X.append(X[:split]);  train_y.append(y[:split])
        test_X.append(X[split:]);   test_y.append(y[split:])

    X_train = np.concatenate(train_X, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    X_test  = np.concatenate(test_X,  axis=0)
    y_test  = np.concatenate(test_y,  axis=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    train_loader = DataLoader(
        StockDataset(X_train, y_train), batch_size=batch_size, shuffle=shuffle_train
    )
    test_loader = DataLoader(
        StockDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    n_feat_str = f"{len(feature_cols)} features ({'w/' if with_sentiment else 'w/o'} sentiment)"
    print(f"  Flat DataLoader | {n_feat_str}")
    print(f"  Train samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")

    return train_loader, test_loader, scaler


if __name__ == "__main__":
    import os

    csv_path = os.path.join(os.path.dirname(__file__), "../samples/dataset.csv")

    print("=== With Sentiment (8 features) ===")
    train_loader, test_loader, scaler = get_dataloaders(
        csv_path, test_size=0.2, batch_size=4, with_sentiment=True
    )
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    for i, (X_batch, y_batch) in enumerate(train_loader):
        print(f"  Batch {i+1}: X={X_batch.shape}, y={y_batch.shape}")

    print("\n=== Without Sentiment (6 features) ===")
    train_loader, test_loader, _ = get_dataloaders(
        csv_path, test_size=0.2, batch_size=4, with_sentiment=False
    )
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    for i, (X_batch, y_batch) in enumerate(train_loader):
        print(f"  Batch {i+1}: X={X_batch.shape}, y={y_batch.shape}")
