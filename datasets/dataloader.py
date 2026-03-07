import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "open", "high", "low", "volume",
    "running_max", "running_min",
    "positiveness", "negativeness",
]
TARGET_COL = "close"


class StockDataset(Dataset):
    """PyTorch Dataset for stock price prediction."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

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
