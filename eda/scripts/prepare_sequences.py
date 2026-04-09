"""
prepare_sequences.py - FIXED (MACD normalised)
"""
import pandas as pd
import numpy as np
import os
 
os.makedirs("data", exist_ok=True)
SEQ_LEN = 6
FEATURE_COLS = ["open","high","low","close","volume","S_t","daily_return","rsi","macd","bb_width","momentum","confidence"]
TARGET_COL = "close"
 
def normalise_col(s):
    mn,mx = s.min(),s.max()
    return (s-mn)/(mx-mn) if mx-mn!=0 else s*0
 
def create_sequences(df, seq_len=SEQ_LEN):
    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df.copy()
    if "rsi" in df.columns: df["rsi"] = df["rsi"]/100.0
    if "macd" in df.columns: df["macd"] = normalise_col(df["macd"])
    data = df[available].values
    target = df[TARGET_COL].values
    dates = df["bucket"].values
    X,y,dl = [],[],[]
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i]); y.append(target[i]); dl.append(dates[i])
    return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32), dl, available
 
def process_market(csv_path, market_name):
    print(f"\n{'='*50}\n  {market_name}\n{'='*50}")
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found"); return
    df = pd.read_csv(csv_path)
    dc = "bucket" if "bucket" in df.columns else "time"
    df = df.rename(columns={dc:"bucket"})
    df["bucket"] = pd.to_datetime(df["bucket"])
    df = df.sort_values("bucket").reset_index(drop=True)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(0)
    X,y,dates,feat_names = create_sequences(df, SEQ_LEN)
    name = market_name.lower()
    np.save(f"data/X_{name}.npy", X)
    np.save(f"data/y_{name}.npy", y)
    print(f"  Rows: {len(df)} | X shape: {X.shape} | y shape: {y.shape}")
    print(f"  Features: {feat_names}")
    print(f"  All values 0-1? min={X.min():.3f} max={X.max():.3f}")
    print(f"  Saved → data/X_{name}.npy + data/y_{name}.npy")
    return X,y
 
if __name__ == "__main__":
    print("prepare_sequences.py — FIXED VERSION (all features normalised 0-1)")
    process_market("data/final_nifty.csv", "nifty50")
    process_market("data/final_sp500.csv", "sp500")
    print("\nDone! Give friend: X_nifty50.npy, y_nifty50.npy, X_sp500.npy, y_sp500.npy")
 