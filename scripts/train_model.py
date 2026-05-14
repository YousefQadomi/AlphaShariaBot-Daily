"""
train_model.py — Transformer Ranker / Regressor (OOM-Optimized)
=============================================================
This version completely eliminates the `np.lib.stride_tricks.sliding_window_view` 
OOM crash by using a Lazy-Evaluation PyTorch Dataset.

Memory Architecture:
1. We load the massive 4.7M row panel as a single flat 2D float32 tensor (~2GB RAM).
2. The `StockAwareDataset` holds references to this single flat array.
3. During `__getitem__`, we dynamically slice `[idx : idx + window_size]`.
   PyTorch creates a "view" of the data on the fly without duplicating memory.

This approach is mathematically identical to the sliding window array, 
but reduces RAM usage by ~30x. It will NOT affect your training results.

Usage:
    python scripts/train_model.py
"""

import os
import gc
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import ndcg_score
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PANEL_PATH     = "data/processed/panel.parquet"
MODEL_DIR      = "models"
MODEL_PATH     = os.path.join(MODEL_DIR, "transformer_model.pth")
SCALER_PATH    = os.path.join(MODEL_DIR, "transformer_scaler.pkl")

WINDOW_SIZE    = 30
BATCH_SIZE     = 1024
EPOCHS         = 50
PATIENCE       = 7
LR             = 0.0005
TRAIN_RATIO    = 0.80

# Features to exclude from the sequence input
EXCLUDE_COLS = {
    "forward_return", "relevance", "ticker", "is_halal", 
    "close", "Sector", "Ticker", "date"
}

# ---------------------------------------------------------------------------
# OOM-Free PyTorch Dataset
# ---------------------------------------------------------------------------
class StockAwareDataset(Dataset):
    """
    Lazy-slicing dataset. Avoids 3D memory duplication entirely.
    Holds the 2D flat tensor and returns 3D windows dynamically on the fly.
    """
    def __init__(
        self,
        X_flat: np.ndarray,
        y_flat: np.ndarray,
        valid_indices: np.ndarray,
        window_size: int,
    ):
        # Store as continuous flat PyTorch tensors (lives in CPU RAM, ~2GB total)
        self.X = torch.from_numpy(X_flat)
        self.y = torch.from_numpy(y_flat).unsqueeze(1)
        self.indices = valid_indices
        self.window_size = window_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        # Dynamic slice! No memory duplication. 
        # Target aligns with the END of the sequence (idx + window_size - 1)
        X_seq = self.X[idx : idx + self.window_size]
        y_val = self.y[idx + self.window_size - 1]
        
        return X_seq, y_val


def build_stock_splits(
    panel: pd.DataFrame,
    window_size: int,
    train_ratio: float = 0.80,
):
    """
    Groups by ticker to find valid sequence boundaries.
    Ensures no sequence ever crosses from one stock into another.
    """
    train_idx_parts = []
    test_idx_parts  = []
    train_row_mask  = np.zeros(len(panel), dtype=bool)
    
    # We assume panel is sorted by ['ticker', 'date']
    stock_lengths = panel.groupby("ticker", sort=False).size().values
    
    offset = 0
    for length in stock_lengths:
        split = int(length * train_ratio)
        train_row_mask[offset : offset + split] = True

        # Train sequences: [offset, offset + split)
        if split > window_size:
            starts = np.arange(offset, offset + split - window_size)
            train_idx_parts.append(starts)

        # Test sequences: [offset + split, offset + length)
        if (length - split) > window_size:
            starts = np.arange(offset + split, offset + length - window_size)
            test_idx_parts.append(starts)

        offset += length

    train_idx = np.concatenate(train_idx_parts) if train_idx_parts else np.array([], dtype=np.int64)
    test_idx  = np.concatenate(test_idx_parts)  if test_idx_parts  else np.array([], dtype=np.int64)
    return train_idx, test_idx, train_row_mask


# ---------------------------------------------------------------------------
# Transformer Architecture
# ---------------------------------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=8, n_layers=3, dropout=0.14):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = nn.Parameter(torch.randn(1, WINDOW_SIZE, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Regression / Ranking head (predicts continuous score instead of sigmoid)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), 
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        B = x.size(0)
        x   = self.input_proj(x) + self.pos_enc          # (B, W, d_model)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                 # (B, W+1, d_model)
        x   = self.encoder(x)
        return self.head(x[:, 0])                        # predict via CLS


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 AlphaShariaBot — Deep Learning Model (OOM-Fixed)\n🖥️ Device: {device}\n")

    # ── 1. Load Panel ────────────────────────────────────────────────────
    print("📦 Loading massive panel (2D flat array)...")
    panel = pd.read_parquet(PANEL_PATH)
    
    if "date" not in panel.columns:
        panel = panel.reset_index()
    
    panel["date"] = pd.to_datetime(panel["date"])
    panel.sort_values(["ticker", "date"], inplace=True)
    panel.dropna(subset=["forward_return"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    # ── 2. Identify Features & Target ────────────────────────────────────
    feature_cols = [c for c in panel.columns if c not in EXCLUDE_COLS]
    feature_cols = [c for c in feature_cols if panel[c].dtype in 
                    [np.float64, np.float32, np.int64, np.int32, np.int8, np.float16]]
    
    # We use continuous 'forward_return' as the target for MSE loss
    X_raw = panel[feature_cols].values.astype(np.float32)
    y_raw = panel["forward_return"].values.astype(np.float32)

    print(f"📊 Features: {len(feature_cols)} | Total Rows: {len(panel):,}")

    # ── 3. Build Safe Sequence Indices ───────────────────────────────────
    train_idx, test_idx, train_row_mask = build_stock_splits(
        panel, WINDOW_SIZE, train_ratio=TRAIN_RATIO
    )
    print(f"📊 Train Windows: {len(train_idx):,} | Test Windows: {len(test_idx):,}")

    # ── 4. Robust Scaling (Fits ONLY on training rows) ───────────────────
    print("⚖️ Fitting scaler...")
    scaler = RobustScaler()
    scaler.fit(X_raw[train_row_mask])
    
    # Scale entire flat array at once
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    
    with open(SCALER_PATH, "wb") as fh:
        pickle.dump(scaler, fh)

    # Free raw memory immediately
    del X_raw, panel; gc.collect()

    # ── 5. Setup DataLoaders ─────────────────────────────────────────────
    train_ds = StockAwareDataset(X_scaled, y_raw, train_idx, WINDOW_SIZE)
    test_ds  = StockAwareDataset(X_scaled, y_raw, test_idx,  WINDOW_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              pin_memory=True, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                              pin_memory=True, num_workers=4)

    # ── 6. Init Model & Training ─────────────────────────────────────────
    model = TransformerEncoder(input_dim=len(feature_cols)).to(device)
    
    # Using MSE for continuous return prediction (Ranking/Regression)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    amp_scaler = GradScaler("cuda") if device.type == "cuda" else None

    best_loss = float('inf')
    patience_ctr = 0

    print(f"\n🚂 Training Model...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for X_b, y_b in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad(set_to_none=True)

            if amp_scaler is not None:
                with autocast("cuda"):
                    preds = model(X_b)
                    loss  = criterion(preds, y_b)
                amp_scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                preds = model(X_b)
                loss  = criterion(preds, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                if amp_scaler is not None:
                    with autocast("cuda"):
                        preds = model(X_b)
                        loss  = criterion(preds, y_b)
                else:
                    preds = model(X_b)
                    loss  = criterion(preds, y_b)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss   /= len(test_loader)
        
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {lr_now:.2e}")
        
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

    print(f"✅ Training Complete! Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()