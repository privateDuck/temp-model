import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

from config import (
    CITIES, EXCLUDED_FROM_MODEL, EXTRA_STATIC, FEATURES_DIR, FEATURE_COLUMNS,
    FFNN_PARAMS, LSTM_PARAMS, LSTM_SEQ_FEATURES, MODEL_DIR, RIDGE_ALPHA, SEED,
    TARGET, TEST_MONTHS, TRAIN_MONTHS,
)
from models import FFNN, LSTMForecaster

np.random.seed(SEED)
torch.manual_seed(SEED)


# Data Preparation
def load_and_prepare():
    df = pd.read_parquet(FEATURES_DIR)
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    # clear_sky_index is null at night by design; 0 is the correct value.
    df["clear_sky_index"] = df["clear_sky_index"].fillna(pd.Series(np.where(df["is_daylight"] == 0, 0.0, np.nan), index=df.index))

    feats = [c for c in FEATURE_COLUMNS
             if c not in EXCLUDED_FROM_MODEL and c in df.columns]
    feats += [c for c in EXTRA_STATIC if c in df.columns]

    df = df.dropna(subset=[TARGET] + feats).reset_index(drop=True)
    print(f"[data] {len(df):,} rows after cleaning "
          f"({df['timestamp'].min()} -> {df['timestamp'].max()})")

    # Chronological split: 24m train / 6m dev / remainder = validation.
    t0 = df["timestamp"].min()
    train_end = t0 + pd.DateOffset(months=TRAIN_MONTHS)
    dev_end = train_end + pd.DateOffset(months=TEST_MONTHS)
    parts = {
        "train": df[df["timestamp"] < train_end],
        "dev": df[(df["timestamp"] >= train_end) & (df["timestamp"] < dev_end)],
        "val": df[df["timestamp"] >= dev_end],
    }
    for name, d in parts.items():
        print(f"[split] {name:5s}: {d['timestamp'].min()} -> "
              f"{d['timestamp'].max()}  ({len(d):,} rows)")

    # Scale numeric features on train only; city dummies pass through.
    scaler = StandardScaler().fit(parts["train"][feats])
    frames = {}
    for name, d in parts.items():
        X = pd.DataFrame(scaler.transform(d[feats]), columns=feats,
                         index=d.index)
        dummies = pd.get_dummies(d["city"], prefix="city").reindex(
            columns=[f"city_{c}" for c in CITIES], fill_value=0).astype(float)
        frames[name] = (pd.concat([X, dummies], axis=1).astype(np.float32),
                        d[TARGET].to_numpy(np.float32))

    return parts, feats, scaler, frames


def build_seq_data(d: pd.DataFrame, seq_scaler: StandardScaler):
    """Per-city 48h windows for the LSTM, strictly inside the split."""
    scaled = d.copy()
    seq_numeric = [c for c in LSTM_SEQ_FEATURES
                   if c not in ("hour_sin", "hour_cos")]
    scaled[seq_numeric] = seq_scaler.transform(d[seq_numeric])

    per_city = {}
    for city in sorted(d["city"].unique()):
        cd = scaled[scaled["city"] == city]
        if len(cd) > LSTM_PARAMS["seq_len"]:
            per_city[city] = (
                cd[LSTM_SEQ_FEATURES].to_numpy(np.float32),
                cd[TARGET].to_numpy(np.float32),
                np.arange(LSTM_PARAMS["seq_len"] - 1, len(cd)),
            )
    return per_city


class SeqDataset(Dataset):
    """Sample = window [t-47h, t] + city one-hot -> target at t+6h."""

    def __init__(self, per_city: dict):
        self.per_city = per_city
        self.city_idx = {c: i for i, c in enumerate(sorted(per_city))}
        self.samples = [(c, int(s))
                        for c, (_X, _y, starts) in per_city.items()
                        for s in starts]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        city, start = self.samples[i]
        X, y, _ = self.per_city[city]
        n = LSTM_PARAMS["seq_len"]
        window = X[start - n + 1: start + 1]
        onehot = np.zeros(len(self.city_idx), np.float32)
        onehot[self.city_idx[city]] = 1.0
        return (torch.from_numpy(window), torch.from_numpy(onehot),
                torch.tensor(y[start]))


# Training (Huber loss, fixed hyperparameters, early stopping on dev MAE)
@torch.no_grad()
def predict_torch(model, loader, kind):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        out = (model(*batch[:-1]) if kind == "lstm" else model(batch[0]))
        preds.append(out.numpy())
        trues.append(batch[-1].numpy())
    return np.concatenate(preds), np.concatenate(trues)


def train_torch(model, train_loader, dev_loader, params, kind):
    loss_fn = nn.HuberLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                            weight_decay=params["weight_decay"])
    best_mae, best_state, best_epoch = np.inf, None, 0

    for epoch in range(1, params["epochs"] + 1):
        model.train()
        for batch in train_loader:
            out = (model(*batch[:-1]) if kind == "lstm" else model(batch[0]))
            loss = loss_fn(out, batch[-1])
            opt.zero_grad()
            loss.backward()
            opt.step()

        dev_pred, dev_true = predict_torch(model, dev_loader, kind)
        dev_mae = mean_absolute_error(dev_true, dev_pred)
        if dev_mae < best_mae:
            best_mae, best_state, best_epoch = (
                dev_mae,
                {k: v.clone() for k, v in model.state_dict().items()},
                epoch)
        elif epoch - best_epoch >= params["patience"]:
            break

    model.load_state_dict(best_state)
    print(f"[train] {kind}: best dev MAE {best_mae:.3f} degC "
          f"(epoch {best_epoch})")
    return model


# Evaluation
def metrics(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)}


def print_table(results):
    print(f"\n{'model':12s} {'split':6s} {'MAE':>7s} {'RMSE':>7s} {'R2':>7s}")
    for name, rows in results.items():
        for split, m in rows.items():
            print(f"{name:12s} {split:6s} {m['MAE']:7.3f} {m['RMSE']:7.3f} "
                  f"{m['R2']:7.3f}")


def ffnn_eval_rows(model, frames, batch_size=2048):
    rows = {}
    for split, (X, y) in frames.items():
        pred, true = predict_torch(
            model,
            DataLoader(TensorDataset(torch.from_numpy(X.to_numpy(np.float32)),
                                     torch.from_numpy(y)),
                       batch_size=batch_size), "ffnn")
        rows[split] = metrics(true, pred)
    return rows


# Runner
def main():
    parts, feats, scaler, frames = load_and_prepare()
    results = {}

    # 1. Ridge linear baseline
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(*frames["train"])
    results["ridge"] = {split: metrics(y, ridge.predict(X))
                        for split, (X, y) in frames.items()}

    # 2. FFNN
    ffnn = FFNN(frames["train"][0].shape[1])
    train_torch(ffnn,
                DataLoader(TensorDataset(torch.from_numpy(frames["train"][0].to_numpy(np.float32)),
                                         torch.from_numpy(frames["train"][1])),
                           batch_size=FFNN_PARAMS["batch_size"], shuffle=True),
                DataLoader(TensorDataset(torch.from_numpy(frames["dev"][0].to_numpy(np.float32)),
                                         torch.from_numpy(frames["dev"][1])),
                           batch_size=1024),
                FFNN_PARAMS, "ffnn")
    results["ffnn"] = ffnn_eval_rows(ffnn, frames)

    # 3. LSTM over 48h raw sequences
    seq_scaler = StandardScaler().fit(
        parts["train"][[c for c in LSTM_SEQ_FEATURES
                        if c not in ("hour_sin", "hour_cos")]])
    seq_train = build_seq_data(parts["train"], seq_scaler)
    seq_dev = build_seq_data(parts["dev"], seq_scaler)
    lstm = LSTMForecaster(len(LSTM_SEQ_FEATURES), len(CITIES))
    train_torch(lstm,
                DataLoader(SeqDataset(seq_train),
                           batch_size=LSTM_PARAMS["batch_size"], shuffle=True),
                DataLoader(SeqDataset(seq_dev), batch_size=1024),
                LSTM_PARAMS, "lstm")
    results["lstm"] = {}
    for split in ("train", "dev", "val"):
        seq = build_seq_data(parts[split], seq_scaler)
        pred, true = predict_torch(
            lstm, DataLoader(SeqDataset(seq), batch_size=2048), "lstm")
        results["lstm"][split] = metrics(true, pred)

    # 4. Persistence baseline: Y(t+6h )forecast = Y(t)
    results["persistence"] = {
        split: metrics(d[TARGET].to_numpy(np.float32),
                       d["temperature_2m"].to_numpy(np.float32))
        for split, d in parts.items()
    }

    print_table(results)
    winner = min(("ridge", "ffnn", "lstm"),
                 key=lambda k: results[k]["val"]["MAE"])
    print(f"\nWinner on validation (most recent 6 months): {winner}")

    # Save model files (and the scalers needed for inference) to model/.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge, MODEL_DIR / "ridge.joblib")
    torch.save(ffnn.state_dict(), MODEL_DIR / "ffnn.pt")
    torch.save(lstm.state_dict(), MODEL_DIR / "lstm.pt")
    joblib.dump({"feature_scaler": scaler,
                 "seq_scaler": seq_scaler,
                 "features": feats,
                 "seq_features": LSTM_SEQ_FEATURES,
                 "seq_len": LSTM_PARAMS["seq_len"]},
                MODEL_DIR / "scaler.joblib")
    print(f"[save] model files -> {MODEL_DIR}/")


if __name__ == "__main__":
    main() 