import joblib
import numpy as np
import pandas as pd
import torch

from config import CITIES, FEATURES_DIR, MODEL_DIR, TEST_MONTHS, TRAIN_MONTHS
from models import FFNN

MODEL_PATH = MODEL_DIR / "ffnn.pt"
ARTIFACTS_PATH = MODEL_DIR / "scaler.joblib"


class WeatherForecaster:
    def __init__(self):
        artifacts = joblib.load(ARTIFACTS_PATH)
        self.features = artifacts["features"]          # scaled numeric cols
        self.scaler = artifacts["feature_scaler"]

        # Input layout must match training: scaled features + 4 city dummies
        self.model = FFNN(len(self.features) + len(CITIES))
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()  # BatchNorm must use running stats

    def prepare_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same cleaning as training (train.py.load_and_prepare).

        clear_sky_index is null at night by design; 0 is the physically correct fill
        """
        df = df.copy()
        if "clear_sky_index" in df.columns and "is_daylight" in df.columns:
            df["clear_sky_index"] = df["clear_sky_index"].fillna(pd.Series(np.where(df["is_daylight"] == 0, 0.0, np.nan), index=df.index))
        return df

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Forecast t+6h temperature (degC) for engineered feature rows.

        Returns one value per input row, in input order. Rows with NaN features produce NaN predictions.
        """
        needed = self.features + ["city"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Input rows missing columns: {missing}")

        rows = self.prepare_rows(df).reset_index(drop=True)
        valid = ~rows[self.features].isna().any(axis=1).to_numpy()
        n_bad = int((~valid).sum())
        if n_bad:
            print(f"[warn] {n_bad} row(s) have NaN features after cleaning "
                  f"-> NaN predictions for those rows")

        preds = np.full(len(rows), np.nan)
        if valid.any():
            X_num = self.scaler.transform(
                rows.loc[valid, self.features]).astype(np.float32)
            dummies = (pd.get_dummies(rows.loc[valid, "city"], prefix="city")
                       .reindex(columns=[f"city_{c}" for c in CITIES],
                                fill_value=0)
                       .to_numpy(np.float32))
            X = np.hstack([X_num, dummies])
            with torch.no_grad():
                preds[valid] = self.model(torch.from_numpy(X)).numpy()
        return preds


# Demo
def main():
    forecaster = WeatherForecaster()
    df = pd.read_parquet(FEATURES_DIR).sort_values(["city", "timestamp"])

    # expect ~0.9 MAE
    t0 = df["timestamp"].min()
    val_start = t0 + pd.DateOffset(months=TRAIN_MONTHS + TEST_MONTHS)
    val = forecaster.prepare_rows(df[df["timestamp"] >= val_start])
    val = val.dropna(subset=forecaster.features
                     + ["target_temperature_2m_t6h"])
    preds = forecaster.predict(val)
    mae = np.mean(np.abs(preds - val["target_temperature_2m_t6h"].to_numpy()))
    print(f"[check] validation MAE: {mae:.3f} degC "
          f"({len(val):,} rows, {val['timestamp'].min()} -> "
          f"{val['timestamp'].max()})")

    # latest engineered row per city
    latest = df.groupby("city", group_keys=False).tail(1)
    forecast = forecaster.predict(latest)
    print("\n[forecast] t+6h temperature for the latest available hour:")
    for (_, row), temp in zip(latest.iterrows(), forecast):
        print(f"  {row['city']:8s} {row['timestamp']} UTC -> "
              f"{temp:6.2f} degC")


""" if __name__ == "__main__":
    main()  """