import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from config import CITIES
from models import FFNN


class WeatherForecaster(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Called once when mlflow.pyfunc.load_model() runs."""
        artifacts = joblib.load(context.artifacts["scaler"])
        self.features = artifacts["features"]          # scaled numeric cols
        self.scaler = artifacts["feature_scaler"]

        # Input layout must match training: scaled features + city dummies
        self.model = FFNN(len(self.features) + len(CITIES))
        self.model.load_state_dict(
            torch.load(context.artifacts["model"], map_location="cpu")
        )
        self.model.eval()  # BatchNorm must use running stats

    def prepare_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same cleaning as training (train.py.load_and_prepare).

        clear_sky_index is null at night by design; 0 is the physically correct fill.
        """
        df = df.copy()
        if "clear_sky_index" in df.columns and "is_daylight" in df.columns:
            df["clear_sky_index"] = df["clear_sky_index"].fillna(
                pd.Series(
                    np.where(df["is_daylight"] == 0, 0.0, np.nan),
                    index=df.index,
                )
            )
        return df

    def predict(
        self, context, model_input: pd.DataFrame, params=None
    ) -> np.ndarray:
        """Forecast t+6h temperature (degC) for engineered feature rows.

        Returns one value per input row, in input order. Rows with NaN features
        produce NaN predictions.
        """
        needed = self.features + ["city"]
        missing = [c for c in needed if c not in model_input.columns]
        if missing:
            raise ValueError(f"Input rows missing columns: {missing}")

        rows = self.prepare_rows(model_input).reset_index(drop=True)
        valid = ~rows[self.features].isna().any(axis=1).to_numpy()
        n_bad = int((~valid).sum())
        if n_bad:
            print(
                f"[warn] {n_bad} row(s) have NaN features after cleaning "
                f"-> NaN predictions for those rows"
            )

        preds = np.full(len(rows), np.nan)
        if valid.any():
            X_num = self.scaler.transform(
                rows.loc[valid, self.features]
            ).astype(np.float32)
            dummies = (
                pd.get_dummies(rows.loc[valid, "city"], prefix="city")
                .reindex(columns=[f"city_{c}" for c in CITIES], fill_value=0)
                .to_numpy(np.float32)
            )
            X = np.hstack([X_num, dummies])
            with torch.no_grad():
                preds[valid] = self.model(torch.from_numpy(X)).numpy()
        return preds