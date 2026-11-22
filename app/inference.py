import math

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mlflow.pyfunc

# --- constants the saved model carries with it (no local imports needed) ---
CITIES = ["colombo", "galle", "berlin", "tokyo"]
FFNN_HIDDEN = (256, 128, 64)
FFNN_DROPOUT = (0.20, 0.15, 0.10)

# Coordinates + variable order for the live fetch (sync with the ETL).
CITY_COORDS = {
    "colombo": (6.9271, 79.8612),
    "galle": (6.0338, 80.2164),
    "berlin": (52.5200, 13.4050),
    "tokyo": (35.6895, 139.6917),
}
HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
               "cloud_cover", "wind_speed_10m", "shortwave_radiation"]
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"


class _FFNN(nn.Module):
    """Same architecture as models.FFNN, inlined for portability."""

    def __init__(self, input_dim: int):
        super().__init__()
        dims = (input_dim, *FFNN_HIDDEN)
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(FFNN_DROPOUT[i]),
            ]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class WeatherForecaster(mlflow.pyfunc.PythonModel):
    """Trained FFNN + preprocessing, packaged as an MLflow pyfunc model.

    The API below loads the already-logged model from the tracking server;
    this class definition is only needed so unpickling works if the pickle
    references module `inference`.
    """

    def load_context(self, context):
        """Called once when mlflow.pyfunc.load_model() runs."""
        artifacts = joblib.load(context.artifacts["scaler"])
        self.features = artifacts["features"]          # scaled numeric cols
        self.scaler = artifacts["feature_scaler"]

        # Input layout must match training: scaled features + city dummies.
        self.model = _FFNN(len(self.features) + len(CITIES))
        self.model.load_state_dict(
            torch.load(context.artifacts["model"], map_location="cpu")
        )
        self.model.eval()  # BatchNorm must use running stats

    def prepare_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same cleaning as training.

        clear_sky_index is null at night by design; 0 is the correct fill.
        """
        df = df.copy()
        if "clear_sky_index" in df.columns and "is_daylight" in df.columns:
            df.loc[df["is_daylight"] == 0, "clear_sky_index"] = 0.0
        return df

    def predict(self, context, model_input: pd.DataFrame,
                params=None) -> np.ndarray:
        """Forecast t+6h temperature (degC) for engineered feature rows.

        Returns one value per input row, in input order. Rows with NaN
        features produce NaN predictions.
        """
        needed = self.features + ["city"]
        missing = [c for c in needed if c not in model_input.columns]
        if missing:
            raise ValueError(f"Input rows missing columns: {missing}")

        rows = self.prepare_rows(model_input).reset_index(drop=True)
        valid = ~rows[self.features].isna().any(axis=1).to_numpy()
        n_bad = int((~valid).sum())
        if n_bad:
            print(f"[warn] {n_bad} row(s) have NaN features after cleaning "
                  f"-> NaN predictions for those rows")

        preds = np.full(len(rows), np.nan)
        if valid.any():
            X_num = self.scaler.transform(
                rows.loc[valid, self.features]).astype(np.float32)
            dummies = (
                pd.get_dummies(rows.loc[valid, "city"], prefix="city")
                .reindex(columns=[f"city_{c}" for c in CITIES], fill_value=0)
                .to_numpy(np.float32)
            )
            X = np.hstack([X_num, dummies])
            with torch.no_grad():
                preds[valid] = self.model(torch.from_numpy(X)).numpy()
        return preds


# ---------------------------------------------------------------------------
# Raw Open-Meteo data -> engineered features (pandas port of the ETL logic)
# ---------------------------------------------------------------------------
def _solar_elevation(ts: pd.Timestamp, lat: float, lon: float) -> float:
    """Solar elevation in degrees. Same NOAA approximation as the ETL."""
    ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    doy = ts.timetuple().tm_yday
    hour_frac = ts.hour + ts.minute / 60.0 + ts.second / 3600.0

    fy = 2.0 * math.pi / 365.0 * (doy - 1 + (hour_frac - 12.0) / 24.0)
    eqtime = 229.75 * (0.000075 + 0.001868 * math.cos(fy)
                       - 0.032077 * math.sin(fy)
                       - 0.014615 * math.cos(2 * fy)
                       - 0.040849 * math.sin(2 * fy))
    decl = (0.006918 - 0.399912 * math.cos(fy) + 0.070257 * math.sin(fy)
            - 0.006758 * math.cos(2 * fy) + 0.000907 * math.sin(2 * fy)
            - 0.002697 * math.cos(3 * fy) + 0.001480 * math.sin(3 * fy))
    tst = hour_frac * 60.0 + eqtime + 4.0 * lon  # lon east positive, UTC
    ha = math.radians(tst / 4.0 - 180.0)
    lat_r = math.radians(lat)
    cos_zen = (math.sin(lat_r) * math.sin(decl)
               + math.cos(lat_r) * math.cos(decl) * math.cos(ha))
    cos_zen = max(-1.0, min(1.0, cos_zen))
    return 90.0 - math.degrees(math.acos(cos_zen))


def engineer_features(raw: pd.DataFrame, city: str, lat: float,
                      lon: float) -> pd.DataFrame:
    """Pandas port of the ETL's PySpark feature engineering (no targets).

    raw: one location's hourly rows with `timestamp` (UTC) and the six
         HOURLY_VARS columns. Needs >= 49h of history for the 48h windows.
    """
    if len(raw) < 49:
        raise ValueError("need >= 49h of hourly history for the 48h features")

    df = raw.sort_values("timestamp").reset_index(drop=True).copy()
    df["city"] = city
    df["latitude"] = float(lat)
    df["longitude"] = float(lon)

    # Lags: t-1/3/6/24h for the three core vars, plus t-12/48h temperature.
    for var, lags in {"temperature_2m": [1, 3, 6, 24, 12, 48],
                      "surface_pressure": [1, 3, 6, 24],
                      "shortwave_radiation": [1, 3, 6, 24]}.items():
        for h in lags:
            df[f"{var}_lag_{h}h"] = df[var].shift(h)

    # Time-based rolling windows: (t-h, t] mirrors the Spark
    # rangeBetween(-(h-1)*3600, 0) frames; std uses ddof=1 like Spark.
    df = df.set_index("timestamp")
    for var in ("temperature_2m", "surface_pressure", "shortwave_radiation"):
        for h in (6, 24):
            r = df[var].rolling(f"{h}h", min_periods=h)
            df[f"{var}_rolling_{h}h_mean"] = r.mean()
            df[f"{var}_rolling_{h}h_std"] = r.std()
    df["surface_pressure_rolling_48h_mean"] = df["surface_pressure"].rolling(
        "48h", min_periods=48).mean()
    df["shortwave_radiation_rolling_3h_mean"] = df["shortwave_radiation"].rolling(
        "3h", min_periods=3).mean()
    df["relative_humidity_2m_rolling_6h_mean"] = df["relative_humidity_2m"].rolling(
        "6h", min_periods=6).mean()
    df = df.reset_index()

    # Cyclical time encodings (UTC).
    hours = df["timestamp"].dt.hour.astype(float)
    doy = df["timestamp"].dt.dayofyear.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["day_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Solar geometry + physics features.
    df["solar_elevation_angle"] = [
        _solar_elevation(ts, lat, lon) for ts in df["timestamp"]]
    elev = df["solar_elevation_angle"].to_numpy()
    cos_zen = np.maximum(np.cos(np.radians(90.0 - elev)), 1e-6)
    clear_sky = np.where(elev > 3.0,
                         1098.0 * cos_zen * np.exp(-0.057 / cos_zen), 0.0)
    df["clear_sky_index"] = np.where(
        clear_sky >= 20.0,
        df["shortwave_radiation"] / np.maximum(clear_sky, 1e-6), np.nan)
    df["is_daylight"] = (elev > 0.0).astype(int)
    df["temp_delta_3h"] = df["temperature_2m"] - df["temperature_2m_lag_3h"]
    df["temperature_anomaly_24h"] = (df["temperature_2m"]
                                     - df["temperature_2m_rolling_24h_mean"])
    return df


def fetch_recent_open_meteo(lat: float, lon: float,
                            past_days: int = 4) -> pd.DataFrame:
    """Fetch the last few days of hourly weather (UTC) from the Open-Meteo
    Forecast API (the Archive API does not have the latest hours yet)."""
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    cache = requests_cache.CachedSession(".openmeteo_cache",
                                         expire_after=600)
    client = openmeteo_requests.Client(
        session=retry(cache, retries=5, backoff_factor=0.2))
    params = {
        "latitude": lat, "longitude": lon, "hourly": HOURLY_VARS,
        "timezone": "UTC", "past_days": past_days, "forecast_days": 1,
    }
    hourly = client.weather_api(FORECAST_API_URL, params=params)[0].Hourly()

    index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left")
    data = {"timestamp": index}
    for i, name in enumerate(HOURLY_VARS):
        data[name] = hourly.Variables(i).ValuesAsNumpy().astype("float64")
    return pd.DataFrame(data) 