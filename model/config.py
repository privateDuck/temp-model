from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEATURES_DIR = "../data/processed/features"
MODEL_DIR = ROOT / "model"

# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
TARGET = "target_temperature_2m_t6h"

# first 24 months -> train, next 6 months -> dev (early stopping),
# remainder (~6 months, most recent) -> validation holdout.
TRAIN_MONTHS = 24
TEST_MONTHS = 6

SEED = 42
RIDGE_ALPHA = 1.0  # fixed, no tuning

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
CITIES = ["colombo", "galle", "berlin", "tokyo"]

# Same list the ETL produces (features.FEATURE_COLUMNS). Filtered to the
# columns actually present in the Parquet at load time.
FEATURE_COLUMNS = [
    *[f"{var}_lag_{h}h"
      for var in ("temperature_2m", "surface_pressure", "shortwave_radiation")
      for h in (1, 3, 6, 24)],
    "temperature_2m_lag_12h", "temperature_2m_lag_48h",
    *[f"{var}_rolling_{h}h_{stat}"
      for var in ("temperature_2m", "surface_pressure", "shortwave_radiation")
      for h in (6, 24) for stat in ("mean", "std")],
    "surface_pressure_rolling_48h_mean",
    "shortwave_radiation_rolling_3h_mean",
    "relative_humidity_2m_rolling_6h_mean",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "solar_elevation_angle", "clear_sky_index", "temp_delta_3h",
    "is_daylight", "temperature_anomaly_24h",
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "cloud_cover", "wind_speed_10m", "shortwave_radiation",
]
EXCLUDED_FROM_MODEL = ["hour_of_day", "day_of_year"]  # info lives in sin/cos
EXTRA_STATIC = ["latitude", "longitude"]              # added to the model

# Raw hourly channels the LSTM reads as an ordered 48h sequence
# (hour_sin/cos already in [-1, 1], passed through unscaled).
LSTM_SEQ_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "cloud_cover", "wind_speed_10m", "shortwave_radiation",
    "hour_sin", "hour_cos",
]

# ---------------------------------------------------------------------------
# Model hyperparameters (fixed; no tuning)
# ---------------------------------------------------------------------------
FFNN_PARAMS = {
    "hidden": (256, 128, 64),
    "dropout": (0.20, 0.15, 0.10),
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 200,
    "patience": 15,          # early stopping on dev MAE
}

LSTM_PARAMS = {
    "seq_len": 48,           # two diurnal cycles
    "hidden": 64,
    "layers": 2,
    "dropout": 0.2,
    "head_hidden": 32,
    "batch_size": 128,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 1,
    "patience": 8,
} 