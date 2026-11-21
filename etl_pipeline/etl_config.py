from datetime import date, timedelta

# ---------------------------------------------------------------------------
CITIES = {
    "colombo": {"latitude": 6.9271, "longitude": 79.8612, "country": "Sri Lanka"},
    "galle":   {"latitude": 6.0338, "longitude": 80.2164, "country": "Sri Lanka"},
    "berlin":  {"latitude": 52.5200, "longitude": 13.4050, "country": "Germany"},
    "tokyo":   {"latitude": 35.6895, "longitude": 139.6917, "country": "Japan"},
}


# Data source
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Every city shares one consistent time base
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]

# Rolling window sizes
# 24h windows capture the full diurnal cycle.
REQUESTED_ROLLING_WINDOWS = [6, 24]

# Targets are temperature_2m and shortwave_radiation at t+6h.
FORECAST_HORIZON_HOURS = 6


# Date range: trailing 3 years. The Archive API lags real time by a few days,
# so we stop 7 days back to guarantee complete hourly blocks.
_TODAY = date.today()
START_DATE = (_TODAY - timedelta(days=3 * 365)).isoformat()
END_DATE = (_TODAY - timedelta(days=7)).isoformat()


# File locations (relative to the directory run_pipeline.py is launched from)
CACHE_DB = ".openmeteo_cache"   # HTTP cache for the Open-Meteo client
RAW_DIR = "data/raw"            # one raw Parquet per city (pre-features)
OUTPUT_DIR = "data/processed"   # feature Parquet, partitioned by city
VALIDATION_REPORT = "data/processed/_validation_summary.json" 