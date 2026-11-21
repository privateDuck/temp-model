"""Fetch 3 years of hourly weather history for each city from the
Open-Meteo Archive API, using the `openmeteo-requests` client package.

Output: one raw Parquet file per city under data/raw/
    data/raw/colombo.parquet
    data/raw/galle.parquet
    data/raw/berlin.parquet
    data/raw/tokyo.parquet

Each file has columns: timestamp (UTC datetime), city, and the six
hourly variables from config.HOURLY_VARIABLES.
"""

import time
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from etl_config import (
    ARCHIVE_URL,
    CACHE_DB,
    CITIES,
    END_DATE,
    HOURLY_VARIABLES,
    RAW_DIR,
    START_DATE,
)


def build_client() -> openmeteo_requests.Client:
    """Open-Meteo recommended client: disk-cached HTTP + automatic retries."""
    cache_session = requests_cache.CachedSession(CACHE_DB, expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_city(
    client: openmeteo_requests.Client, city_name: str, city_cfg: dict
) -> pd.DataFrame:
    """Fetch one city's hourly history and return a tidy DataFrame."""
    params = {
        "latitude": city_cfg["latitude"],
        "longitude": city_cfg["longitude"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": HOURLY_VARIABLES,
        # UTC for all cities: one time base, no DST surprises, and the
        # true solar geometry is recovered later via longitude (features.py).
        "timezone": "UTC",
    }

    response = client.weather_api(ARCHIVE_URL, params=params)[0]
    hourly = response.Hourly()

    # Build the UTC hourly index from the response's own timestamps, so the
    # index always matches the values even if Open-Meteo trims the range.
    index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    data = {"timestamp": index}
    var_index = {name: i for i, name in enumerate(HOURLY_VARIABLES)}
    for name in HOURLY_VARIABLES:
        # openmeteo-requests returns float32; promote to float64 so Spark
        # reads double precision and rolling std stays numerically stable.
        data[name] = (
            hourly.Variables(var_index[name]).ValuesAsNumpy().astype("float64")
        )

    df = pd.DataFrame(data)
    df.insert(1, "city", city_name)
    return df


def main() -> None:
    raw_dir = Path(RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    client = build_client()

    for city_name, city_cfg in CITIES.items():
        out_path = raw_dir / f"{city_name}.parquet"
        if out_path.exists():
            print(f"[skip] {city_name}: {out_path} already exists "
                  f"(delete it to force a re-fetch)")
            continue

        print(f"[fetch] {city_name} ({city_cfg['country']}) "
              f"{START_DATE} -> {END_DATE} ...", flush=True)
        df = fetch_city(client, city_name, city_cfg)
        df.to_parquet(out_path, index=False)
        print(f"[done]  {city_name}: {len(df):,} rows "
              f"({df['timestamp'].min()} -> {df['timestamp'].max()}) "
              f"-> {out_path}")

        # Be gentle with the free API tier between cities.
        time.sleep(1.0)


if __name__ == "__main__":
    main() 