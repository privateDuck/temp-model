import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

MODAL_ENDPOINT = os.environ["MODAL_ENDPOINT_URL"]  # .../predict
LOG_PATH = Path("data/live_predictions.csv")

CITY_COORDS = {
    "colombo": (6.9271, 79.8612),
    "galle": (6.0338, 80.2164),
    "berlin": (52.5200, 13.4050),
    "tokyo": (35.6895, 139.6917),
}

COLUMNS = [
    "logged_at_utc",
    "city",
    "observed_at_utc",
    "forecast_time_utc",
    "temperature_now_c",
    "predicted_temperature_c",
    "actual_temperature_c",
    "abs_error_c",
    "inference_ms",
    "total_ms",
]


def load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    return pd.DataFrame(columns=COLUMNS)


def fetch_actuals(lat: float, lon: float) -> dict[str, float]:
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly=temperature_2m"
        f"&past_days=2&forecast_days=1&timezone=UTC"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    return {
        t: temp
        for t, temp in zip(hourly["time"], hourly["temperature_2m"])
        if temp is not None
    }


def backfill_actuals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not df["actual_temperature_c"].isna().any():
        return df

    pending_cities = df.loc[df["actual_temperature_c"].isna(), "city"].unique()

    for city in pending_cities:
        if city not in CITY_COORDS:
            continue
        lat, lon = CITY_COORDS[city]
        actual_map = fetch_actuals(lat, lon)

        mask = (df["city"] == city) & (df["actual_temperature_c"].isna())
        for idx in df[mask].index:
            # Normalize timestamp format to Open-Meteo's "YYYY-MM-DDTHH:00"
            target_dt = pd.to_datetime(df.at[idx, "forecast_time_utc"], utc=True)
            target_key = target_dt.strftime("%Y-%m-%dT%H:00")

            if target_key in actual_map:
                actual = float(actual_map[target_key])
                pred = float(df.at[idx, "predicted_temperature_c"])
                df.at[idx, "actual_temperature_c"] = round(actual, 2)
                df.at[idx, "abs_error_c"] = round(abs(pred - actual), 3)

    return df


def log_new_predictions(df: pd.DataFrame) -> pd.DataFrame:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_rows = []

    for city, (lat, lon) in CITY_COORDS.items():
        resp = requests.post(
            MODAL_ENDPOINT,
            json={"latitude": float(lat), "longitude": float(lon)},
            timeout=120,  # Allows cold-start container spin-up
        )
        resp.raise_for_status()
        data = resp.json()

        new_rows.append(
            {
                "logged_at_utc": now_utc,
                "city": city,
                "observed_at_utc": data["observed_at"],
                "forecast_time_utc": data["forecast_time"],
                "temperature_now_c": round(float(data["temperature_now_c"]), 2),
                "predicted_temperature_c": round(float(data["predicted_temperature_c"]), 2),
                "actual_temperature_c": None,
                "abs_error_c": None,
                "inference_ms": round(float(data["inference_ms"]), 2),
                "total_ms": round(float(data["total_ms"]), 2),
            }
        )

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def main():
    df = load_log()
    df = backfill_actuals(df)
    df = log_new_predictions(df)
    df.to_csv(LOG_PATH, index=False)

    scored = df.dropna(subset=["abs_error_c"])
    print(f"Total logged: {len(df)} | Reconciled with ground truth: {len(scored)}")
    if not scored.empty:
        print(f"Live Production MAE: {scored['abs_error_c'].mean():.3f} °C")


if __name__ == "__main__":
    main()