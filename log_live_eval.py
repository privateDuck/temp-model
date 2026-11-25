import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests

MODAL_ENDPOINT = os.environ["MODAL_ENDPOINT_URL"]
LOG_PATH = Path("live/live_predictions.csv")

CITY_COORDS = {
    "Colombo": (6.9271, 79.8612),
    "London": (51.5074, -0.1278),
    "Tokyo": (35.6762, 139.6503),
    "New_York": (40.7128, -74.0060),
}

COLUMNS = [
    "logged_at_utc",
    "city",
    "target_timestamp_utc",
    "predicted_temp_c",
    "actual_temp_c",
    "abs_error_c",
    "api_latency_ms",
]


def load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    return pd.DataFrame(columns=COLUMNS)


def fetch_actuals(lat: float, lon: float) -> dict[str, float]:
    """Fetch the last 24 hours of observed hourly temperatures from Open-Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly=temperature_2m"
        f"&past_days=1&forecast_days=1&timezone=UTC"
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
    """Fill in actual temperatures and absolute error for predictions whose target hour has passed."""
    if df.empty or not df["actual_temp_c"].isna().any():
        return df

    now_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    pending_cities = df.loc[df["actual_temp_c"].isna(), "city"].unique()

    for city in pending_cities:
        if city not in CITY_COORDS:
            continue
        lat, lon = CITY_COORDS[city]
        actual_map = fetch_actuals(lat, lon)

        mask = (df["city"] == city) & (df["actual_temp_c"].isna())
        for idx in df[mask].index:
            target_str = df.at[idx, "target_timestamp_utc"]  # format: YYYY-MM-DDTHH:00
            target_dt = datetime.strptime(target_str, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
            if target_dt <= now_hour and target_str in actual_map:
                actual = float(actual_map[target_str])
                pred = float(df.at[idx, "predicted_temp_c"])
                df.at[idx, "actual_temp_c"] = round(actual, 2)
                df.at[idx, "abs_error_c"] = round(abs(pred - actual), 3)

    return df


def log_new_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Call the live Modal endpoint for each city and append the new t+6h forecast."""
    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.replace(minute=0, second=0, microsecond=0)
    target_hour = (current_hour + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M")

    new_rows = []
    for city, (lat, lon) in CITY_COORDS.items():
        resp = requests.post(
            MODAL_ENDPOINT,
            json={"latitude": lat, "longitude": lon, "city": city},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()

        # Adjust key names below to match your exact Modal endpoint JSON response
        pred_temp = payload.get("forecast_6h_temp_c") or payload.get("prediction")
        latency = payload.get("inference_latency_ms", None)

        new_rows.append(
            {
                "logged_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "city": city,
                "target_timestamp_utc": target_hour,
                "predicted_temp_c": round(float(pred_temp), 2),
                "actual_temp_c": None,
                "abs_error_c": None,
                "api_latency_ms": latency,
            }
        )

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def main():
    df = load_log()
    df = backfill_actuals(df)
    df = log_new_predictions(df)
    df.to_csv(LOG_PATH, index=False)

    scored = df.dropna(subset=["abs_error_c"])
    print(f"Total logged predictions: {len(df)} | Reconciled with ground truth: {len(scored)}")
    if not scored.empty:
        print(f"Live Production MAE: {scored['abs_error_c'].mean():.3f} °C")


if __name__ == "__main__":
    main()