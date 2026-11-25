import os

import requests
import streamlit as st
import pandas as pd

API_URL = os.environ.get("API_URL", "http://localhost:8000")
CITY_COORDS = {
    "colombo": (6.9271, 79.8612),
    "galle": (6.0338, 80.2164),
    "berlin": (52.5200, 13.4050),
    "tokyo": (35.6895, 139.6917),
}

st.set_page_config(page_title="6h Temperature Forecast", layout="centered")
st.title("6 Hour Temperature Forecast")

choice = st.selectbox("Location", list(CITY_COORDS) + ["Custom coordinates"])
if choice == "Custom coordinates":
    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", -90.0, 90.0, 6.9271, format="%.4f")
    lon = col2.number_input("Longitude", -180.0, 180.0, 79.8612, format="%.4f")
else:
    lat, lon = CITY_COORDS[choice]
    st.caption(f"latitude {lat}, longitude {lon}")

if st.button("Predict temperature at t+6h", type="primary"):
    with st.spinner("Fetching weather from Open-Meteo and running inference ..."):
        try:
            r = requests.post(
                f"{API_URL}/predict",
                json={"latitude": float(lat), "longitude": float(lon)},
                timeout=120,
            )
        except requests.RequestException as exc:
            st.error(f"Cannot reach the API: {exc}")
            st.stop()

    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        st.error(f"API error {r.status_code}: {detail}")
        st.stop()

    data = r.json()
    delta = data["predicted_temperature_c"] - data["temperature_now_c"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature now", f"{data['temperature_now_c']:.1f} C")
    c2.metric("Forecast t+6h", f"{data['predicted_temperature_c']:.1f} C",
              f"{delta:+.1f} C")
    c3.metric("Inference time", f"{data['inference_ms']:.0f} ms")

    st.caption(
        f"Observed {data['observed_at']} UTC  |  "
        f"forecast valid {data['forecast_time']} UTC  |  "
        f"data fetch {data['fetch_ms']:.0f} ms, total {data['total_ms']:.0f} ms"
    )

    with st.expander("Full API response"):
        st.json(data)

st.divider()
st.subheader("Live Production Monitoring (Automated Evaluation)")

LIVE_LOG_URL = os.environ.get(
    "LIVE_LOG_URL",
    "https://raw.githubusercontent.com/privateDuck/temp-model/main/live/live_predictions.csv",
)


@st.cache_data(ttl=300)
def load_live_metrics(url: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(url)
        return df if not df.empty else None
    except Exception:
        return None


live_df = load_live_metrics(LIVE_LOG_URL)

if live_df is None:
    st.info("No live evaluation logs found yet. Waiting for the first GitHub Actions run.")
else:
    scored = live_df.dropna(subset=["actual_temperature_c", "abs_error_c"]).copy()
    pending_count = len(live_df) - len(scored)

    if scored.empty:
        st.info(
            f"{pending_count} forecast(s) logged and awaiting ground-truth reconciliation "
            "(actuals backfill once the t+6h target timestamp passes)."
        )
    else:
        scored["forecast_time_utc"] = pd.to_datetime(scored["forecast_time_utc"])
        scored = scored.sort_values("forecast_time_utc")

        # Filter by city
        cities = sorted(scored["city"].unique())
        selected_city = st.selectbox("Monitoring city", cities, index=0)
        city_df = scored[scored["city"] == selected_city].copy()

        # Rolling MAE
        window = min(12, len(city_df))
        city_df["rolling_mae_c"] = city_df["abs_error_c"].rolling(window=window, min_periods=1).mean()

        m1, m2, m3 = st.columns(3)
        m1.metric("Reconciled Forecasts", f"{len(city_df)} ({pending_count} pending)")
        m2.metric("Overall Live MAE", f"{city_df['abs_error_c'].mean():.2f} °C")
        m3.metric(f"Rolling MAE (last {window})", f"{city_df['rolling_mae_c'].iloc[-1]:.2f} °C")

        st.caption("Predicted vs. Actual Temperature (°C)")
        temp_chart_df = city_df.set_index("forecast_time_utc")[
            ["predicted_temperature_c", "actual_temperature_c"]
        ].rename(
            columns={
                "predicted_temperature_c": "Predicted (t+6h)",
                "actual_temperature_c": "Actual",
            }
        )
        st.line_chart(temp_chart_df)

        st.caption("Rolling Mean Absolute Error (°C)")
        mae_chart_df = city_df.set_index("forecast_time_utc")[["rolling_mae_c"]].rename(
            columns={"rolling_mae_c": "Rolling MAE (°C)"}
        )
        st.line_chart(mae_chart_df)


st.divider()
st.caption("Model: FFNN (256-128-64) | Forecasts are most reliable for the four training cities.") 