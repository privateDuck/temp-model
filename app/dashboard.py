import os

import requests
import streamlit as st

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
st.caption(
    "Model: FFNN (256-128-64)"
    "Forecasts are most reliable for the four training cities."
) 