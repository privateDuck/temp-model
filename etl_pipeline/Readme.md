# Weather Forecast ETL Temperature & Solar Irradiance

PySpark feature-engineering pipeline for short-term (t+6h) forecasting of
`temperature_2m` and `shortwave_radiation`, built on Open-Meteo Archive API
history for 4 cities:

| City    | Latitude | Longitude  |
|---------|----------|------------|
| Colombo | 6.9271 N | 79.8612 E  |
| Galle   | 6.0338 N | 80.2164 E  |
| Berlin  | 52.5200 N| 13.4050 E  |
| Tokyo   | 35.6895 N| 139.6917 E |

## Pipeline flow

```
open-meteo archive API
        │  fetch_data.py  (openmeteo-requests + cache + retry)
        ▼
data/raw/<city>.parquet
        │  run_pipeline.py -> features.py
        │  Window(partitionBy city, orderBy timestamp)
        │    lag / lead / range-rolling mean & std / UDF solar geometry
        ▼
data/processed/features/city=<name>/*.parquet
```

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py
python run_pipeline.py --skip-fetch --drop-null-targets
```

Output dataset layout: `data/processed/features/city=colombo|galle|berlin|tokyo/`.
Read it back with either engine:

```python
df = spark.read.parquet("data/processed/features")
# or
df = pd.read_parquet("data/processed/features")
```

## Feature dictionary

| Group | Features |
|-------|----------|
| Lags (t-1h, t-3h, t-6h, t-24h) | `temperature_2m_lag_{1,3,6,24}h`, `surface_pressure_lag_{1,3,6,24}h`, `shortwave_radiation_lag_{1,3,6,24}h` |
| Rolling stats (6h, 24h) | `{temperature_2m,surface_pressure,shortwave_radiation}_rolling_{6,24}h_{mean,std}` |
| Cyclical time | `hour_sin/cos` (period 24h), `day_sin/cos` (period 365.25d) |
| Targets | `target_temperature_2m_t6h`, `target_shortwave_radiation_t6h` |

* The 6h rolling window captures weather-front passage (rapid pressure drops,
radiation steps)
* The 24h window captures the full diurnal cycle so the model
sees where the current hour sits relative to the daily norm.

| # | Feature | Group | Why selected |
|---|---------|-------|--------------|
| 1 | `temperature_lag_12h` | lag | Bridges the gap between t-6h and t-24h. A half-day-ago same-phase anchor separates the diurnal cycle from the synoptic trend. |
| 2 | `temperature_lag_48h` | lag | Two days ago, same hour: captures multi-day regimes (heat waves, cold spells) a single t-24h lag cannot resolve. |
| 3 | `surface_pressure_rolling_48h_mean` | rolling | Mid-latitude fronts (Berlin, Tokyo) evolve over 24-72h. A 48h mean exposes sustained pressure tendency behind the 6h/24h windows - directly serves the "pressure drop" signal. |
| 4 | `shortwave_radiation_rolling_3h_mean` | rolling | Recent irradiance smoothed over the forecast horizon itself, 6h/24h windows dilute the most predictive short-term cloud signal. |
| 5 | `relative_humidity_rolling_6h_mean` | rolling | Rising humidity is a leading indicator of cloud formation and precipitation onset. The short window keeps it responsive. |
| 6 | `solar_elevation_angle` | physics | Strongest physical driver of both targets. Gives the model true solar geometry per city/season instead of forcing it to learn it from time encodings alone (NOAA approximation via UDF). |
| 7 | `clear_sky_index` | physics/cross | Measured radiation / Haurwitz clear-sky ceiling: separates "how cloudy" from "how high the sun is", so the signal transfers across cities (Colombo vs Berlin). Null at night/dawn by design. |
| 8 | `temp_delta_3h` | derived | Diurnal rate of change (t - t-3h). Frontal passages and marine breezes show up as abnormal slopes before they appear in levels. |
| 9 | `is_daylight` | derived | Cheap binary (sun above horizon). Lets tree models split day/night radiation behaviour in one node. |
| 10 | `temperature_anomaly_24h` | derived/cross | Signed deviation from the 24h rolling mean: the classic front-detection signal, complementing rolling std which only measures unsigned variability. |



* `temperature_lag_12h`: Bridges the gap between t-6h and t-24h
* `temperature_lag_48h`: Two days ago, same hour: captures slowly evolving multi-day regimes
* `surface_pressure_rolling_48h_mean`: Mid-latitude fronts (Berlin, Tokyo) evolve over 24-72h
* `shortwave_radiation_rolling_3h_mean`: Recent irradiance smoothed over the forecast horizon itself
* `relative_humidity_rolling_6h_mean`: Rising humidity is a leading indicator of cloud formation and precipitation onset
* `solar_elevation_angle`: Strongest physical driver of both targets
* `clear_sky_index`: Measured radiation divided by the theoretical clear-sky ceiling (Haurwitz model): separates 'how cloudy' from 'how high the sun is', so the signal transfers across cities (Colombo vs Berlin)
* `temp_delta_3h`: Diurnal rate of change (t - t-3h). Frontal passages and marine breezes show up as abnormal slopes before they appear in levels
* `is_daylight`: Sun above horizon. Lets tree models split day/night
* `temperature_anomaly_24h`: Signed deviation from the 24h rolling mean: the classic front-detection signal, complementing rolling std which only measures unsigned variability


## Design notes

- **UTC everywhere.** Data is fetched in UTC and `spark.sql.session.timeZone`
  is pinned to UTC, so `hour()`, `dayofyear()` and all window boundaries are
  deterministic regardless of the host machine. True local solar geometry is
  recovered physically via `solar_elevation_angle` (longitude-corrected), so
  no per-city timezone column is needed.
- **Time-range rolling windows.** Rolling stats use
  `rangeBetween(-(n-1)*3600, 0)` on unix seconds, not row positions. Correct
  even if the source data has missing hours.
- **Nulls.** Lags/rolls/targets are null at series edges. `clear_sky_index`
  is null at night. Keep them (tree models / LightGBM handle nulls natively)
  or run with `--drop-null-targets` for training-frame assembly.
- **Partitioning.** Output is written `partitionBy("city")` — the standard
  Spark layout, directly usable as a group key for city-aware train/test
  splits (e.g. split by time *within* each city, never randomly across the
  whole dataset, to avoid temporal leakage).
- **Next steps (out of scope here).** sklearn baseline (HistGradientBoosting
  per target) and a PyTorch LSTM/TCN consuming the same Parquet. Both read
  `FEATURE_COLUMNS`/`TARGET_COLUMNS` from `features.py`. 