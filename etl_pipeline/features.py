import math
from datetime import datetime, timezone

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf

from etl_config import CITIES, FORECAST_HORIZON_HOURS, REQUESTED_ROLLING_WINDOWS


# Feature specifications
LAG_SPECS = {
    "temperature_2m": [1, 3, 6, 24],
    "surface_pressure": [1, 3, 6, 24],
    "shortwave_radiation": [1, 3, 6, 24],
}

EXTRA_LAG_SPECS = {"temperature_2m": [12, 48]}
EXTRA_ROLLING_SPECS = {
    "surface_pressure": [48],
    "shortwave_radiation": [3],
    "relative_humidity_2m": [6],
}

ROLLING_STATS_VARS = ["temperature_2m", "surface_pressure", "shortwave_radiation"]


# Solar geometry (NOAA local approximation, evaluated via a Python UDF)
def _solar_elevation(epoch_seconds: int, lat: float, lon: float) -> float:
    """Solar elevation in degrees for a UTC epoch, latitude, longitude (deg E).

    Simplified NOAA solar position: fractional-year Fourier terms for the
    equation of time and solar declination. Accuracy ~0.1-0.3 deg, far below
    the resolution a forecaster needs. Vectorised in Spark through a UDF.
    """
    ts = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    day_of_year = ts.timetuple().tm_yday
    hour_frac = ts.hour + ts.minute / 60.0 + ts.second / 3600.0

    fy = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hour_frac - 12.0) / 24.0)
    eqtime = 229.75 * (
        0.000075
        + 0.001868 * math.cos(fy)
        - 0.032077 * math.sin(fy)
        - 0.014615 * math.cos(2 * fy)
        - 0.040849 * math.sin(2 * fy)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(fy)
        + 0.070257 * math.sin(fy)
        - 0.006758 * math.cos(2 * fy)
        + 0.000907 * math.sin(2 * fy)
        - 0.002697 * math.cos(3 * fy)
        + 0.001480 * math.sin(3 * fy)
    )
    # True solar time: longitude correction in minutes
    time_offset = eqtime + 4.0 * lon
    tst = hour_frac * 60.0 + time_offset
    hour_angle = math.radians(tst / 4.0 - 180.0)

    lat_r = math.radians(lat)
    cos_zenith = (
        math.sin(lat_r) * math.sin(decl)
        + math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle)
    )
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    return 90.0 - math.degrees(math.acos(cos_zenith))


solar_elevation_udf = udf(_solar_elevation, DoubleType())


# Window helpers
def _ordered_window() -> Window:
    """Row-order window per city: lags and lead targets."""
    return Window.partitionBy("city").orderBy("timestamp")


def _rolling_window(hours: int) -> Window:
    """Time-based rolling window: all rows in (t - (hours-1)h, t].

    rangeBetween on unix seconds (not row positions) keeps the window correct
    even if a few hours are missing from the source data.
    """
    return (
        Window.partitionBy("city")
        .orderBy("ts_unix")
        .rangeBetween(-(hours - 1) * 3600, 0)
    )


def _apply_lags(df: DataFrame, specs: dict) -> DataFrame:
    for var, lag_hours in specs.items():
        for h in lag_hours:
            df = df.withColumn(f"{var}_lag_{h}h", F.lag(var, h).over(_ordered_window()))
    return df


def _apply_rolling_stats(df: DataFrame) -> DataFrame:
    windows = {}

    def window_for(hours: int) -> Window:
        if hours not in windows:
            windows[hours] = _rolling_window(hours)
        return windows[hours]

    # Requested: mean + std over each requested window for the core variables.
    for var in ROLLING_STATS_VARS:
        for hours in REQUESTED_ROLLING_WINDOWS:
            w = window_for(hours)
            df = df.withColumn(
                f"{var}_rolling_{hours}h_mean", F.mean(var).over(w)
            ).withColumn(f"{var}_rolling_{hours}h_std", F.stddev(var).over(w))

    # Extra rolling means
    for var, hours_list in EXTRA_ROLLING_SPECS.items():
        for hours in hours_list:
            df = df.withColumn(
                f"{var}_rolling_{hours}h_mean", F.mean(var).over(window_for(hours))
            )
    return df


def _apply_cyclical_time(df: DataFrame) -> DataFrame:
    df = (
        df.withColumn("hour_of_day", F.hour("timestamp").cast("double"))
        .withColumn("day_of_year", F.dayofyear("timestamp").cast("double"))
        .withColumn("hour_sin", F.sin(2 * math.pi * F.col("hour_of_day") / 24.0))
        .withColumn("hour_cos", F.cos(2 * math.pi * F.col("hour_of_day") / 24.0))
        .withColumn("day_sin", F.sin(2 * math.pi * F.col("day_of_year") / 365.25))
        .withColumn("day_cos", F.cos(2 * math.pi * F.col("day_of_year") / 365.25))
    )
    return df


def _apply_physics_features(df: DataFrame, cities_df: DataFrame) -> DataFrame:
    """Join per-city coordinates, add solar elevation and derived features."""
    df = df.join(cities_df, on="city", how="left")

    df = df.withColumn(
        "solar_elevation_angle",
        solar_elevation_udf(F.col("ts_unix"), F.col("latitude"), F.col("longitude")),
    )

    # Haurwitz clear-sky GHI (W/m^2)
    zenith = F.radians(90.0 - F.col("solar_elevation_angle"))
    cos_zenith = F.cos(zenith)
    clear_sky_ghi = F.when(
        F.col("solar_elevation_angle") > 3.0,
        F.lit(1098.0) * cos_zenith * F.exp(F.lit(-0.057) / cos_zenith),
    ).otherwise(F.lit(0.0))

    # Clear-sky index is undefined at night / dawn -> null (models can impute or drop
    df = df.withColumn("_clear_sky_ghi", clear_sky_ghi)
    df = df.withColumn(
        "clear_sky_index",
        F.when(F.col("_clear_sky_ghi") >= 20.0,
               F.col("shortwave_radiation") / F.col("_clear_sky_ghi"))
        .otherwise(F.lit(None).cast("double")),
    ).drop("_clear_sky_ghi")

    df = df.withColumn("is_daylight",
                       (F.col("solar_elevation_angle") > 0.0).cast("int"))
    return df


def _apply_derived_features(df: DataFrame) -> DataFrame:
    df = df.withColumn(
        "temp_delta_3h",
        F.col("temperature_2m") - F.col("temperature_2m_lag_3h"),
    )
    df = df.withColumn(
        "temperature_anomaly_24h",
        F.col("temperature_2m") - F.col("temperature_2m_rolling_24h_mean"),
    )
    return df


def _apply_targets(df: DataFrame) -> DataFrame:
    w = _ordered_window()
    df = df.withColumn(
        f"target_temperature_2m_t{FORECAST_HORIZON_HOURS}h",
        F.lead("temperature_2m", FORECAST_HORIZON_HOURS).over(w),
    ).withColumn(
        f"target_shortwave_radiation_t{FORECAST_HORIZON_HOURS}h",
        F.lead("shortwave_radiation", FORECAST_HORIZON_HOURS).over(w),
    )
    return df



# Public entry point
FEATURE_COLUMNS = [
    *[f"{var}_lag_{h}h" for var, hs in LAG_SPECS.items() for h in hs],
    *[f"{var}_lag_{h}h" for var, hs in EXTRA_LAG_SPECS.items() for h in hs],
    *[
        f"{var}_rolling_{h}h_{stat}"
        for var in ROLLING_STATS_VARS
        for h in REQUESTED_ROLLING_WINDOWS
        for stat in ("mean", "std")
    ],
    *[
        f"{var}_rolling_{h}h_mean"
        for var, hs in EXTRA_ROLLING_SPECS.items()
        for h in hs
    ],
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "solar_elevation_angle", "clear_sky_index", "temp_delta_3h",
    "is_daylight", "temperature_anomaly_24h",
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "cloud_cover", "wind_speed_10m", "shortwave_radiation",
    "hour_of_day", "day_of_year",
]

TARGET_COLUMNS = [
    f"target_temperature_2m_t{FORECAST_HORIZON_HOURS}h",
    f"target_shortwave_radiation_t{FORECAST_HORIZON_HOURS}h",
]


def build_feature_df(df: DataFrame, spark) -> DataFrame:
    """Run the full feature engineering chain on a raw hourly DataFrame.

    Expects columns: timestamp, city, and the variables in HOURLY_VARIABLES.
    """
    cities_df = spark.createDataFrame(
        [
            (name, float(cfg["latitude"]), float(cfg["longitude"]))
            for name, cfg in CITIES.items()
        ],
        schema=["city", "latitude", "longitude"],
    )

    df = df.withColumn("ts_unix", F.unix_timestamp("timestamp"))
    for var in ("temperature_2m", "relative_humidity_2m", "surface_pressure",
                "cloud_cover", "wind_speed_10m", "shortwave_radiation"):
        df = df.withColumn(var, F.col(var).cast("double"))

    df = _apply_lags(df, LAG_SPECS)
    df = _apply_lags(df, EXTRA_LAG_SPECS)
    df = _apply_rolling_stats(df)
    df = _apply_cyclical_time(df)
    df = _apply_physics_features(df, cities_df)
    df = _apply_derived_features(df)
    df = _apply_targets(df)

    return df.drop("ts_unix") 