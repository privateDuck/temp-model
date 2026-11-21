"""
Steps
1. fetch_data.main()
2. SparkSession (local[*], UTC time zone)
3. features.build_feature_df()
4. Validation summary writes to data/processed/_validation_summary.json
5. Write Parquet partitioned by city to data/processed/features/
"""

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from etl_config import OUTPUT_DIR, RAW_DIR, VALIDATION_REPORT
from features import FEATURE_COLUMNS, TARGET_COLUMNS, build_feature_df


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.master("local[*]")
        .appName("weather-etl-6h-forecast")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def run_validation(feature_df, output_dir: Path) -> dict:
    """Sanity checks + machine-readable summary for downstream ML work."""
    report = {
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "per_city": {},
    }

    rows = feature_df.groupBy("city").agg(
        F.count("*").alias("n_rows"),
        F.min("timestamp").alias("first_timestamp"),
        F.max("timestamp").alias("last_timestamp"),
        *[
            F.sum(F.col(t).isNull().cast("int")).alias(f"null_{t}")
            for t in TARGET_COLUMNS
        ],
        *[
            F.sum(F.col(c).isNull().cast("int")).alias(f"null_{c}")
            for c in ("temperature_2m", "surface_pressure", "shortwave_radiation")
        ],
    ).orderBy("city")

    total = 0
    for row in rows.collect():
        d = row.asDict()
        total += d["n_rows"]
        # A 6h-lead target must be null for exactly the last 6 hours per city.
        d["target_nulls_explained"] = d[f"null_{TARGET_COLUMNS[0]}"] <= 6
        report["per_city"][d["city"]] = d
        print(
            f"[validate] {d['city']}: {d['n_rows']:,} rows, "
            f"{d['first_timestamp']} -> {d['last_timestamp']}, "
            f"null targets: t={d[f'null_{TARGET_COLUMNS[0]}']}, "
            f"rad={d[f'null_{TARGET_COLUMNS[1]}']}"
        )

    report["total_rows"] = total
    report["null_target_within_bounds"] = all(
        v["target_nulls_explained"] for v in report["per_city"].values()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(VALIDATION_REPORT)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[validate] total rows: {total:,}; summary -> {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Weather ETL pipeline")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="reuse existing data/raw parquet files")
    parser.add_argument("--drop-null-targets", action="store_true",
                        help="drop edge rows where the t+6h label is null")
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--output-dir", default=str(Path(OUTPUT_DIR) / "features"))
    args = parser.parse_args()

    if not args.skip_fetch:
        import fetch_data
        fetch_data.main()
    else:
        print(f"[skip] fetch (using existing files in {args.raw_dir})")

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists() or not any(raw_dir.glob("*.parquet")):
        raise SystemExit(f"No raw parquet files in {raw_dir}; run with fetch enabled.")

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    raw_df = spark.read.parquet(str(raw_dir))
    print(f"[etl] raw rows: {raw_df.count():,} across "
          f"{raw_df.select('city').distinct().count()} cities")

    feature_df = build_feature_df(raw_df, spark)

    if args.drop_null_targets:
        before = feature_df.count()
        feature_df = feature_df.dropna(subset=TARGET_COLUMNS)
        print(f"[etl] dropped {before - feature_df.count():,} rows "
              f"with null t+6h targets (first/last {6}h per city)")

    output_dir = Path(args.output_dir)
    (
        feature_df.repartition("city")
        .write.mode("overwrite")
        .partitionBy("city")
        .parquet(str(output_dir))
    )
    print(f"[etl] wrote partitioned Parquet -> {output_dir}/ "
          f"(city=colombo|galle|berlin|tokyo/)")

    run_validation(feature_df, output_dir)

    spark.stop()


if __name__ == "__main__":
    main() 