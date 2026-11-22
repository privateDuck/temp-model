import dagshub
import mlflow
import mlflow.pyfunc
import pandas as pd

from inference import WeatherForecaster

dagshub.init(
    repo_owner="privateDuck",
    repo_name="temp-model",
    mlflow=True,
)

sample_df = pd.read_parquet('../data/processed/features/').dropna().head(5)

mlflow.set_experiment("openmeteo-forecaster")

with mlflow.start_run(run_name="ffnn-v2"):
    mlflow.log_params({
        "model_type": "FFNN",
        "forecast_horizon": "6h",
    })
    mlflow.log_metrics({
        "val_mae": 0.906,
    })

    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=WeatherForecaster(),
        artifacts={
            "model": str('model/ffnn.pt'),
            "scaler": str('model/scaler.joblib'),
        },
        code_paths=["config.py", "models.py", "inference.py"],
        input_example=sample_df,
        registered_model_name="openmeteo-forecaster",
    )

loaded = mlflow.pyfunc.load_model(model_info.model_uri)
print("Sample predictions:", loaded.predict(sample_df))