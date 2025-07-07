# register_model.py
import mlflow
import joblib
from mlflow.sklearn import log_model

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("salary_prediction_pipeline")

model_path = "dags/artifacts/best_pipeline.pkl"
model = joblib.load(model_path)

with mlflow.start_run(run_name="Register_Best_Model"):
    mlflow.sklearn.log_model(
        model,
        artifact_path="best_model",
        registered_model_name="Best_Salary_Predictor"
    )
    print("✅ Model registered in MLflow.")
