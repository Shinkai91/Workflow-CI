import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import os
import joblib

# Inisialisasi koneksi ke DagsHub (ganti dengan repo-mu)
dagshub.init(repo_owner='Shinkai91', repo_name='air-quality-model', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Shinkai91/air-quality-model.mlflow")
mlflow.set_experiment("air-quality-prediction")

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--model_output", type=str, required=True)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Load data
data = pd.read_csv(args.data_path)
data = data.drop(columns=["Unnamed: 15", "Unnamed: 16"], errors="ignore")

X = data.drop(columns=["AH"])
y = data["AH"]

X = pd.get_dummies(X, drop_first=True)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# Autologging
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestRegressor(random_state=args.random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    evs = explained_variance_score(y_test, predictions)
    rmse = np.sqrt(mse)

    # Manual logging
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("evs", evs)

    mlflow.log_param("data_path", args.data_path)
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    mlflow.log_param("features_count", X_train.shape[1])

    # Save model
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    joblib.dump(model, args.model_output)
    mlflow.log_artifact(args.model_output)

    print("✅ Model training selesai dan disimpan:", args.model_output)
    print("📍 Lihat hasil tracking MLflow di DagsHub:")
    print("🔗 https://dagshub.com/Shinkai91/air-quality-model.mlflow")