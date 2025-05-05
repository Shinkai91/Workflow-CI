import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import numpy as np  # Tambahkan impor numpy
import mlflow
import mlflow.sklearn
import dagshub
from modelling_tuning import tune_model

# Inisialisasi koneksi DagsHub
dagshub.init(repo_owner='Shinkai91', repo_name='air-quality-model', mlflow=True)

# Konfigurasi MLflow untuk tracking di DagsHub
mlflow.set_tracking_uri("https://dagshub.com/Shinkai91/air-quality-model.mlflow")

# Jika Anda ingin menyetel nama eksperimen (opsional)
mlflow.set_experiment("air-quality-prediction")

# Memuat dataset
data_path = "Membangun_model/Air-Quality_preprocessing.csv"
data = pd.read_csv(data_path)

# Menghapus kolom yang tidak diperlukan
data = data.drop(columns=["Unnamed: 15", "Unnamed: 16"])

# Memisahkan data menjadi fitur dan target
X = data.drop(columns=["AH"])  # Target adalah AH (Absolute Humidity)
y = data["AH"]

# Mengonversi nilai non-numerik menjadi numerik atau menghapus baris dengan nilai non-numerik
X = pd.get_dummies(X, drop_first=True)  # Mengonversi kolom kategorikal menjadi dummy variables
X = X.apply(pd.to_numeric, errors='coerce')  # Mengonversi semua nilai menjadi numerik
X = X.dropna()  # Menghapus baris dengan nilai NaN
y = y.loc[X.index]  # Menyesuaikan target agar sesuai dengan indeks fitur

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Mengaktifkan autologging MLflow
mlflow.sklearn.autolog()

# 1. Melatih model dasar
print("\n=== Training Basic Model ===")
with mlflow.start_run(run_name="basic_model"):
    # Inisialisasi dan melatih model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Membuat prediksi
    predictions = model.predict(X_test)

    # Menghitung metrik standar
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Basic Model - Mean Squared Error: {mse}")
    print(f"Basic Model - R² Score: {r2}")

    # Menghitung metrik tambahan (manual logging)
    mae = mean_absolute_error(y_test, predictions)
    evs = explained_variance_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))  # Hitung RMSE secara manual
    print(f"Basic Model - Mean Absolute Error: {mae}")
    print(f"Basic Model - Root Mean Squared Error: {rmse}")
    print(f"Basic Model - Explained Variance Score: {evs}")

    # Mencatat metrik dasar secara manual
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    
    # Mencatat metrik tambahan secara manual
    mlflow.log_metric("mae", mae)         # Metrik tambahan 1
    mlflow.log_metric("rmse", rmse)       # Metrik tambahan 2
    mlflow.log_metric("evs", evs)         # Metrik tambahan 3
    
    # Mencatat parameter dataset
    mlflow.log_param("data_path", data_path)
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    mlflow.log_param("features_count", X_train.shape[1])

# 2. Melatih model dengan hyperparameter tuning (memanggil fungsi dari modelling_tuning.py)
print("\n=== Training Model with Hyperparameter Tuning ===")
best_model = tune_model(X_train, y_train, X_test, y_test)

# Simpan URL DagsHub di file
with open("DagsHub.txt", "w") as f:
    f.write("https://dagshub.com/Shinkai91/air-quality-model")

print("\nPelatihan selesai! Data telah dicatat di:")
print("https://dagshub.com/Shinkai91/air-quality-model")