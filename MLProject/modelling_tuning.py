from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

def tune_model(X_train, y_train, X_test, y_test):
    """
    Melakukan hyperparameter tuning pada model RandomForestRegressor
    menggunakan GridSearchCV dan mencatat hasilnya dengan MLflow.
    
    Args:
        X_train: Fitur data training
        y_train: Target data training
        X_test: Fitur data testing
        y_test: Target data testing
    
    Returns:
        Model terbaik hasil tuning
    """
    # Hyperparameter tuning menggunakan GridSearchCV
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=2,
        n_jobs=-1,
    )

    # Memulai run MLflow
    with mlflow.start_run(run_name="tuned_model"):
        # Melatih model dengan GridSearchCV
        grid_search.fit(X_train, y_train)

        # Model terbaik dari GridSearchCV
        best_model = grid_search.best_estimator_

        # Membuat prediksi
        predictions = best_model.predict(X_test)

        # Menghitung metrik standar
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Menghitung metrik tambahan
        mae = mean_absolute_error(y_test, predictions)
        evs = explained_variance_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))  # Hitung RMSE secara manual
        
        # Output metrik
        print(f"Best Model - Mean Squared Error: {mse}")
        print(f"Best Model - R² Score: {r2}")
        print(f"Best Model - Mean Absolute Error: {mae}")
        print(f"Best Model - Root Mean Squared Error: {rmse}")
        print(f"Best Model - Explained Variance Score: {evs}")
        print(f"Best Parameters: {grid_search.best_params_}")

        # Mencatat parameter terbaik
        mlflow.log_params(grid_search.best_params_)

        # Mencatat metrik dasar secara manual
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Mencatat metrik tambahan secara manual
        mlflow.log_metric("mae", mae)         # Metrik tambahan 1
        mlflow.log_metric("rmse", rmse)       # Metrik tambahan 2
        mlflow.log_metric("evs", evs)         # Metrik tambahan 3
        
        # Tambahkan informasi lain yang mungkin berguna
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("scoring", "neg_mean_squared_error")
        
        # Simpan model terbaik
        mlflow.sklearn.log_model(best_model, "tuned_model")
        
        # Simpan informasi fitur penting
        feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Simpan feature importances sebagai artifact
        temp_fi_path = "feature_importances.csv"
        feature_importances.to_csv(temp_fi_path, index=False)
        mlflow.log_artifact(temp_fi_path)
        import os
        if os.path.exists(temp_fi_path):
            os.remove(temp_fi_path)

    return best_model

# Jika file ini dijalankan secara langsung, beri contoh penggunaan
if __name__ == "__main__":
    print("File ini berisi fungsi tune_model() yang dapat diimpor dan digunakan dalam file lain.")
    print("Untuk menggunakan fungsi ini, impor dengan: from modelling_tuning import tune_model")