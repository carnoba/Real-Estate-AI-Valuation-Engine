import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import os

# Import preprocessing logic
from preprocess_data import preprocess_ames_housing

def train_and_evaluate():
    # 1. Load and Preprocess Data
    file_path = "AmesHousing.csv"
    df_cleaned, _ = preprocess_ames_housing(file_path)
    
    if df_cleaned is None:
        print("Data loading failed.")
        return

    # Define Features and Target
    X = df_cleaned.drop(columns=['SalePrice'])
    y = df_cleaned['SalePrice']
    
    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mean_price = y.mean()
    print(f"Mean Sale Price: ${mean_price:,.2f}")
    print(f"Target RMSE (10% of Mean): < ${0.1 * mean_price:,.2f}\n")

    # 2. Model Benchmark (The Trinity)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(learning_rate=0.05, n_estimators=500, random_state=42)
    }
    
    results = {}
    best_model_name = ""
    best_rmse = float('inf')

    print("--- Model Benchmarking ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"RMSE": rmse, "R2": r2}
        print(f"{name}: RMSE = ${rmse:,.2f}, R2 = {r2:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name

    # 3. Hyperparameter Optimization (XGBoost)
    print("\n--- Optimizing XGBoost ---")
    param_grid = {
        'max_depth': [3, 5, 7],
        'reg_lambda': [1, 5, 10]
    }
    
    xgb_search = GridSearchCV(XGBRegressor(learning_rate=0.05, n_estimators=500, random_state=42), 
                              param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    
    best_xgb = xgb_search.best_estimator_
    y_pred_opt = best_xgb.predict(X_test)
    rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
    r2_opt = r2_score(y_test, y_pred_opt)
    
    print(f"Optimized XGBoost Results: RMSE = ${rmse_opt:,.2f}, R2 = {r2_opt:.4f}")
    print(f"Best Params: {xgb_search.best_params_}")

    # Success Check
    success = rmse_opt < (0.1 * mean_price)
    print(f"\nSuccess Criterion (RMSE < 10% Mean): {'PASS' if success else 'FAIL'}")

    # 4. Professional Interpretation (SHAP)
    print("\n--- Generating SHAP Summary ---")
    explainer = shap.Explainer(best_xgb)
    shap_values = explainer(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    print("SHAP Summary Plot saved as 'shap_summary.png'")

    # Specific analysis for TotalSF and HouseAge
    # We look for their index or position in the summary
    # (Simplified for the console report)
    
    return results, best_xgb, xgb_search.best_params_, rmse_opt, r2_opt, mean_price

if __name__ == "__main__":
    train_and_evaluate()
