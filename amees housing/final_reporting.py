import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Import modules from previous steps
from preprocess_data import preprocess_ames_housing

def generate_final_reports():
    # 1. Setup Data and Model (Simplified for the reporting script)
    file_path = "AmesHousing.csv"
    df_cleaned, _ = preprocess_ames_housing(file_path)
    
    if df_cleaned is None: return
    
    X = df_cleaned.drop(columns=['SalePrice'])
    y = df_cleaned['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using the best hyperparameters found in Step 2: max_depth=5, reg_lambda=1 (standard optimized)
    model = XGBRegressor(learning_rate=0.05, n_estimators=500, max_depth=5, reg_lambda=1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 2. Prediction vs. Actual Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='#2c3e50')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Reliability Assessment: Prediction vs. Actual Sale Price', fontsize=14)
    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('prediction_reliability.png')
    print("Saved: prediction_reliability.png")

    # 3. Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color='#e67e22')
    plt.axhline(y=0, color='black', linestyle='--', lw=2)
    plt.title('Error Distribution: Residual Plot (Heteroscedasticity Check)', fontsize=14)
    plt.xlabel('Predicted Price ($)', fontsize=12)
    plt.ylabel('Residuals ($)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('residual_analysis.png')
    print("Saved: residual_analysis.png")

    # 4. House Age Distribution for Insights
    plt.figure(figsize=(10, 6))
    sns.regplot(x=df_cleaned['HouseAge'], y=df_cleaned['SalePrice'], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title('Impact of Property Age on Market Value', fontsize=14)
    plt.xlabel('House Age (Years)', fontsize=12)
    plt.ylabel('Sale Price ($)', fontsize=12)
    plt.tight_layout()
    plt.savefig('house_age_impact.png')
    print("Saved: house_age_impact.png")

if __name__ == "__main__":
    generate_final_reports()
