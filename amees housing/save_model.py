import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from preprocess_data import preprocess_ames_housing
import joblib

def training_and_save():
    # Load and Preprocess
    df_cleaned, _ = preprocess_ames_housing("AmesHousing.csv")
    
    # Save the medians and column names for the Streamlit app
    feature_info = {
        'columns': list(df_cleaned.drop(columns=['SalePrice']).columns),
        'medians': df_cleaned.drop(columns=['SalePrice']).median().to_dict()
    }
    joblib.dump(feature_info, 'feature_info.pkl')
    
    # Train final model
    X = df_cleaned.drop(columns=['SalePrice'])
    y = df_cleaned['SalePrice']
    
    model = XGBRegressor(learning_rate=0.05, n_estimators=500, max_depth=5, reg_lambda=5)
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, 'ames_model.pkl')
    print("Model and feature info saved successfully.")

if __name__ == "__main__":
    training_and_save()
