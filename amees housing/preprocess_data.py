import pandas as pd
import numpy as np

def preprocess_ames_housing(file_path):
    """
    Performs professional-grade Data Preprocessing on the Ames Housing dataset.
    Follows strict methodology rules for a real estate stakeholder report.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        initial_count = len(df)
        
        # 4. Ethical & Professional Standards: Drop unique identifiers
        # In this dataset, 'Order' and 'PID' are unique identifiers.
        ids_to_drop = ['Order', 'PID']
        df = df.drop(columns=[col for col in ids_to_drop if col in df.columns])
        
        # 1. Smart Imputation Logic
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Numerical: Median
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
                
        # Categorical: Mode
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    # Fallback if mode is empty (rare)
                    df[col] = df[col].fillna('Unknown')
        
        # 2. 3-Sigma Outlier Filtering
        # Calculate Mean and Std for SalePrice and Gr Liv Area
        outlier_cols = ['SalePrice', 'Gr Liv Area']
        mask = pd.Series([True] * len(df))
        
        outlier_summary = {}
        for col in outlier_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                removed_count = len(df) - col_mask.sum()
                outlier_summary[col] = removed_count
                mask = mask & col_mask
        
        df_filtered = df[mask].copy()
        total_outliers_removed = initial_count - len(df_filtered)
        
        # 3. Advanced Feature Engineering
        # Create 'TotalSF'
        if all(col in df_filtered.columns for col in ['Total Bsmt SF', '1st Flr SF', '2nd Flr SF']):
            df_filtered['TotalSF'] = df_filtered['Total Bsmt SF'] + df_filtered['1st Flr SF'] + df_filtered['2nd Flr SF']
        
        # Create 'HouseAge'
        if all(col in df_filtered.columns for col in ['Yr Sold', 'Year Built']):
            df_filtered['HouseAge'] = df_filtered['Yr Sold'] - df_filtered['Year Built']
            # Ensure no negative age (could happen with bad data)
            df_filtered['HouseAge'] = df_filtered['HouseAge'].apply(lambda x: max(x, 0))
            
        # One-Hot Encoding for all categorical variables
        df_cleaned = pd.get_dummies(df_filtered, columns=categorical_cols, drop_first=True)
        
        # Final cleanup: Ensure column names are compatible with XGBoost (no brackets/commas)
        df_cleaned.columns = [c.replace('[', '').replace(']', '').replace('<', '') for c in df_cleaned.columns]
        
        print(f"--- Data Preprocessing Summary ---")
        print(f"Initial record count: {initial_count}")
        print(f"Records remaining: {len(df_cleaned)}")
        print(f"Total outliers removed (3-Sigma): {total_outliers_removed}")
        for col, count in outlier_summary.items():
            print(f"  - Outliers in '{col}': {count}")
        print(f"Feature Engineering: Created 'TotalSF', 'HouseAge', and applied One-Hot Encoding.")
        print(f"----------------------------------")
        
        return df_cleaned, total_outliers_removed

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, 0

if __name__ == "__main__":
    file_path = r"e:\programming\data science\amees housing\AmesHousing.csv"
    df_cleaned, outliers_removed = preprocess_ames_housing(file_path)
    if df_cleaned is not None:
        print("\nCleaned Dataframe 'df_cleaned' is now ready for Stakeholder Report.")
        # print(df_cleaned.head())
