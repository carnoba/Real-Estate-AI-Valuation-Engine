import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config for a professional look
st.set_page_config(page_title="Ames Housing Prediction Dashboard", layout="wide")

st.title("🏡 Ames Housing Market Value Predictor")
st.markdown("""
This dashboard provides a professional-grade prediction of property values in Ames, Iowa, 
based on our optimized XGBoost model. 
""")

# Load model and feature info
@st.cache_resource
def load_assets():
    model = joblib.load('ames_model.pkl')
    feature_info = joblib.load('feature_info.pkl')
    return model, feature_info

if not os.path.exists('ames_model.pkl') or not os.path.exists('feature_info.pkl'):
    st.error("Model assets not found. Please run the training script first.")
else:
    model, feature_info = load_assets()
    all_columns = feature_info['columns']
    medians = feature_info['medians']

    # --- Sidebar Inputs ---
    st.sidebar.header("Property Specifications")
    
    # User Inputs
    total_sf = st.sidebar.slider("Total Square Footage (TotalSF)", 
                                  min_value=500, max_value=6000, value=2500, step=50)
    
    house_age = st.sidebar.slider("House Age (Years)", 
                                   min_value=0, max_value=150, value=20, step=1)
    
    overall_qual = st.sidebar.slider("Overall Quality (1-10)", 
                                      min_value=1, max_value=10, value=6, step=1)

    # --- Prediction Logic ---
    # Create a base dataframe with medians
    input_df = pd.DataFrame([medians])
    
    # Update with user inputs
    input_df['TotalSF'] = total_sf
    input_df['HouseAge'] = house_age
    input_df['Overall Qual'] = overall_qual
    
    # Ensure columns match training
    input_df = input_df[all_columns]
    
    # Make Prediction
    prediction = model.predict(input_df)[0]

    # Display Prediction
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Predicted Market Value")
        st.metric(label="Estimated Sale Price", value=f"${prediction:,.2f}")
        st.info(f"This prediction is based on a property with {total_sf} sqft, {house_age} years of age, and a quality rating of {overall_qual}/10.")

    with col2:
        st.subheader("Model Insights")
        st.write("The model utilizes over 200 features including neighborhood metrics and structural details. The values shown here are adjusted to reflect the most common property characteristics in Ames.")

    # --- Reporting & Reliability ---
    st.divider()
    st.header("Stakeholder Report: Model Reliability")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.subheader("Prediction vs. Actual")
        if os.path.exists('prediction_reliability.png'):
            st.image('prediction_reliability.png', use_container_width=True)
        else:
            st.warning("Prediction vs. Actual plot not found.")
            
    with viz_col2:
        st.subheader("Residual Distribution")
        if os.path.exists('residual_analysis.png'):
            st.image('residual_analysis.png', use_container_width=True)
        else:
            st.warning("Residual analysis plot not found.")

    st.markdown("""
    **Developer Note:**
    - The model achieved an **R² of 91%**.
    - Residuals are centered around zero, indicating no systematic bias in predictions.
    - Outliers were handled using a 3-Sigma filtering approach for stability.
    """)
