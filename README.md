# 🏠 Ames Housing: Strategic Market Value Predictor

This repository contains a professional-grade predictive modeling pipeline for the Ames Housing dataset. The project leverages advanced regression techniques and a Streamlit dashboard to provide actionable pricing insights for real estate stakeholders.

## 🚀 Project Highlights
- **Model Accuracy:** Achieved a robust **91% R² score** using an optimized XGBoost Regressor.
- **Error Reliability:** Maintained an **RMSE below 10%** of the mean price ($16,450), exceeding industrial standards.
- **Interpretability:** Integrated **SHAP Values** to decode feature impacts like the "Age Cliff" and "SF Premium."

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Scikit-learn, XGBoost, SHAP, Pandas, Matplotlib, Seaborn
- **Deployment:** Streamlit Dashboard

## 📊 Methodology
1. **Preprocessing:** Applied **3-Sigma Outlier Filtering** and Median/Mode imputation for data integrity.
2. **Feature Engineering:** Engineered `TotalSF` (interaction term) and `HouseAge` to capture real-world depreciation.
3. **Model Trinity:** Benchmarked Linear Regression vs. Random Forest vs. XGBoost.

## 💻 Installation & Usage
1. Clone the repo: `git clone [Your Link]`
2. Install requirements: `pip install -r requirements.txt`
3. Train & Export: `python scripts/save_model.py`
4. Launch Dashboard: `streamlit run app/app.py`

## 📁 Presentation
The full 10-slide professional presentation is available in folder, detailing business insights and strategic recommendations.
