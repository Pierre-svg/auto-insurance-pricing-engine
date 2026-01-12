import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

# --- 1. DATA LOADING ---
def load_mtpl_data():
    print("--- ⬇️ Downloading French Motor Claims Data (OpenML) ---")
    # freq contains risk features and claim counts
    df = fetch_openml(data_id=41214, as_frame=True, parser='auto').frame
    
    # ACTUARIAL CLEANING:
    # 1. Cap Exposure at 1.0 (some data errors have >1 year)
    df['Exposure'] = df['Exposure'].clip(upper=1.0)
    
    # 2. ClaimAmount is huge for some; let's clip for stability (optional for frequency, needed for severity)
    df['ClaimAmount'] = df['ClaimAmount'].clip(upper=200000)
    
    # 3. Rename columns for clarity
    df = df.rename(columns={'ClaimNb': 'ClaimCount'})
    
    return df

# --- 2. PREPROCESSING PIPELINE ---
def build_preprocessor():
    # Categorical variables: Car Brand (VehBrand), Gas Type (VehGas), Region (Area)
    # We use OneHotEncoder to turn them into 1s and 0s
    cat_features = ['VehBrand', 'VehGas', 'Region', 'Area']
    
    # Numeric/Continuous variables: Vehicle Age, Driver Age, Density, BonusMalus
    # GLMs love "Binned" data (turning ages into groups like 18-25, 26-30...)
    # KBinsDiscretizer does this automatically
    bin_features = ['VehAge', 'DrivAge', 'Density']
    
    # BonusMalus is already a risk score, we keep it numeric but log-transform Density
    # (Population density is exponential, so we log it to make it linear)
    
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("bins", KBinsDiscretizer(n_bins=10, encode='onehot-dense'), bin_features),
            ("bonus", FunctionTransformer(np.log1p), ['BonusMalus']), # Log transform helpful for GLMs
        ],
        remainder="drop",
    )
    return preprocessor

# --- 3. MAIN EXECUTION ---
def run_pricing_engine():
    # Load Data
    df = load_mtpl_data()
    
    # Filter: We strictly need Positive Exposure
    df = df[df['Exposure'] > 0].copy()
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Average Frequency: {df['ClaimCount'].sum() / df['Exposure'].sum():.4f} claims/year")

    # Define X (Features), y (Target), and w (Exposure/Weight)
    # CRITICAL: In insurance, we weight samples by how long they were insured.
    # A driver with 0 crashes in 1 day is different than 0 crashes in 365 days.
    features = ['VehBrand', 'VehGas', 'Region', 'Area', 'VehAge', 'DrivAge', 'Density', 'BonusMalus']
    X = df[features]
    y = df['ClaimCount']
    w = df['Exposure']
    
    # Split Data (Stratified splitting is better, but simple random is fine for MVP)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )
    
    # --- MODEL 1: TRADITIONAL GLM (Poisson) ---
    print("\n--- 🏛️ Training Traditional Actuarial GLM ---")
    glm_pipeline = make_pipeline(
        build_preprocessor(),
        PoissonRegressor(alpha=1e-4, solver='newton-cholesky')
    )
    
    glm_pipeline.fit(X_train, y_train, poissonregressor__sample_weight=w_train)
    
    # Predict
    # GLM predicts "Claims per Unit of Exposure"
    y_pred_glm_freq = glm_pipeline.predict(X_test)
    
    # To get actual expected count, we multiply by exposure
    y_pred_glm_count = y_pred_glm_freq * w_test
    
    # Evaluate
    deviance = mean_poisson_deviance(y_test, y_pred_glm_count)
    print(f"GLM Poisson Deviance: {deviance:.4f} (Lower is better)")
    
    # --- VISUALIZATION: One-Way Analysis (Driver Age) ---
    # This proves you think like an actuary
    plot_data = X_test.copy()
    plot_data['Actual_Freq'] = y_test / w_test
    plot_data['GLM_Freq'] = y_pred_glm_freq
    
    # Group by Driver Age
    # We clip age at 90 for cleaner charts
    plot_data['Age_Group'] = pd.cut(plot_data['DrivAge'], bins=np.arange(18, 90, 2))
    summary = plot_data.groupby('Age_Group', observed=True)[['Actual_Freq', 'GLM_Freq']].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(summary.index.astype(str), summary['GLM_Freq'], label='GLM Prediction', color='red', lw=2)
    plt.plot(summary.index.astype(str), summary['Actual_Freq'], 'o', label='Actual Data', alpha=0.3, color='blue')
    plt.xticks(rotation=45)
    plt.title('Risk by Driver Age: The "Bathtub Curve"')
    plt.ylabel('Claims Frequency (per Year)')
    plt.xlabel('Driver Age Group')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('glm_driver_age.png')
    plt.show()

if __name__ == "__main__":
    run_pricing_engine()