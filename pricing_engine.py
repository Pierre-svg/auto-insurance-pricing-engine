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
from xgboost import XGBRegressor # <--- NEW IMPORT

# --- 1. DATA LOADING ---
def load_mtpl_data():
    print("--- ⬇️ Downloading French Motor Claims Data (OpenML) ---")
    df = fetch_openml(data_id=41214, as_frame=True, parser='auto').frame
    df['Exposure'] = df['Exposure'].clip(upper=1.0)
    df = df.rename(columns={'ClaimNb': 'ClaimCount'})
    return df

# --- 2. PREPROCESSING PIPELINE ---
def build_preprocessor():
    cat_features = ['VehBrand', 'VehGas', 'Region', 'Area']
    # Note: For XGBoost, we could technically use raw data, but for a fair 
    # comparison with GLM, we will use the same engineered features.
    bin_features = ['VehAge', 'DrivAge', 'Density']
    
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("bins", KBinsDiscretizer(n_bins=10, encode='onehot-dense'), bin_features),
            ("bonus", FunctionTransformer(np.log1p), ['BonusMalus']),
        ],
        remainder="drop",
    )
    return preprocessor

# --- 3. MAIN EXECUTION ---
# ... (Imports remain the same) ...

def run_pricing_engine():
    # Load & Split
    df = load_mtpl_data()
    # Filter for positive exposure
    df = df[df['Exposure'] > 0].copy()
    
    features = ['VehBrand', 'VehGas', 'Region', 'Area', 'VehAge', 'DrivAge', 'Density', 'BonusMalus']
    X = df[features]
    y = df['ClaimCount']
    w = df['Exposure']
    
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )
    
    # --- MODEL 1: TRADITIONAL GLM ---
    print("\n--- 🏛️ Training Traditional Actuarial GLM ---")
    preprocessor = build_preprocessor()
    glm_pipeline = make_pipeline(
        preprocessor,
        PoissonRegressor(alpha=1e-4, solver='newton-cholesky')
    )
    glm_pipeline.fit(X_train, y_train, poissonregressor__sample_weight=w_train)
    
    # --- MODEL 2: XGBOOST (The Challenger) ---
    print("--- 🤖 Training XGBoost (Gradient Boosting) ---")
    # TWEAK: Increased max_depth to 6 to capture non-linear "young driver" spikes
    # TWEAK: Increased n_estimators to 300 for better convergence
    xgb_pipeline = make_pipeline(
        build_preprocessor(),
        XGBRegressor(objective='count:poisson', n_estimators=300, learning_rate=0.05, max_depth=6)
    )
    xgb_pipeline.fit(X_train, y_train, xgbregressor__sample_weight=w_train)

    # --- EVALUATION ---
    # Predict (Frequency per unit exposure)
    pred_glm_freq = glm_pipeline.predict(X_test)
    pred_xgb_freq = xgb_pipeline.predict(X_test)
    
    # Convert to expected counts for Deviance calculation
    pred_glm_count = pred_glm_freq * w_test
    pred_xgb_count = pred_xgb_freq * w_test
    
    dev_glm = mean_poisson_deviance(y_test, pred_glm_count)
    dev_xgb = mean_poisson_deviance(y_test, pred_xgb_count)
    
    print(f"\n[Results]")
    print(f"GLM Poisson Deviance:    {dev_glm:.4f}")
    print(f"XGBoost Poisson Deviance:{dev_xgb:.4f}")
    
    # --- VISUALIZATION: THE CORRECT ACTUARIAL WAY ---
    # We must sum claims and sum exposure, NOT average the rates.
    plot_data = X_test.copy()
    plot_data['ClaimCount'] = y_test
    plot_data['Exposure'] = w_test
    plot_data['GLM_Pred_Count'] = pred_glm_count
    plot_data['XGB_Pred_Count'] = pred_xgb_count
    
    # Group by Driver Age
    plot_data['Age_Group'] = pd.cut(plot_data['DrivAge'], bins=np.arange(18, 90, 2))
    
    # CRITICAL FIX: Sum first, then divide!
    grouped = plot_data.groupby('Age_Group', observed=True)[['ClaimCount', 'Exposure', 'GLM_Pred_Count', 'XGB_Pred_Count']].sum()
    
    # Calculate weighted averages
    summary = pd.DataFrame(index=grouped.index)
    summary['Actual_Freq'] = grouped['ClaimCount'] / grouped['Exposure']
    summary['GLM_Freq'] = grouped['GLM_Pred_Count'] / grouped['Exposure']
    summary['XGB_Freq'] = grouped['XGB_Pred_Count'] / grouped['Exposure']
    
    plt.figure(figsize=(12, 6))
    
    # Plot Actual
    plt.plot(summary.index.astype(str), summary['Actual_Freq'], 'o', label='Actual Data (Weighted)', alpha=0.5, color='blue')
    
    # Plot GLM
    plt.plot(summary.index.astype(str), summary['GLM_Freq'], label=f'GLM (Dev={dev_glm:.4f})', color='red', lw=2, linestyle='--')
    
    # Plot XGBoost
    plt.plot(summary.index.astype(str), summary['XGB_Freq'], label=f'XGBoost (Dev={dev_xgb:.4f})', color='green', lw=2)
    
    plt.xticks(rotation=45)
    plt.title('Actuarial Showdown: GLM vs XGBoost (Corrected Weighted View)')
    plt.ylabel('Claims Frequency')
    plt.xlabel('Driver Age Group')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('model_comparison.png')
    print("✅ Comparison chart saved as 'model_comparison.png'")
    plt.show()

if __name__ == "__main__":
    run_pricing_engine()