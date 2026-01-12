import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMRegressor

# --- 1. SETUP (Same as before) ---
def load_data():
    print("--- ⬇️ Loading Data for SHAP Analysis ---")
    df = fetch_openml(data_id=41214, as_frame=True, parser='auto').frame
    df['Exposure'] = df['Exposure'].clip(upper=1.0)
    df = df.rename(columns={'ClaimNb': 'ClaimCount'})
    df = df[df['Exposure'] > 0].copy()
    return df

def get_preprocessor():
    # LightGBM likes Ordinal Encoding (Categories -> Integers)
    cat_features = ['VehBrand', 'VehGas', 'Region', 'Area']
    num_features = ['VehAge', 'DrivAge', 'Density', 'BonusMalus']
    
    preprocessor = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features),
            ("num", StandardScaler(), num_features),
        ], remainder="drop"
    )
    return preprocessor

# --- 2. TRAIN BEST MODEL ---
def run_explanation():
    df = load_data()
    X = df.drop(columns=['ClaimCount', 'Exposure', 'IDpol'])
    y = df['ClaimCount']
    w = df['Exposure']
    
    # Preprocess manually so SHAP can understand the features
    preprocessor = get_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    # --- FIX: ROBUST FEATURE NAMING ---
    try:
        # Try modern sklearn method
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
        num_names = preprocessor.named_transformers_['num'].get_feature_names_out()
        feature_names = np.r_[cat_names, num_names]
    except AttributeError:
        # Fallback for older sklearn versions
        feature_names = preprocessor.get_feature_names_out()

    # Convert to DataFrame
    X_final = pd.DataFrame(X_transformed, columns=feature_names)
    
    print(f"Feature Names Found: {feature_names.tolist()}") # Debug print

    print("--- ⚡ Training LightGBM for Explanation ---")
    model = LGBMRegressor(objective='poisson', n_estimators=300, max_depth=5, verbose=-1)
    model.fit(X_final, y, sample_weight=w)
    
    print("--- 🕵️‍♀️ Calculating SHAP Values ---")
    explainer = shap.TreeExplainer(model)
    
    # Sample 2000 rows
    sample_idx = np.random.choice(X_final.index, size=2000, replace=False)
    X_sample = X_final.loc[sample_idx]
    shap_values = explainer.shap_values(X_sample)
    
    # --- PLOT 1: SUMMARY ---
    plt.figure(figsize=(10, 6))
    plt.title("What drives Insurance Risk? (SHAP Summary)")
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    print("✅ Saved 'shap_summary.png'")
    plt.close()
    
    # --- PLOT 2: DEPENDENCE (AUTO-DETECT NAME) ---
    print("Generating Dependence Plot for Driver Age...")
    
    # FIND the column that contains 'DrivAge' (handles 'num__DrivAge' or 'DrivAge' or other variants)
    age_col = [c for c in X_sample.columns if "DrivAge" in c][0]
    print(f"-> Detected Driver Age column as: '{age_col}'")
    
    plt.figure()
    shap.dependence_plot(age_col, shap_values, X_sample, show=False)
    plt.title("Isolated Impact of Driver Age")
    plt.tight_layout()
    plt.savefig('shap_dependence_age.png')
    print("✅ Saved 'shap_dependence_age.png'")
    
if __name__ == "__main__":
    run_explanation()