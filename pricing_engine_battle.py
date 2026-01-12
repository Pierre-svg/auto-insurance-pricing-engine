import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor # <--- NEW CHALLENGER

# --- 1. DATA LOADING ---
def load_mtpl_data():
    print("--- ⬇️ Downloading French Motor Claims Data ---")
    df = fetch_openml(data_id=41214, as_frame=True, parser='auto').frame
    df['Exposure'] = df['Exposure'].clip(upper=1.0)
    df = df.rename(columns={'ClaimNb': 'ClaimCount'})
    return df

# --- 2. PREPROCESSING PIPELINES ---
def build_pipelines():
    cat_features = ['VehBrand', 'VehGas', 'Region', 'Area']
    num_features = ['VehAge', 'DrivAge', 'Density', 'BonusMalus']
    
    # A. GLM Pipeline (Bins + OneHot)
    prep_glm = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("bins", KBinsDiscretizer(n_bins=10, encode='onehot-dense'), ['VehAge', 'DrivAge', 'Density']),
            ("log", FunctionTransformer(np.log1p), ['BonusMalus']),
        ], remainder="drop"
    )
    
    # B. Boosting Pipeline (Ordinal + Raw Numbers)
    # Both XGBoost and LightGBM like this format
    prep_boost = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features),
            ("num", StandardScaler(), num_features), 
        ], remainder="drop"
    )
    
    return prep_glm, prep_boost

# --- 3. MAIN BATTLE ENGINE ---
def run_battle_engine():
    # Load & Split
    df = load_mtpl_data()
    df = df[df['Exposure'] > 0].copy()
    
    X = df.drop(columns=['ClaimCount', 'Exposure', 'IDpol'])
    y = df['ClaimCount']
    w = df['Exposure']
    
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )
    
    prep_glm, prep_boost = build_pipelines()
    
    # --- MODEL 1: GLM (The Baseline) ---
    print("\n--- 🏛️ Training GLM ---")
    glm = make_pipeline(prep_glm, PoissonRegressor(alpha=1e-4, solver='newton-cholesky'))
    glm.fit(X_train, y_train, poissonregressor__sample_weight=w_train)
    
    # --- MODEL 2: XGBOOST (The Champion) ---
    print("--- 🤖 Training XGBoost ---")
    xgb = make_pipeline(
        prep_boost,
        XGBRegressor(objective='count:poisson', n_estimators=300, learning_rate=0.05, max_depth=5, n_jobs=-1)
    )
    xgb.fit(X_train, y_train, xgbregressor__sample_weight=w_train)

    # --- MODEL 3: LIGHTGBM (The Challenger) ---
    print("--- ⚡ Training LightGBM ---")
    lgbm = make_pipeline(
        prep_boost,
        LGBMRegressor(objective='poisson', n_estimators=300, learning_rate=0.05, max_depth=5, n_jobs=-1, verbose=-1)
    )
    # LightGBM handles weights in .fit() just like the others
    lgbm.fit(X_train, y_train, lgbmregressor__sample_weight=w_train)

    # --- EVALUATION ---
    print("\n--- 📊 Final Scoreboard (Poisson Deviance) ---")
    
    # Helper to calculate deviance
    def get_deviance(model, name):
        pred_freq = model.predict(X_test)
        pred_count = pred_freq * w_test
        dev = mean_poisson_deviance(y_test, pred_count)
        print(f"{name}: {dev:.5f}")
        return pred_count, dev

    pred_glm, dev_glm = get_deviance(glm, "GLM")
    pred_xgb, dev_xgb = get_deviance(xgb, "XGBoost")
    pred_lgbm, dev_lgbm = get_deviance(lgbm, "LightGBM")
    
    # --- VISUALIZATION ---
    plot_data = X_test.copy()
    plot_data['Exposure'] = w_test
    plot_data['ClaimCount'] = y_test
    plot_data['GLM'] = pred_glm
    plot_data['XGB'] = pred_xgb
    plot_data['LGBM'] = pred_lgbm
    
    # Group by Age
    plot_data['Age_Group'] = pd.cut(plot_data['DrivAge'], bins=np.arange(18, 90, 2))
    grouped = plot_data.groupby('Age_Group', observed=True)[['ClaimCount', 'Exposure', 'GLM', 'XGB', 'LGBM']].sum()
    
    summary = pd.DataFrame(index=grouped.index)
    summary['Actual'] = grouped['ClaimCount'] / grouped['Exposure']
    summary['GLM'] = grouped['GLM'] / grouped['Exposure']
    summary['XGB'] = grouped['XGB'] / grouped['Exposure']
    summary['LGBM'] = grouped['LGBM'] / grouped['Exposure']
    
    plt.figure(figsize=(12, 6))
    plt.plot(summary.index.astype(str), summary['Actual'], 'o', label='Actual', alpha=0.3, color='blue')
    plt.plot(summary.index.astype(str), summary['GLM'], label=f'GLM ({dev_glm:.4f})', color='red', ls='--')
    plt.plot(summary.index.astype(str), summary['XGB'], label=f'XGB ({dev_xgb:.4f})', color='green', lw=2)
    plt.plot(summary.index.astype(str), summary['LGBM'], label=f'LGBM ({dev_lgbm:.4f})', color='orange', lw=2, linestyle='-.')
    
    plt.xticks(rotation=45)
    plt.title('Algorithm Battle: GLM vs XGBoost vs LightGBM')
    plt.ylabel('Claims Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('algorithm_battle.png')
    plt.show()

if __name__ == "__main__":
    run_battle_engine()