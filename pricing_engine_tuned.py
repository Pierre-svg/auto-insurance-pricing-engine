import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from xgboost import XGBRegressor

# --- 1. DATA LOADING ---
def load_mtpl_data():
    print("--- ⬇️ Downloading French Motor Claims Data ---")
    df = fetch_openml(data_id=41214, as_frame=True, parser='auto').frame
    df['Exposure'] = df['Exposure'].clip(upper=1.0)
    df = df.rename(columns={'ClaimNb': 'ClaimCount'})
    return df

# --- 2. PIPELINE FACTORY (Split Pipelines) ---
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
    
    # B. XGBoost Pipeline (Raw Numbers + Ordinal)
    prep_xgb = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features),
            ("num", StandardScaler(), num_features), 
        ], remainder="drop"
    )
    
    return prep_glm, prep_xgb

# --- 3. MAIN TUNING ENGINE ---
def run_tuning_engine():
    # Load & Split
    df = load_mtpl_data()
    df = df[df['Exposure'] > 0].copy()
    
    features = ['VehBrand', 'VehGas', 'Region', 'Area', 'VehAge', 'DrivAge', 'Density', 'BonusMalus']
    X = df[features]
    y = df['ClaimCount']
    w = df['Exposure']
    
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )
    
    prep_glm, prep_xgb = build_pipelines()
    
    # --- MODEL 1: GLM (Baseline) ---
    print("\n--- 🏛️ Training Baseline GLM ---")
    glm = make_pipeline(prep_glm, PoissonRegressor(alpha=1e-4, solver='newton-cholesky'))
    glm.fit(X_train, y_train, poissonregressor__sample_weight=w_train)
    
    # --- MODEL 2: XGBOOST (Hyperparameter Tuning) ---
    print("--- 🤖 Tuning XGBoost (Testing 20 combinations...) ---")
    
    # 1. Define the Search Space
    param_dist = {
        'xgbregressor__n_estimators': [200, 400, 600],
        'xgbregressor__max_depth': [3, 4, 5, 6, 8],          # 3 is shallow, 8 is very complex
        'xgbregressor__learning_rate': [0.01, 0.05, 0.1],    # Lower = more robust
        'xgbregressor__colsample_bytree': [0.5, 0.7, 1.0],   # Feature randomness
        'xgbregressor__reg_alpha': [0, 0.1, 1, 10],          # L1 Regularization (Important!)
        'xgbregressor__reg_lambda': [1, 5, 10],              # L2 Regularization
    }
    
    # 2. Setup the Pipeline
    xgb_pipeline = make_pipeline(
        prep_xgb,
        XGBRegressor(objective='count:poisson', n_jobs=-1, random_state=42)
    )
    
    # 3. Setup RandomizedSearchCV
    search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=param_dist,
        n_iter=15,    # Try 15 random settings
        scoring='neg_mean_poisson_deviance', # Actuarial Metric
        cv=3,         # 3-Fold Cross Validation
        verbose=1,
        random_state=42
    )
    
    # 4. Run the Search (Pass weights!)
    # Note: We use 'xgbregressor__sample_weight' to pass weights into the pipeline step
    search.fit(X_train, y_train, xgbregressor__sample_weight=w_train)
    
    print(f"\n✅ Best Parameters Found:\n{search.best_params_}")
    best_xgb = search.best_estimator_

    # --- EVALUATION ---
    print("\n--- 📊 Final Showdown ---")
    
    # Predictions
    # Note: .predict() gives frequency, we multiply by Exposure (w) to get Counts for Deviance
    pred_glm_freq = glm.predict(X_test)
    pred_xgb_freq = best_xgb.predict(X_test)
    
    dev_glm = mean_poisson_deviance(y_test, pred_glm_freq * w_test)
    dev_xgb = mean_poisson_deviance(y_test, pred_xgb_freq * w_test)
    
    print(f"GLM Deviance:     {dev_glm:.5f}")
    print(f"XGBoost Deviance: {dev_xgb:.5f}")
    print(f"Improvement:      {(dev_glm - dev_xgb) / dev_glm:.2%} better")

    # --- VISUALIZATION ---
    # We compare the "Tuned" XGBoost against the GLM on the famous Age Curve
    plot_data = X_test.copy()
    plot_data['ClaimCount'] = y_test
    plot_data['Exposure'] = w_test
    plot_data['GLM_Pred'] = pred_glm_freq * w_test
    plot_data['XGB_Pred'] = pred_xgb_freq * w_test
    
    plot_data['Age_Group'] = pd.cut(plot_data['DrivAge'], bins=np.arange(18, 90, 2))
    grouped = plot_data.groupby('Age_Group', observed=True)[['ClaimCount', 'Exposure', 'GLM_Pred', 'XGB_Pred']].sum()
    
    summary = pd.DataFrame(index=grouped.index)
    summary['Actual'] = grouped['ClaimCount'] / grouped['Exposure']
    summary['GLM'] = grouped['GLM_Pred'] / grouped['Exposure']
    summary['XGB_Tuned'] = grouped['XGB_Pred'] / grouped['Exposure']
    
    plt.figure(figsize=(12, 6))
    plt.plot(summary.index.astype(str), summary['Actual'], 'o', label='Actual Data', alpha=0.4, color='blue')
    plt.plot(summary.index.astype(str), summary['GLM'], label=f'GLM (Dev={dev_glm:.4f})', color='red', lw=2, ls='--')
    plt.plot(summary.index.astype(str), summary['XGB_Tuned'], label=f'XGBoost Tuned (Dev={dev_xgb:.4f})', color='green', lw=2)
    
    plt.xticks(rotation=45)
    plt.title('Tuned XGBoost vs GLM: Capturing the Young Driver Risk')
    plt.ylabel('Claims Frequency')
    plt.xlabel('Driver Age Group')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tuned_showdown.png')
    print("✅ Saved chart to 'tuned_showdown.png'")
    plt.show()

if __name__ == "__main__":
    run_tuning_engine()