import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.linear_model import PoissonRegressor
from lightgbm import LGBMRegressor

# --- SETUP (Same as before) ---
def load_data():
    print("--- ⬇️ Loading Data ---")
    df = fetch_openml(data_id=41214, as_frame=True, parser='auto').frame
    df['Exposure'] = df['Exposure'].clip(upper=1.0)
    df = df.rename(columns={'ClaimNb': 'ClaimCount'})
    df = df[df['Exposure'] > 0].copy()
    return df

def get_pipelines():
    cat_features = ['VehBrand', 'VehGas', 'Region', 'Area']
    num_features = ['VehAge', 'DrivAge', 'Density', 'BonusMalus']
    
    # GLM Prep
    prep_glm = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ("bins", KBinsDiscretizer(n_bins=10, encode='onehot-dense'), ['VehAge', 'DrivAge', 'Density']),
        ("log", FunctionTransformer(np.log1p), ['BonusMalus']),
    ], remainder="drop")
    
    # LightGBM Prep
    prep_lgbm = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features),
        ("num", StandardScaler(), num_features),
    ], remainder="drop")
    
    return prep_glm, prep_lgbm

# --- LORENZ CURVE FUNCTION ---
def plot_lorenz(y_true, y_pred, exposure, model_name, ax):
    # 1. Calculate Risk Premium (Prediction)
    # 2. Sort data by Predicted Risk (Low -> High)
    data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'exposure': exposure})
    data = data.sort_values('y_pred')
    
    # 3. Calculate Cumulative distributions
    cum_exposure = data['exposure'].cumsum() / data['exposure'].sum()
    cum_claims = data['y_true'].cumsum() / data['y_true'].sum()
    
    # 4. Gini Coefficient (Area between curve and diagonal)
    # Approx using Trapezoidal rule
    area_under_curve = np.trapz(cum_claims, cum_exposure)
    gini = 2 * (area_under_curve - 0.5)
    
    # 5. Plot
    ax.plot(cum_exposure, cum_claims, label=f'{model_name} (Gini={gini:.3f})', lw=2)
    return gini

# --- MAIN EXECUTION ---
def run_evaluation():
    df = load_data()
    X = df.drop(columns=['ClaimCount', 'Exposure', 'IDpol'])
    y = df['ClaimCount']
    w = df['Exposure']
    
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.2, random_state=42)
    prep_glm, prep_lgbm = get_pipelines()
    
    print("--- 🏛️ Training GLM ---")
    glm = make_pipeline(prep_glm, PoissonRegressor(alpha=1e-4, solver='newton-cholesky'))
    glm.fit(X_train, y_train, poissonregressor__sample_weight=w_train)
    
    print("--- ⚡ Training LightGBM ---")
    lgbm = make_pipeline(prep_lgbm, LGBMRegressor(
    objective='poisson', 
    n_estimators=100,         # Reduce number of trees (300 -> 100)
    max_depth=3,              # Shallower trees (5 -> 3) preventing complex fake rules
    learning_rate=0.05,       # Learn slower
    min_child_samples=50,     # Crucial: Require at least 50 cars in a "bucket" to make a rule
    reg_alpha=1.0,            # L1 Regularization (removes noisy features)
    verbose=-1
))
    lgbm.fit(X_train, y_train, lgbmregressor__sample_weight=w_train)
    
    print("--- 📉 Generating Lorenz Curve ---")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Perfect Line (Diagonal)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guessing (Gini=0)')
    
    # GLM Curve
    pred_glm = glm.predict(X_test) * w_test # Predicted Counts
    plot_lorenz(y_test, pred_glm, w_test, "GLM", ax)
    
    # LightGBM Curve
    pred_lgbm = lgbm.predict(X_test) * w_test
    plot_lorenz(y_test, pred_lgbm, w_test, "LightGBM", ax)
    
    ax.set_title("Ordered Lorenz Curve: Sorting Risk")
    ax.set_xlabel("Cumulative Exposure (Sorted by Model Risk)")
    ax.set_ylabel("Cumulative Actual Claims")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('lorenz_curve.png')
    print("✅ Saved 'lorenz_curve.png'")
    plt.show()

if __name__ == "__main__":
    run_evaluation()