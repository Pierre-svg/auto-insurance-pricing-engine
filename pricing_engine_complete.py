import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PoissonRegressor, GammaRegressor

# --- 1. DATA LOADING ---
def load_pricing_data():
    print("--- ⬇️ Downloading Data ---")
    df_freq = fetch_openml(data_id=41214, as_frame=True, parser='auto').frame
    df_freq['IDpol'] = df_freq['IDpol'].astype(int)
    df_freq = df_freq.rename(columns={'ClaimNb': 'ClaimCount'})
    df_freq['Exposure'] = df_freq['Exposure'].clip(upper=1.0)
    
    df_sev = fetch_openml(data_id=41215, as_frame=True, parser='auto').frame
    df_sev['IDpol'] = df_sev['IDpol'].astype(int)
    df_sev_agg = df_sev.groupby('IDpol')['ClaimAmount'].sum().reset_index()
    
    df = pd.merge(df_freq, df_sev_agg, on='IDpol', how='left')
    df['ClaimAmount'] = df['ClaimAmount'].fillna(0).clip(upper=200000)
    return df

# --- 2. PREPROCESSING ---
def build_preprocessor():
    # We use the same preprocessor as before
    cat_features = ['VehBrand', 'VehGas', 'Region', 'Area']
    bin_features = ['VehAge', 'DrivAge', 'Density']
    
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("bins", KBinsDiscretizer(n_bins=10, encode='onehot-dense', strategy='quantile'), bin_features),
            ("bonus", FunctionTransformer(np.log1p), ['BonusMalus']),
        ],
        remainder="drop",
    )
    return preprocessor

# --- 3. MAIN ENGINE ---
def run_pricing_engine():
    df = load_pricing_data()
    df = df[df['Exposure'] > 0].copy()
    
    # Feature List
    features = ['VehBrand', 'VehGas', 'Region', 'Area', 'VehAge', 'DrivAge', 'Density', 'BonusMalus']
    
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    # --- TRAIN MODELS ---
    print("\n--- 🧠 Training Pricing Models ---")
    
    # Frequency (Poisson)
    freq_model = make_pipeline(build_preprocessor(), PoissonRegressor(alpha=1e-4, solver='newton-cholesky'))
    freq_model.fit(df_train[features], df_train['ClaimCount'], poissonregressor__sample_weight=df_train['Exposure'])
    
    # Severity (Gamma)
    df_claims = df_train[(df_train['ClaimCount'] > 0) & (df_train['ClaimAmount'] > 0)].copy()
    df_claims['AvgSeverity'] = df_claims['ClaimAmount'] / df_claims['ClaimCount']
    
    sev_model = make_pipeline(build_preprocessor(), GammaRegressor(alpha=10.0, solver='newton-cholesky'))
    sev_model.fit(df_claims[features], df_claims['AvgSeverity'], gammaregressor__sample_weight=df_claims['ClaimCount'])
    
    # --- PREDICTIONS ---
    print("--- 🔮 Predicting Premiums ---")
    df_test['PurePremium'] = freq_model.predict(df_test[features]) * sev_model.predict(df_test[features])
    
    # --- VISUALIZATION 1: AGE (The Bathtub) ---
    df_test['Age_Group'] = pd.cut(df_test['DrivAge'], bins=np.arange(18, 90, 5))
    age_summary = df_test.groupby('Age_Group', observed=True)['PurePremium'].mean()
    
    plt.figure(figsize=(10, 5))
    plt.plot(age_summary.index.astype(str), age_summary.values, color='purple', lw=3, marker='o')
    plt.title('Impact of AGE on Price (Bathtub Curve)')
    plt.ylabel('Avg Premium (€)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('impact_age.png')
    
    # --- VISUALIZATION 2: BONUS-MALUS (Driver History) ---
    # BonusMalus < 100 is Good, > 100 is Bad
    # We bin it to see the trend clearly
    df_test['Bonus_Group'] = pd.cut(df_test['BonusMalus'], bins=[50, 60, 70, 80, 90, 100, 110, 120, 150])
    bonus_summary = df_test.groupby('Bonus_Group', observed=True)['PurePremium'].mean()
    
    plt.figure(figsize=(10, 5))
    plt.plot(bonus_summary.index.astype(str), bonus_summary.values, color='green', lw=3, marker='s')
    plt.title('Impact of DRIVER HISTORY on Price (Bonus-Malus)')
    plt.xlabel('Bonus-Malus Score (Lower is Better)')
    plt.ylabel('Avg Premium (€)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('impact_bonus.png')
    print("✅ Charts saved: 'impact_age.png' and 'impact_bonus.png'")

    # --- SHOWDOWN: THE QUOTE CALCULATOR ---
    print("\n--- 🆚  QUOTE COMPARISON (Same Age, Different Profile) ---")
    
    # Let's take two 20-year-olds
    # Profile A: Good driver (Bonus 50), Old Car (15 years), Small City (Density 50)
    profile_safe = pd.DataFrame({
        'VehBrand': ['B12'], 'VehGas': ['Regular'], 'Region': ['R24'], 'Area': ['C'],
        'VehAge': [15], 'DrivAge': [20], 'Density': [50], 'BonusMalus': [50]
    })
    
    # Profile B: Bad driver (Bonus 100), New Car (1 year), Big City (Density 20000)
    profile_risky = pd.DataFrame({
        'VehBrand': ['B12'], 'VehGas': ['Regular'], 'Region': ['R24'], 'Area': ['C'],
        'VehAge': [1], 'DrivAge': [20], 'Density': [20000], 'BonusMalus': [120]
    })
    
    price_safe = freq_model.predict(profile_safe)[0] * sev_model.predict(profile_safe)[0]
    price_risky = freq_model.predict(profile_risky)[0] * sev_model.predict(profile_risky)[0]
    
    print(f"Driver A (Safe, 20yo):  €{price_safe:.2f}")
    print(f"Driver B (Risky, 20yo): €{price_risky:.2f}")
    print(f"Ratio: Risky driver pays {price_risky/price_safe:.1f}x more!")

if __name__ == "__main__":
    run_pricing_engine()