"""
AeroPredict - Delay Time Regression Model
==========================================
Trains a GBT Regressor to predict HOW MANY MINUTES a delayed flight
will be late (regression on Arr_Delay).

Uses same features as the SVM classifier but keeps Arr_Delay as target.
Reads from raw US_flights_2023.csv since preprocessed CSV drops Arr_Delay.

Run with:
    python train_regressor.py

Output:
    delay_regressor.pkl       <- GBT Regressor model
    regressor_scaler.pkl      <- StandardScaler
    regressor_columns.pkl     <- Feature column names
"""

import time
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

start = time.time()

print("\n" + "="*60)
print("  AeroPredict - Delay Regression Model Training")
print("="*60)

# ==============================
# 1. LOAD RAW FLIGHT DATA
# ==============================
print("\n[1/6] Loading raw flight data (sampling 15%)...")

COLS_NEEDED = [
    'FlightDate', 'Airline', 'Dep_Airport', 'Arr_Airport',
    'DepTime_label', 'Distance_type', 'Flight_Duration',
    'Aicraft_age', 'Arr_Delay', 'Day_Of_Week'
]

df = pd.read_csv(
    "flight_delay/US_flights_2023.csv",
    usecols=[c for c in COLS_NEEDED],
    low_memory=False
).sample(frac=0.15, random_state=42)

print(f"      Loaded: {len(df):,} rows")

# Drop rows with missing Arr_Delay
df = df.dropna(subset=['Arr_Delay'])
print(f"      After dropna: {len(df):,} rows")

# ==============================
# 2. BUILD REGRESSION DATASET
# ==============================
print("\n[2/6] Preparing regression targets...")

# Clip extreme delays (>5 hours are outliers, skew model)
df['Arr_Delay'] = df['Arr_Delay'].clip(-30, 300)

# For regression, keep delayed flights + balanced sample of on-time
df_delayed = df[df['Arr_Delay'] > 15].copy()
n_delayed  = len(df_delayed)

df_ontime  = df[df['Arr_Delay'] <= 15].sample(
    n=min(n_delayed // 2, len(df[df['Arr_Delay'] <= 15])),
    random_state=42
).copy()
df_ontime['Arr_Delay'] = df_ontime['Arr_Delay'].clip(lower=0)

df_reg = pd.concat([df_delayed, df_ontime], ignore_index=True)
df_reg = df_reg.sample(frac=1, random_state=42)   # shuffle

print(f"      Delayed rows: {n_delayed:,}")
print(f"      On-time rows: {len(df_ontime):,}")
print(f"      Total training set: {len(df_reg):,}")

# ==============================
# 3. MERGE WEATHER + AIRPORTS
# ==============================
print("\n[3/6] Merging weather and airport coordinates...")

weather  = pd.read_csv("flight_delay/weather_meteo_by_airport.csv")
airports = pd.read_csv("flight_delay/airports_geolocation.csv")
weather.columns = weather.columns.str.strip()

df_reg['FlightDate'] = pd.to_datetime(df_reg['FlightDate'], errors='coerce')
weather['time']      = pd.to_datetime(weather['time'], errors='coerce')

# Merge airports (lat/lon)
df_reg = df_reg.merge(
    airports[['IATA_CODE', 'LATITUDE', 'LONGITUDE']],
    left_on='Dep_Airport', right_on='IATA_CODE', how='left'
).drop(columns=['IATA_CODE'], errors='ignore')

# Merge weather
df_reg = df_reg.merge(
    weather[['airport_id', 'time', 'tavg', 'prcp', 'wspd', 'pres']],
    left_on=['Dep_Airport', 'FlightDate'],
    right_on=['airport_id', 'time'],
    how='left'
).drop(columns=['airport_id', 'time'], errors='ignore')

print("      Merge complete.")

# ==============================
# 4. FEATURE ENGINEERING
# ==============================
print("\n[4/6] Engineering features...")

df_reg['month']      = df_reg['FlightDate'].dt.month
df_reg['day']        = df_reg['FlightDate'].dt.day
df_reg['is_weekend'] = df_reg['Day_Of_Week'].isin([6, 7]).astype(int)

# Fill missing values
weather_cols = ['tavg', 'prcp', 'wspd', 'pres']
df_reg[weather_cols] = df_reg[weather_cols].fillna(df_reg[weather_cols].median())
df_reg['LATITUDE']      = df_reg['LATITUDE'].fillna(39.0)
df_reg['LONGITUDE']     = df_reg['LONGITUDE'].fillna(-98.0)
df_reg['Aicraft_age']   = df_reg['Aicraft_age'].fillna(12)
df_reg['Flight_Duration']= df_reg['Flight_Duration'].fillna(df_reg['Flight_Duration'].median())
df_reg['Day_Of_Week']   = df_reg['Day_Of_Week'].fillna(1)
df_reg[['Airline', 'DepTime_label', 'Distance_type']] = \
    df_reg[['Airline', 'DepTime_label', 'Distance_type']].fillna('Unknown')

# One-hot encode categorical columns (same as SVM)
CATEGORICAL_COLS = ['Airline', 'DepTime_label', 'Distance_type']
df_enc = pd.get_dummies(df_reg, columns=CATEGORICAL_COLS, drop_first=False)

# Drop columns not needed as features
DROP_COLS = ['Dep_Airport', 'Arr_Airport', 'FlightDate', 'Arr_Delay']
X = df_enc.drop(columns=DROP_COLS, errors='ignore')
y = df_enc['Arr_Delay'].values

reg_columns = X.columns.tolist()
print(f"      Feature count: {len(reg_columns)}")

# ==============================
# 5. TRAIN GBT REGRESSOR
# ==============================
print("\n[5/6] Training GBT Regressor...")

X_scaled = StandardScaler().fit_transform(X.values)
reg_scaler = StandardScaler()
X_scaled   = reg_scaler.fit_transform(X.values)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

regressor = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    verbose=1
)
regressor.fit(X_tr, y_tr)

# Evaluate
y_pred = np.clip(regressor.predict(X_te), 0, 300)
mae    = mean_absolute_error(y_te, y_pred)
rmse   = mean_squared_error(y_te, y_pred) ** 0.5
r2     = r2_score(y_te, y_pred)

print(f"\n  {'-'*50}")
print(f"  [GBT Regressor - scikit-learn]")
print(f"  MAE  : {mae:.1f} minutes  (avg prediction error)")
print(f"  RMSE : {rmse:.1f} minutes")
print(f"  R2   : {r2:.3f}")
print(f"  {'-'*50}\n")

# ==============================
# 6. SAVE ARTIFACTS
# ==============================
print("[6/6] Saving regression artifacts...")

joblib.dump(regressor,   'delay_regressor.pkl')
joblib.dump(reg_scaler,  'regressor_scaler.pkl')
joblib.dump(reg_columns, 'regressor_columns.pkl')

print("      [SAVED] delay_regressor.pkl")
print("      [SAVED] regressor_scaler.pkl")
print("      [SAVED] regressor_columns.pkl")

print(f"\n{'='*60}")
print(f"  DONE in {time.time() - start:.1f}s")
print(f"  Restart Flask after: spark-submit flask_app.py")
print(f"{'='*60}\n")
