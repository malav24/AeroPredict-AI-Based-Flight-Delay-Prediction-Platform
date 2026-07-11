from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import os

def run_hybrid_pipeline():
    start_time = time.time()
    print("\n[START] Connecting to your active PySpark Session...")
    
    # Since you started PySpark from CMD, getOrCreate() will just bind to your active session
    spark = SparkSession.builder.appName("AeroPredictPhase1").getOrCreate()

    print("\n[PYSPARK] Loading Datasets in Distributed Memory...")
    # Paths to the raw datasets based on your project layout
    base_path = "flight_delay/"
    try:
        flights = spark.read.csv(base_path + "US_flights_2023.csv", header=True, inferSchema=True)
        weather = spark.read.csv(base_path + "weather_meteo_by_airport.csv", header=True, inferSchema=True)
        cancelled = spark.read.csv(base_path + "Cancelled_Diverted_2023.csv", header=True, inferSchema=True)
        print(f"[SUCCESS] Loaded Raw Data: Main dataset length is {flights.count()} rows.")
    except Exception as e:
        print(f"[ERROR] Missing data files. Error: {e}")
        spark.stop()
        return

    print("\n[PYSPARK] Commencing Data Integration & Cleaning...")
    
    # 1. Anti Join to remove cancelled flights (Left Anti Join)
    # We only want to predict delays for flights that actually operated.
    flights_clean = flights.join(cancelled, ["FlightDate", "Airline", "Tail_Number"], "left_anti")
    
    # 2. Join Weather data 
    # Link flights to weather based on departure airport and the date.
    flights_clean = flights_clean.join(weather, 
                                       (flights_clean.Dep_Airport == weather.airport_id) & 
                                       (flights_clean.FlightDate == weather.time), 
                                       "left")
    
    # 3. Create Target Variable (1 if delayed > 15 mins, 0 otherwise)
    flights_clean = flights_clean.withColumn("is_delayed", when(col("Arr_Delay") > 15, 1).otherwise(0))
    
    # 4. Fill Missing Weather (PySpark Native Imputation)
    flights_clean = flights_clean.fillna({
        'tavg': 10.0, 
        'prcp': 0.0, 
        'wspd': 5.0, 
        'pres': 1010.0,
        'Aicraft_age': 10.0
    })

    print("\n[PYSPARK] Selecting Stress Factors and Dropping Data Leakage...")
    features = ['Airline', 'DepTime_label', 'Flight_Duration', 'Distance_type', 
                'Aicraft_age', 'tavg', 'prcp', 'wspd', 'is_delayed']
    
    # Select available columns only and drop remaining NA
    available_cols = [c for c in features if c in flights_clean.columns]
    flights_final = flights_clean.select(available_cols).dropna()

    print("\n[PYSPARK] Sampling 5% of Cleaned Data to Pandas for AI Engine...")
    # Sample fraction (5%) to make Scikit-Learn training fast for Phase 1 Demo
    sample_df = flights_final.sample(fraction=0.05, seed=42)
    
    try:
        # Distributed -> Local conversion
        pandas_df = sample_df.toPandas()
        print(f"[SUCCESS] AI Engine DataFrame ready: {pandas_df.shape[0]} rows x {pandas_df.shape[1]} columns.")
        spark.stop() # Free up Java/Scala memory
    except Exception as e:
        print(f"[ERROR] Error converting to Pandas: {e}")
        spark.stop()
        return

    print("\n[SCIKIT-LEARN] Encoding Categories for Machine Learning...")
    # One-Hot Encoding for categories
    categorical_cols = ['Airline', 'DepTime_label', 'Distance_type']
    categorical_cols = [c for c in categorical_cols if c in pandas_df.columns]
    
    pandas_df = pd.get_dummies(pandas_df, columns=categorical_cols, drop_first=True)
    
    X = pandas_df.drop('is_delayed', axis=1)
    y = pandas_df['is_delayed']
    
    # Split into testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n[SCIKIT-LEARN] Training Baseline Random Forest Model...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print("\n" + "="*60)
    print(f" PHASE 1: AI MODEL EVALUATION REPORT")
    print("="*60)
    print(f"Algorithm Used: Random Forest (Scikit-Learn)")
    print(f"Data Pipeline:  Apache PySpark")
    print(f"Training Size:  {len(X_train)} flights (Sampled)")
    print(f"Evaluating:     {len(X_test)} unseen flights")
    print(f"--> PREDICTIVE ACCURACY: {acc * 100:.2f}% <--")
    print("="*60)
    
    print(f"\n[FINISH] Phase 1 Complete in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_hybrid_pipeline()
