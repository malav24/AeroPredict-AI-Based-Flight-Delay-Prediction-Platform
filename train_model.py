"""
AeroPredict Phase 2 - Combined Training Script
===============================================
Dual-output pipeline:
  1. GBT (Spark MLlib)  -> saved as Spark PipelineModel (for viva demo)
  2. SVM (scikit-learn) -> saved as .pkl files (for Streamlit app)

Run with:
    spark-submit train_model.py
"""

import time
import numpy as np
import pandas as pd
import joblib

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

start = time.time()

# ==============================
# 1. START SPARK & READ PARQUET
# ==============================
print("\n" + "="*60)
print("  AeroPredict Phase 2 - Dual Model Training")
print("="*60)
print("\n[1/6] Starting Spark and reading preprocessed HDFS Parquet...")

spark = SparkSession.builder \
    .appName("AeroPredict_GBT_SVM_Training") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(
    "hdfs://localhost:9000/flight_project/processed/flights_final"
)
print(f"      Rows loaded: {df.count():,}")

# ==============================
# 2. TRAIN / TEST SPLIT (SPARK)
# ==============================
print("[2/6] Splitting data (80/20)...")
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# ==============================
# 3. TRAIN GBT (SPARK MLLIB)
# ==============================
print("[3/6] Training GBT model via Spark MLlib...")

gbt = GBTClassifier(
    labelCol="delay_label",
    featuresCol="features",
    maxIter=20,
    maxDepth=5,
    stepSize=0.1,
    seed=42
)

gbt_pipeline = Pipeline(stages=[gbt])
gbt_model    = gbt_pipeline.fit(train_df)

gbt_predictions = gbt_model.transform(test_df)

evaluator = BinaryClassificationEvaluator(
    labelCol="delay_label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
acc_eval = MulticlassClassificationEvaluator(
    labelCol="delay_label",
    predictionCol="prediction",
    metricName="accuracy"
)
gbt_auc = evaluator.evaluate(gbt_predictions)
gbt_acc = acc_eval.evaluate(gbt_predictions)

print(f"\n  {'-'*50}")
print(f"  [GBT - Spark MLlib]")
print(f"  Accuracy  : {gbt_acc * 100:.2f}%")
print(f"  AUC-ROC   : {gbt_auc:.4f}")
print(f"  {'-'*50}\n")

# ==============================
# 4. SAVE GBT MODEL (SPARK)
# ==============================
print("[4/6] Saving Spark GBT PipelineModel...")
gbt_model.write().overwrite().save("gbt_spark_model")
print("      [SAVED] gbt_spark_model/")

# ── SPARK UI CHECKPOINT ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SPARK UI IS LIVE -> Open http://localhost:4040")
print("  Browse: Jobs, Stages, DAG Visualization, Storage tabs")
print("="*60)
input("  Press ENTER when done viewing Spark UI to continue...\n")
# ──────────────────────────────────────────────────────────────────────────────

spark.stop()

# ==============================
# 5. TRAIN SVM FOR STREAMLIT APP
# ==============================
# The Streamlit app builds features using pandas get_dummies (named columns).
# We train the SVM on the preprocessed CSV to match that exact format.
print("[5/6] Training app-compatible SVM from preprocessed CSV...")

CSV_PATH = "flight_delay/preprocessed_flight_datav2.csv"
df_csv = pd.read_csv(CSV_PATH).sample(frac=0.15, random_state=42)
print(f"      CSV sample size: {len(df_csv):,} rows")

CATEGORICAL_COLS = ['Airline', 'DepTime_label', 'Distance_type']
df_enc = pd.get_dummies(df_csv, columns=CATEGORICAL_COLS, drop_first=False)
df_enc.drop(
    columns=[c for c in ['Dep_Airport', 'Arr_Airport'] if c in df_enc.columns],
    inplace=True
)

X_app = df_enc.drop(columns=['is_delayed'])
y_app = df_enc['is_delayed'].astype(int).values
app_columns = X_app.columns.tolist()

app_scaler   = StandardScaler()
X_app_scaled = app_scaler.fit_transform(X_app.values)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_app_scaled, y_app, test_size=0.2, random_state=42, stratify=y_app
)

app_svm   = LinearSVC(C=0.1, max_iter=2000, random_state=42, class_weight='balanced')
app_model = CalibratedClassifierCV(app_svm, cv=3)
app_model.fit(X_tr, y_tr)

app_preds = app_model.predict(X_te)
app_acc   = accuracy_score(y_te, app_preds)

print(f"\n  {'-'*50}")
print(f"  [SVM - scikit-learn]")
print(f"  Accuracy  : {app_acc * 100:.2f}%")
print(f"  {'-'*50}")
print("\nClassification Report (SVM):")
print(classification_report(y_te, app_preds, target_names=["On-Time", "Delayed"]))

# ==============================
# 6. SAVE SVM ARTIFACTS
# ==============================
print("[6/6] Saving scikit-learn artifacts for Streamlit...")
joblib.dump(app_model,   'aeropredict_model.pkl')
joblib.dump(app_scaler,  'scaler.pkl')
joblib.dump(app_columns, 'model_columns.pkl')

print("      [SAVED] aeropredict_model.pkl")
print("      [SAVED] scaler.pkl")
print("      [SAVED] model_columns.pkl")

print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE in {time.time() - start:.1f}s")
print("  GBT model -> gbt_spark_model/  (viva demo)")
print("  SVM model -> aeropredict_model.pkl  (Streamlit app)")
print(f"{'='*60}\n")
print("Next: streamlit run app.py")
