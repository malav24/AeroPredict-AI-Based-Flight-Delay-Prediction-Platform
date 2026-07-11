# -*- coding: utf-8 -*-
"""
AeroPredict - Big Data Technology Benchmark
============================================
Compares time taken by each technology in the pipeline:
  - Hadoop HDFS  : data loading
  - PySpark      : GBT ML prediction
  - MongoDB      : NoSQL insert + read

Run with:
    spark-submit benchmark.py
"""

import sys
import time
from datetime import datetime

# Force UTF-8 output to avoid Windows cp1252 encoding errors
sys.stdout.reconfigure(encoding='utf-8')

# ── Spark ──────────────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
    .appName("AeroPredict_Benchmark") \
    .config("spark.driver.memory", "3g") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── MongoDB ────────────────────────────────────────────────────
from pymongo import MongoClient
try:
    mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    mongo_client.server_info()
    mongo_col = mongo_client["aeropredict"]["predictions"]
    mongo_ok  = True
except Exception as me:
    mongo_ok  = False
    mongo_col = None

results = {}

print("\n" + "="*62)
print("   AeroPredict -- Big Data Technology Benchmark")
print("="*62)

# ══════════════════════════════════════════════════════════════
# BENCHMARK 1: HDFS Data Loading
# ══════════════════════════════════════════════════════════════
HDFS_PATH = "hdfs://localhost:9000/flight_project/raw/US_flights_2023.csv"
print(f"\n[1/3] HADOOP HDFS -- Loading flight data...")
print(f"      Path: {HDFS_PATH}")

try:
    t0 = time.time()
    df_hdfs   = spark.read.csv(HDFS_PATH, header=True, inferSchema=True)
    row_count = df_hdfs.count()   # triggers actual distributed read
    t1 = time.time()
    hdfs_time = round(t1 - t0, 3)
    results["HDFS"] = {"time_sec": hdfs_time, "detail": f"{row_count:,} rows loaded from HDFS"}
    print(f"      [OK] {row_count:,} rows in {hdfs_time}s")
except Exception as e:
    results["HDFS"] = {"time_sec": None, "detail": f"HDFS not available"}
    print(f"      [SKIP] HDFS error: {e}")

# ══════════════════════════════════════════════════════════════
# BENCHMARK 2: PySpark GBT Prediction
# ══════════════════════════════════════════════════════════════
print("\n[2/3] PYSPARK -- Running GBT prediction...")

try:
    pipeline_model = PipelineModel.load("gbt_full_pipeline")

    sample_row = [{
        "Airline"              : "Spirit Air Lines",
        "DepTime_label"        : "Night",
        "Distance_type"        : "Short Haul >1500Mi",
        "tavg"                 : 0.06,
        "prcp"                 : 2.13,
        "wspd"                 : 15.0,
        "pres"                 : 1013.0,
        "LATITUDE"             : 41.98,
        "LONGITUDE"            : -87.9,
        "month"                : 1,
        "day_of_week"          : 2,
        "dep_airport_delay_avg": 0.22,
        "arr_airport_delay_avg": 0.25,
        "delay_label"          : 0,
    }]

    # Warm-up (not timed — removes JVM cold-start)
    _df = spark.createDataFrame(sample_row)
    _   = pipeline_model.transform(_df).select("prediction").first()

    # Timed run
    t0 = time.time()
    pred_df    = spark.createDataFrame(sample_row)
    result_row = pipeline_model.transform(pred_df).select("prediction","probability").first()
    t1 = time.time()

    spark_time = round(t1 - t0, 3)
    delay_prob = float(result_row["probability"][1])
    results["Spark GBT"] = {
        "time_sec": spark_time,
        "detail"  : f"delay_prob = {delay_prob*100:.1f}%"
    }
    print(f"      [OK] Prediction in {spark_time}s  (delay prob = {delay_prob*100:.1f}%)")

except Exception as e:
    results["Spark GBT"] = {"time_sec": None, "detail": f"GBT not available: {e}"}
    print(f"      [SKIP] {e}")

# ══════════════════════════════════════════════════════════════
# BENCHMARK 3: MongoDB Insert + Read
# ══════════════════════════════════════════════════════════════
print("\n[3/3] MONGODB -- Insert + Read benchmark...")

if mongo_ok:
    benchmark_doc = {
        "timestamp" : datetime.utcnow(),
        "source"    : "benchmark",
        "route"     : "ORD -> JFK",
        "airline"   : "NK -- Spirit Air Lines",
        "prediction": "ON TIME",
        "delay_prob": 13.9,
        "model_used": "GBT (Spark MLlib)",
    }

    # Timed INSERT
    t0 = time.time()
    inserted = mongo_col.insert_one(benchmark_doc)
    t1 = time.time()
    insert_ms = round((t1 - t0) * 1000, 2)

    # Timed READ (latest 10 docs)
    t0 = time.time()
    docs = list(mongo_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(10))
    t1 = time.time()
    read_ms = round((t1 - t0) * 1000, 2)

    # Clean up benchmark doc
    mongo_col.delete_one({"_id": inserted.inserted_id})

    results["MongoDB Write"] = {"time_sec": insert_ms / 1000, "detail": f"{insert_ms} ms"}
    results["MongoDB Read"]  = {"time_sec": read_ms   / 1000, "detail": f"{read_ms} ms  ({len(docs)} docs fetched)"}
    print(f"      [OK] Insert: {insert_ms} ms  |  Read: {read_ms} ms")
else:
    results["MongoDB Write"] = {"time_sec": None, "detail": "MongoDB not connected"}
    results["MongoDB Read"]  = {"time_sec": None, "detail": "MongoDB not connected"}
    print("      [SKIP] MongoDB not available")

# ══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("   BENCHMARK RESULTS SUMMARY")
print("="*62)
print(f"  {'Technology':<28}  {'Time':>10}  {'Details'}")
print(f"  {'-'*28}  {'-'*10}  {'-'*20}")

display = [
    ("HDFS",         "Hadoop HDFS  (data load)"),
    ("Spark GBT",    "PySpark GBT  (prediction)"),
    ("MongoDB Write", "MongoDB      (insert)"),
    ("MongoDB Read",  "MongoDB      (read 10)"),
]

for key, label in display:
    r    = results.get(key, {})
    t    = r.get("time_sec")
    d    = r.get("detail", "N/A")
    if t is None:
        t_str = "N/A"
    elif t >= 1:
        t_str = f"{t:.3f} s"
    else:
        t_str = f"{t*1000:.1f} ms"
    print(f"  {label:<28}  {t_str:>10}  {d}")

print("="*62)
print()
print("  Key Observations:")
print("  * HDFS    -- Distributed storage; high throughput for bulk data")
print("  * Spark   -- Parallel in-memory ML inference on large datasets")
print("  * MongoDB -- Sub-millisecond NoSQL read/write for live logging")
print()
print("  Pipeline Flow:")
print("  HDFS (store) --> Spark (process+train) --> MongoDB (log results)")
print("="*62 + "\n")

spark.stop()
