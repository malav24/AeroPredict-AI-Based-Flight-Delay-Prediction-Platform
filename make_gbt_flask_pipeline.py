"""
AeroPredict - GBT Flask Pipeline Builder
=========================================
Creates a full Spark ML Pipeline (feature engineering + GBT) from HDFS data.
This pipeline is used by the Flask HTML frontend for predictions.

Run ONCE with:
    spark-submit make_gbt_flask_pipeline.py

Output:
    gbt_full_pipeline/     <- Combined Spark PipelineModel for Flask
    airport_delay_lookup.csv  <- Airport delay averages for auto-fill
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, month, dayofweek, when, to_date
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

start = time.time()

print("\n" + "="*60)
print("  Building GBT Flask Pipeline from HDFS data")
print("="*60)

spark = SparkSession.builder \
    .appName("AeroPredict_GBT_Flask_Pipeline") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
    .config("spark.memory.fraction", "0.8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ==============================
# 1. LOAD RAW DATA FROM HDFS
# ==============================
print("[1/7] Loading raw data from HDFS...")

flights = spark.read.csv(
    "hdfs://localhost:9000/flight_project/raw/US_flights_2023.csv",
    header=True, inferSchema=True
).sample(fraction=0.10, seed=42)

airports = spark.read.csv(
    "hdfs://localhost:9000/flight_project/raw/airports_geolocation.csv",
    header=True, inferSchema=True
)

weather = spark.read.csv(
    "hdfs://localhost:9000/flight_project/raw/weather_meteo_by_airport.csv",
    header=True, inferSchema=True
)

cancelled = spark.read.csv(
    "hdfs://localhost:9000/flight_project/raw/Cancelled_Diverted_2023.csv",
    header=True, inferSchema=True
)

print(f"      Flights sample: {flights.count():,} rows")

# ==============================
# 2. CLEAN & JOIN
# ==============================
print("[2/7] Cleaning and joining datasets...")

flights = flights.join(
    cancelled.select("FlightDate", "Airline", "Tail_Number"),
    ["FlightDate", "Airline", "Tail_Number"], "left_anti"
)
flights = flights.dropDuplicates()
flights = flights.withColumn("FlightDate", to_date("FlightDate"))
weather  = weather.withColumn("time", to_date("time"))

airports_clean = airports.withColumnRenamed("IATA_CODE", "Dep_Airport")
flights = flights.join(
    airports_clean.select("Dep_Airport", "LATITUDE", "LONGITUDE"),
    "Dep_Airport", "left"
)

weather.cache()
flights = flights.join(
    weather,
    (flights.Dep_Airport == weather.airport_id) &
    (flights.FlightDate == weather.time),
    "left"
).drop("airport_id", "time")

# ==============================
# 3. FILL MISSING VALUES
# ==============================
weather_cols = ['tavg', 'prcp', 'wspd', 'pres']
mean_values  = flights.select(*[avg(c).alias(c) for c in weather_cols]).first().asDict()
flights = flights.fillna(mean_values)
flights = flights.fillna({
    "Dep_Delay": 0, "Arr_Delay": 0,
    "DepTime_label": "Unknown", "Distance_type": "Unknown"
})

# ==============================
# 4. FEATURE ENGINEERING
# ==============================
print("[3/7] Engineering features...")

flights = flights \
    .withColumn("month",       month("FlightDate")) \
    .withColumn("day_of_week", dayofweek("FlightDate")) \
    .withColumn("delay_label", when(flights.Arr_Delay > 15, 1).otherwise(0))

# Target encoding
dep_delay_avg = flights.groupBy("Dep_Airport").agg(
    avg("delay_label").alias("dep_airport_delay_avg")
)
arr_delay_avg = flights.groupBy("Arr_Airport").agg(
    avg("delay_label").alias("arr_airport_delay_avg")
)
flights = flights.join(dep_delay_avg, "Dep_Airport", "left")
flights = flights.join(arr_delay_avg, "Arr_Airport", "left")

# ==============================
# 5. SAVE LOOKUP TABLES
# ==============================
print("[4/7] Saving airport delay lookup for Flask auto-fill...")

dep_delay_avg.toPandas().to_csv(
    "airport_delay_lookup.csv", index=False
)
airports.select("IATA_CODE", "LATITUDE", "LONGITUDE").toPandas().to_csv(
    "airport_coords.csv", index=False
)
print("      [SAVED] airport_delay_lookup.csv")
print("      [SAVED] airport_coords.csv")

# ==============================
# 6. BUILD COMBINED PIPELINE
# ==============================
print("[5/7] Building combined Spark ML Pipeline...")

categorical_cols = ["Airline", "DepTime_label", "Distance_type"]
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    for c in categorical_cols
]
encoder = OneHotEncoder(
    inputCols=[c + "_idx" for c in categorical_cols],
    outputCols=[c + "_vec" for c in categorical_cols]
)
assembler = VectorAssembler(
    inputCols=[
        "tavg", "prcp", "wspd", "pres",
        "LATITUDE", "LONGITUDE",
        "month", "day_of_week",
        "dep_airport_delay_avg", "arr_airport_delay_avg",
        "Airline_vec", "DepTime_label_vec", "Distance_type_vec"
    ],
    outputCol="features",
    handleInvalid="skip"
)
gbt = GBTClassifier(
    labelCol="delay_label",
    featuresCol="features",
    maxIter=10, maxDepth=4, stepSize=0.1, seed=42
)

full_pipeline = Pipeline(stages=indexers + [encoder, assembler, gbt])

# ==============================
# 7. FIT AND SAVE
# ==============================
print("[6/7] Training combined GBT pipeline...")

train_df, test_df = flights.randomSplit([0.8, 0.2], seed=42)
pipeline_model = full_pipeline.fit(train_df)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
test_preds = pipeline_model.transform(test_df)
acc = MulticlassClassificationEvaluator(
    labelCol="delay_label", predictionCol="prediction", metricName="accuracy"
).evaluate(test_preds)
auc = BinaryClassificationEvaluator(
    labelCol="delay_label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
).evaluate(test_preds)

print(f"\n  {'-'*50}")
print(f"  [GBT Full Pipeline]")
print(f"  Accuracy : {acc * 100:.2f}%")
print(f"  AUC-ROC  : {auc:.4f}")
print(f"  {'-'*50}\n")

print("[7/7] Saving GBT full pipeline...")
pipeline_model.write().overwrite().save("gbt_full_pipeline")
print("      [SAVED] gbt_full_pipeline/")

print("\n>>> Spark UI is still live at http://localhost:4040")
print(">>> Take your DAG screenshots now.")
input(">>> Press ENTER when done to shut down Spark...\n")

spark.stop()

print(f"\n{'='*60}")
print(f"  DONE in {time.time() - start:.1f}s")
print("  Now run: cd 'AI-BIG DATA Project' && python main.py")
print(f"{'='*60}\n")
