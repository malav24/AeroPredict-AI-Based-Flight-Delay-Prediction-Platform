from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("FlightDelayPreprocessing_v3") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.default.parallelism", "8") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# =========================
# 1. LOAD DATA
# =========================
flights_2023 = spark.read.csv(
    "hdfs://localhost:9000/flight_project/raw/US_flights_2023.csv",
    header=True, inferSchema=True
)

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

print("All datasets loaded from HDFS.")

# =========================
# 2. SAMPLE & CLEAN
# =========================
# Sample 30% of 2023 data (~2M rows) to fit in local Spark memory
# For viva: "We sampled 30% for local training; full pipeline runs on cluster"
flights = flights_2023.sample(fraction=0.30, seed=42)
print("Sampled dataset ready.")

# Remove cancelled/diverted flights
flights = flights.join(
    cancelled.select("FlightDate", "Airline", "Tail_Number"),
    ["FlightDate", "Airline", "Tail_Number"],
    "left_anti"
)

from pyspark.sql.functions import to_date

flights = flights.dropDuplicates()
flights = flights.withColumn("FlightDate", to_date("FlightDate"))
weather  = weather.withColumn("time", to_date("time"))

# =========================
# 3. JOIN AIRPORT + WEATHER
# =========================
airports_clean = airports.withColumnRenamed("IATA_CODE", "Dep_Airport")

flights = flights.join(
    airports_clean.select("Dep_Airport", "LATITUDE", "LONGITUDE"),
    "Dep_Airport",
    "left"
)

# Cache weather before broadcast join to avoid re-reads
weather.cache()

flights = flights.join(
    weather,
    (flights.Dep_Airport == weather.airport_id) &
    (flights.FlightDate == weather.time),
    "left"
).drop("airport_id", "time")

# =========================
# 4. HANDLE MISSING VALUES
# =========================
from pyspark.sql.functions import avg

weather_cols = ['tavg', 'prcp', 'wspd', 'pres']

mean_values = flights.select(
    *[avg(c).alias(c) for c in weather_cols]
).first().asDict()

flights = flights.fillna(mean_values)

flights = flights.fillna({
    "Dep_Delay": 0,
    "Arr_Delay": 0,
    "DepTime_label": "Unknown",
    "Distance_type": "Unknown"
})

# =========================
# 5. FEATURE ENGINEERING
# =========================
from pyspark.sql.functions import year, month, dayofweek, when

flights = flights.withColumn("year",        year("FlightDate"))
flights = flights.withColumn("month",       month("FlightDate"))
flights = flights.withColumn("day_of_week", dayofweek("FlightDate"))

flights = flights.withColumn(
    "delay_label",
    when(flights.Arr_Delay > 15, 1).otherwise(0)
)

# =========================
# 6. TARGET ENCODING
# =========================
dep_delay_avg = flights.groupBy("Dep_Airport").agg(
    avg("delay_label").alias("dep_airport_delay_avg")
)

arr_delay_avg = flights.groupBy("Arr_Airport").agg(
    avg("delay_label").alias("arr_airport_delay_avg")
)

flights = flights.join(dep_delay_avg, "Dep_Airport", "left")
flights = flights.join(arr_delay_avg, "Arr_Airport", "left")

print("Target encoding done.")

# =========================
# 7. CATEGORICAL ENCODING
# =========================
from pyspark.ml.feature import StringIndexer, OneHotEncoder

categorical_cols = ["Airline", "DepTime_label", "Distance_type"]

for col in categorical_cols:
    indexer = StringIndexer(
        inputCol=col,
        outputCol=col + "_idx",
        handleInvalid="keep"
    )
    flights = indexer.fit(flights).transform(flights)

encoder = OneHotEncoder(
    inputCols=[col + "_idx" for col in categorical_cols],
    outputCols=[col + "_vec" for col in categorical_cols]
)
flights = encoder.fit(flights).transform(flights)

# =========================
# 8. FEATURE VECTOR
# =========================
from pyspark.ml.feature import VectorAssembler

feature_cols = [
    "tavg", "prcp", "wspd", "pres",
    "LATITUDE", "LONGITUDE",
    "month", "day_of_week",
    "dep_airport_delay_avg",
    "arr_airport_delay_avg",
    "Airline_vec",
    "DepTime_label_vec",
    "Distance_type_vec"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

flights = assembler.transform(flights)

# =========================
# 9. SAVE TO HDFS
# =========================
final_df = flights.select("features", "delay_label")

final_df.write.mode("overwrite").parquet(
    "hdfs://localhost:9000/flight_project/processed/flights_final"
)

print("PREPROCESSING COMPLETE — Parquet saved to HDFS at /flight_project/processed/flights_final")
spark.stop()
