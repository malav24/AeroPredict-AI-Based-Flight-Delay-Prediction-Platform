# AeroPredict: Flight Delay Risk Classification & Regression Dashboard

AeroPredict is an end-to-end, industrial-grade **Big Data & Artificial Intelligence** application designed to predict flight delays, assess delay risk probabilities, and estimate exact late arrival times (in minutes). It features a high-performance distributed ETL engine powered by **Apache Spark**, a dual-tier ML model architecture (classification + regression) built with **PySpark MLlib** and **Scikit-Learn**, and a premium web dashboard built using **Flask**, **MongoDB**, and **Plotly**.

This project was developed as a final course project for **Big Data Visualization and Data Analytics (Semester 6)**.

---

## 🏗️ System Architecture

The following diagram outlines the data flow from raw storage to the real-time user interface:

```
                  +----------------------------------------------+
                  |  RAW DATASETS (6.5 Million Records in HDFS)  |
                  |  - US_flights_2023.csv (1.1 GB)              |
                  |  - weather_meteo_by_airport.csv (Daily)      |
                  |  - Cancelled_Diverted_2023.csv (Exclusions)  |
                  |  - airports_geolocation.csv (GPS coords)     |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |     Apache Spark Distributed ETL Engine      |
                  |     - Removes cancelled/diverted flights     |
                  |     - Attaches airport geolocations          |
                  |     - Joins weather variables on date+airport|
                  |     - Natively imputes missing sensor data   |
                  +----------------------------------------------+
                                    /            \
                                   /              \
                                  v                v
       +--------------------------------------+  +-------------------------------------+
       |   CLASSIFICATION PIPELINE (PySpark)  |  |  REGRESSION PIPELINE (Scikit-Learn)  |
       |   - GBT Classifier (Spark MLlib)     |  |  - Gradient Boosting Regressor      |
       |   - Learns delay risk & probability  |  |  - Trains on subset of delays       |
       |   - Fallback: SVM model (joblib)     |  |  - Estimates delay times in minutes |
       +--------------------------------------+  +-------------------------------------+
                                 \                /
                                  \              /
                                   v            v
                  +----------------------------------------------+
                  |            Flask Premium Frontend            |
                  |   - /predict-page: User enters flight data   |
                  |     (Autofills weather & flight metrics)     |
                  |   - /visualizations: Plotly analytics        |
                  |   - Settings: Engine toggle (Spark vs SVM)   |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |            MongoDB Logging System            |
                  |   - Stores all runtime predictions in DB     |
                  |   - Collection: aeropredict.predictions      |
                  +----------------------------------------------+
```

---

## 📊 Core Project Statistics

Below are the key figures and performance metrics computed on the flight logs:

| Metric | Value | Meaning / Context |
| :--- | :--- | :--- |
| **Total Raw Records** | **6.5 Million** flights | Total US domestic flights processed through HDFS. |
| **Cleaned Dataset** | **1.96 Million** flights | Count after filtering cancellations, diversions, and duplicates. |
| **HDFS Target Directory** | `hdfs://localhost:9000/flight_project/` | Active Hadoop Distributed File System repository. |
| **Spark partitions** | **16 Partitions** | The scale of parallel execution chunks across the cluster. |
| **Spark ETL Run Time** | **~16 seconds** | Time Spark takes to join, clean, and impute 6.5M rows. |
| **Classifier Accuracy** | **80.88%** | Predictive accuracy of the classification model. |
| **Risk Thresholds** | **Low ($\le$ 25%), Med (25% - 40%), High ($\ge$ 40%)** | Delay probability brackets. |
| **Regressor MAE** | **~18.2 minutes** | Mean Absolute Error for estimating late arrival times. |
| **MongoDB Port** | **27017** | Database connection port for runtime auditing. |

---

## 🧠 Key Features

### 1. Dual-Tier Hybrid Machine Learning
- **Tier 1 (Risk Classification)**: Uses a **Spark MLlib Gradient Boosted Trees (GBT) Classifier** (or a local Scikit-Learn SVM fallback) to evaluate the probability of a flight being delayed by more than 15 minutes.
- **Tier 2 (Delay Time Regression)**: If Tier 1 flags a **Medium** or **High** delay risk, a **Scikit-Learn Gradient Boosting Regressor** is triggered to estimate the delay length in minutes. This avoids model skew from the 80% of flights that arrive on time.

### 2. Live Interactive Visualizations
A dedicated dashboard rendering live interactive Plotly charts, including:
- **Geospatial Delay Hotspots**: An interactive maps dashboard displaying North American airport flight nodes colored by delay frequency.
- **Delay Heatmap**: Cross-references departure hours and days of the week to pinpoint peak-delay intervals.
- **Weather Correlation Graph**: Scatter sample mapping temperature and rainfall levels against delay lengths, indicating a strong correlation with precipitation (**16.47% feature importance**).

### 3. Autofill Intelligence
When booking details are entered, Flask queries static pre-computed lookup tables (`weather_averages.csv`, `route_stats.csv`, and `airport_coords.csv`) to automatically populate complex weather metrics and average flight times, ensuring a seamless user experience.

---

## 📁 Repository Structure

```
├── flight_delay/               # Raw flight dataset files (Ignored in Git except scripts)
│   ├── airports_geolocation.csv # Airport coordinates metadata (lat/lon)
│   ├── datapreprocessing.py     # Initial data processing scripts
│   └── datatraining.py          # Legacy training scripts
├── static/                     # Web app static assets
│   ├── styles.css               # Styling rules (modern dashboard design)
│   ├── script.js                # Frontend API calls & DOM handling
│   ├── viz_data/                # Pre-processed JSON outputs for Plotly charts
│   └── ...                      # High-res image assets & icons
├── templates/                  # Flask Jinja2 HTML templates
│   ├── index.html               # Home landing page (interactive cabin cards)
│   ├── predict.html             # Predictive UI form and results panel
│   ├── visualizations.html      # Plots dashboard
│   └── settings.html            # Backend toggle (Spark vs SVM) & MongoDB status
├── gbt_full_pipeline/          # Saved Spark MLlib GBT Classifier Model (Ignored)
├── aeropredict_model.pkl       # SVM Classifier fallback pickle
├── delay_regressor.pkl         # Gradient Boosting Regressor model pickle
├── scaler.pkl                  # SVM MinMax Scaler
├── regressor_scaler.pkl        # Regressor Standard Scaler
├── airport_coords.csv          # Lookups for airport locations
├── airport_delay_lookup.csv    # Lookups for airport historical delay ratios
├── route_stats.csv             # Lookups for typical flight times
├── weather_averages.csv        # Lookups for historical daily weather averages
├── flask_app.py                # Main Flask web application backend
├── train_model.py              # Script to train classifier fallback (SVM)
├── train_regressor.py          # Script to train delay regressor (GBT Regressor)
├── make_gbt_flask_pipeline.py  # Script to train Spark MLlib GBT model
├── requirements.txt            # Python dependencies
├── .gitignore                  # Standard git exclusions for large data/models
└── README.md                   # This project guide
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.10+**
- **Java JDK 8 or 11** (Required for Apache Spark)
- **Apache Spark 3.5.x** (with Hadoop binaries configured in your environment path)
- **MongoDB** (Running locally on port `27017`)

### 2. Environment Setup
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd AeroPredict
   ```

2. Create a virtual environment and install python dependencies:
   ```bash
   python -m venv venv
   source venv/Scripts/activate     # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ensure Hadoop and PySpark paths are declared:
   ```powershell
   $env:SPARK_HOME="C:\path\to\spark-3.5.x-bin-hadoop3"
   $env:PATH="$env:SPARK_HOME\bin;$env:PATH"
   ```

### 3. Database Initialization (Optional)
Ensure a local instance of MongoDB is running:
```bash
# Test connection via CLI or MongoDB Compass
mongosh "mongodb://localhost:27017"
```
*Note: Flask connects automatically. If MongoDB is offline, Flask will display a warning in the console and continue predicting in offline mode, bypassing database logs.*

---

## 🚀 How to Run

### 1. Model Training
*Note: Pre-trained SVM and Regressor pickles are included in the repository for immediate startup. If you wish to re-train the models, follow these commands:*

- **Train Spark ML GBT Classifier**:
  ```bash
  python make_gbt_flask_pipeline.py
  ```
- **Train Fallback SVM Classifier**:
  ```bash
  python train_model.py
  ```
- **Train Gradient Boosting Regressor**:
  ```bash
  python train_regressor.py
  ```

### 2. Start the Web Dashboard
Since the web application uses PySpark capabilities for the Tier-1 prediction model, it must be launched via `spark-submit`:
```bash
spark-submit flask_app.py
```
After the Spark session starts up, open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 🏫 Credits & Course Details
- **Course**: Big Data Visualization and Data Analytics (Semester 6)
- **Project Team**: [Your Name/Teammate Names]
- **Institution**: [Your University/College Name]
