"""
AeroPredict - Flask HTML Frontend
===================================
Uses the GBT Full Pipeline (Spark MLlib) for predictions when available,
falls back to SVM (scikit-learn) otherwise.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os, json, joblib
import numpy as np
from datetime import datetime, date
from functools import lru_cache
from pathlib import Path
from pymongo import MongoClient

app = Flask(__name__)

# ==============================
# MONGODB CONNECTION
# ==============================
mongo_ok  = False
mongo_col = None
try:
    _mc = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    _mc.server_info()                          # test connection
    mongo_col = _mc["aeropredict"]["predictions"]
    mongo_ok  = True
    print("MongoDB connected — aeropredict.predictions")
except Exception as _me:
    print(f"MongoDB not available: {_me}")

# ==============================
# SPARK + GBT MODEL (optional)
# ==============================
print("Starting Spark session...")
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
    .appName("AeroPredict_Flask") \
    .config("spark.driver.memory", "3g") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

PIPELINE_PATH = "gbt_full_pipeline"
DELAY_LOOKUP  = "airport_delay_lookup.csv"
COORDS_LOOKUP = "airport_coords.csv"
WEATHER_LOOKUP= "weather_averages.csv"
DATA_PATH     = "flight_delay/preprocessed_flight_datav2.csv"
VIZ_DATA_DIR  = Path(app.static_folder) / "viz_data"

gbt_loaded = False
pipeline_model = None
try:
    pipeline_model = PipelineModel.load(PIPELINE_PATH)
    gbt_loaded = True
    print("GBT PipelineModel loaded.")
except Exception as e:
    print(f"GBT not available: {e}")

# ==============================
# SVM FALLBACK MODEL
# ==============================
svm_loaded  = False
svm_model   = None
svm_scaler  = None
svm_columns = None
try:
    svm_model   = joblib.load("aeropredict_model.pkl")
    svm_scaler  = joblib.load("scaler.pkl")
    svm_columns = joblib.load("model_columns.pkl")
    svm_loaded  = True
    print("SVM fallback model loaded.")
except Exception as e:
    print(f"SVM fallback not available: {e}")

model_loaded = gbt_loaded or svm_loaded

# ==============================
# REGRESSION MODEL (delay mins)
# ==============================
reg_loaded    = False
reg_model     = None
reg_scaler    = None
reg_columns   = None
try:
    reg_model   = joblib.load("delay_regressor.pkl")
    reg_scaler  = joblib.load("regressor_scaler.pkl")
    reg_columns = joblib.load("regressor_columns.pkl")
    reg_loaded  = True
    print("Delay regressor loaded.")
except Exception as e:
    print(f"Regressor not available (run train_regressor.py): {e}")

# ==============================
# LOOKUP TABLES
# ==============================
delay_lookup   = {}
coords_lookup  = {}
weather_lookup = {}
airports_list  = []

try:
    df_delay = pd.read_csv(DELAY_LOOKUP)
    delay_lookup = dict(zip(df_delay["Dep_Airport"], df_delay["dep_airport_delay_avg"]))
    df_coords = pd.read_csv(COORDS_LOOKUP)
    for _, r in df_coords.iterrows():
        coords_lookup[r["IATA_CODE"]] = (float(r["LATITUDE"]), float(r["LONGITUDE"]))
    airports_list = sorted(delay_lookup.keys())
    print(f"Loaded {len(airports_list)} airports.")
except Exception as e:
    print(f"Lookup load error: {e}")
    airports_list = ["ATL","DFW","LAX","ORD","DEN","JFK","SFO","BOS","EWR","MIA"]

try:
    df_w = pd.read_csv(WEATHER_LOOKUP)
    for _, r in df_w.iterrows():
        key = (str(r.get("airport_id", r.get("Dep_Airport", ""))), int(r.get("month", 0)))
        weather_lookup[key] = (
            float(r.get("tavg", 15.0)), float(r.get("prcp", 2.0)),
            float(r.get("wspd", 15.0)), float(r.get("pres", 1013.0))
        )
    print("Weather lookup loaded.")
except Exception as e:
    print(f"Weather load error: {e}")

# ==============================
# AIRLINE DISPLAY MAP
# ==============================
AIRLINE_DISPLAY = {
    'Alaska Airlines Inc.'      : 'AS — Alaska Airlines',
    'Allegiant Air'             : 'G4 — Allegiant Air',
    'American Airlines Inc.'    : 'AA — American Airlines',
    'American Eagle Airlines Inc.': 'MQ — American Eagle',
    'Delta Air Lines Inc'       : 'DL — Delta Air Lines',
    'Endeavor Air'              : '9E — Endeavor Air',
    'Frontier Airlines Inc.'    : 'F9 — Frontier Airlines',
    'Hawaiian Airlines Inc.'    : 'HA — Hawaiian Airlines',
    'JetBlue Airways'           : 'B6 — JetBlue Airways',
    'PSA Airlines'              : 'OH — PSA Airlines',
    'Republic Airways'          : 'YX — Republic Airways',
    'Skywest Airlines Inc.'     : 'OO — SkyWest Airlines',
    'Southwest Airlines Co.'    : 'WN — Southwest Airlines',
    'Spirit Air Lines'          : 'NK — Spirit Air Lines',
    'United Air Lines Inc.'     : 'UA — United Airlines',
}

analytics_stats = {
    "total_flights" : 1966327,
    "avg_delay"     : 12.4,
    "model_accuracy": "80.88%",
    "model_type"    : "GBT (Spark MLlib)",
    "airlines"      : len(AIRLINE_DISPLAY),
    "airports"      : len(airports_list),
}

# ==============================
# HELPERS
# ==============================
DAY_ORDER = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
TIME_ORDER = ['Morning', 'Afternoon', 'Evening', 'Night']


def _records(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@lru_cache(maxsize=1)
def load_visualization_payload():
    temp_df = pd.DataFrame(_records(VIZ_DATA_DIR / "weather_vs_delay.json"))
    temp_df = temp_df.rename(columns={"tavg": "temperature_c", "Arr_Delay": "arrival_delay"})
    temp_df["arrival_delay"] = pd.to_numeric(temp_df["arrival_delay"], errors="coerce")
    temp_df["temperature_c"] = pd.to_numeric(temp_df["temperature_c"], errors="coerce")
    temp_df = temp_df.dropna(subset=["temperature_c", "arrival_delay"])
    temp_df = temp_df[temp_df["temperature_c"].between(-35, 45)]
    delay_floor = float(temp_df["arrival_delay"].quantile(0.02))
    delay_ceiling = float(temp_df["arrival_delay"].quantile(0.98))
    temp_df["arrival_delay_clip"] = temp_df["arrival_delay"].clip(delay_floor, delay_ceiling)
    temp_df["temp_band"] = (np.round(temp_df["temperature_c"] / 2) * 2).astype(int)
    temp_trend = (
        temp_df.groupby("temp_band")
        .agg(
            avg_delay=("arrival_delay_clip", "mean"),
            median_delay=("arrival_delay_clip", "median"),
            flight_count=("arrival_delay", "size"),
        )
        .reset_index()
        .sort_values("temp_band")
    )
    temp_trend = temp_trend[temp_trend["flight_count"] >= 40]
    temp_points = temp_df.sample(min(len(temp_df), 2000), random_state=42)

    prcp_df = pd.DataFrame(_records(VIZ_DATA_DIR / "scatterplot_weather.json"))
    prcp_df["prcp"] = pd.to_numeric(prcp_df["prcp"], errors="coerce")
    prcp_df["Average_Delay"] = pd.to_numeric(prcp_df["Average_Delay"], errors="coerce")
    prcp_df = prcp_df.dropna(subset=["prcp", "Average_Delay"])
    prcp_df = prcp_df[prcp_df["prcp"] <= prcp_df["prcp"].quantile(0.99)]
    prcp_df["delay_clip"] = prcp_df["Average_Delay"].clip(
        prcp_df["Average_Delay"].quantile(0.02),
        prcp_df["Average_Delay"].quantile(0.98),
    )
    prcp_df["rain_band"] = (np.floor(prcp_df["prcp"] / 2) * 2).astype(int)
    prcp_trend = (
        prcp_df.groupby("rain_band")
        .agg(avg_delay=("delay_clip", "mean"), point_count=("delay_clip", "size"))
        .reset_index()
        .sort_values("rain_band")
    )

    map_df = pd.DataFrame(_records(VIZ_DATA_DIR / "airport_delay_map.json"))
    map_df["Average_Delay"] = pd.to_numeric(map_df["Average_Delay"], errors="coerce").fillna(0.0)
    map_df["Average_Delay_Display"] = map_df["Average_Delay"].clip(-15, 45)
    map_df["Bubble_Size"] = 10 + np.clip(map_df["Average_Delay_Display"], 0, None) * 0.8

    heatmap_df = pd.DataFrame(_records(VIZ_DATA_DIR / "delay_heatmap.json"))
    heatmap_df["Day"] = pd.Categorical(heatmap_df["Day"], categories=DAY_ORDER, ordered=True)
    heatmap_df["DepTime_label"] = pd.Categorical(
        heatmap_df["DepTime_label"], categories=TIME_ORDER, ordered=True
    )
    heatmap_df = heatmap_df.sort_values(["Day", "DepTime_label"])
    heatmap_matrix = (
        heatmap_df.pivot(index="Day", columns="DepTime_label", values="Average_Delay")
        .reindex(index=DAY_ORDER, columns=TIME_ORDER)
    )

    monthly_avg_df = pd.DataFrame(_records(VIZ_DATA_DIR / "monthly_delay.json"))
    monthly_count_df = pd.DataFrame(_records(VIZ_DATA_DIR / "monthly_delay_count.json"))
    airline_df = pd.DataFrame(_records(VIZ_DATA_DIR / "top_delayed_airlines.json")).sort_values(
        "Average_Arrival_Delay", ascending=False
    )

    peak_month = monthly_avg_df.loc[
        monthly_avg_df["Average_Arrival_Delay_Minutes"].idxmax()
    ]
    top_airline = airline_df.iloc[0]
    busiest_delay_month = monthly_count_df.loc[monthly_count_df["Total_Delays"].idxmax()]

    return {
        "summary": {
            "peak_month": {
                "label": str(peak_month["Month"]),
                "value": round(float(peak_month["Average_Arrival_Delay_Minutes"]), 2),
            },
            "top_airline": {
                "label": str(top_airline["Airline"]),
                "value": round(float(top_airline["Average_Arrival_Delay"]), 2),
            },
            "delay_volume_peak": {
                "label": str(busiest_delay_month["Month"]),
                "value": int(busiest_delay_month["Total_Delays"]),
            },
        },
        "temperature_delay": {
            "trend": temp_trend.to_dict(orient="records"),
            "sample": temp_points[["temperature_c", "arrival_delay_clip"]].rename(
                columns={"arrival_delay_clip": "arrival_delay"}
            ).to_dict(orient="records"),
        },
        "precipitation_delay": prcp_trend.to_dict(orient="records"),
        "airport_map": map_df[
            ["Dep_Airport", "LATITUDE", "LONGITUDE", "Average_Delay", "Average_Delay_Display", "Bubble_Size"]
        ].to_dict(orient="records"),
        "delay_heatmap": {
            "days": DAY_ORDER,
            "times": TIME_ORDER,
            "values": heatmap_matrix.fillna(0).round(2).values.tolist(),
        },
        "monthly_delay": monthly_avg_df.to_dict(orient="records"),
        "monthly_delay_count": monthly_count_df.to_dict(orient="records"),
        "top_airlines": airline_df.to_dict(orient="records"),
    }


def get_weather(airport, month):
    for key in weather_lookup:
        if key[0] == airport and key[1] == month:
            return weather_lookup[key]
    return (15.0, 2.0, 15.0, 1013.0)

def get_coords(airport):
    if airport in coords_lookup:
        return coords_lookup[airport]
    return (39.0, -98.0)

def get_delay_avg(airport):
    return float(delay_lookup.get(airport, 0.20))

def log_prediction(doc):
    """Safely insert a prediction document into MongoDB."""
    if not mongo_ok:
        return
    try:
        mongo_col.insert_one(doc)
    except Exception as e:
        print(f"MongoDB insert error: {e}")

def get_route_stats(dep, arr):
    dep_c = coords_lookup.get(dep, (39.0, -98.0))
    arr_c = coords_lookup.get(arr, (41.9, -87.9))
    lat_d = abs(float(dep_c[0]) - float(arr_c[0]))
    lon_d = abs(float(dep_c[1]) - float(arr_c[1]))
    est_dist = ((lat_d**2 + lon_d**2) ** 0.5) * 69
    est_dur  = max(30, int(est_dist / 8))
    if est_dist < 800:
        dist_type = 'Short Haul >1500Mi'
    elif est_dist < 1500:
        dist_type = 'Medium Haul <3000Mi'
    else:
        dist_type = 'Long Haul <6000Mi'
    return est_dur, dist_type

def parse_date(date_str):
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        return dt.month, dt.day, dt.weekday()+1, days[dt.weekday()], dt.weekday() >= 5
    except:
        return 1, 1, 1, 'Monday', False

def svm_predict(airline_key, dep_airport, arr_airport, dep_time, month, day,
                day_of_week, is_weekend, tavg, prcp, wspd, pres, lat, lon, duration, dist_type):
    row = {c: 0 for c in svm_columns}
    ak = f'Airline_{airline_key}'
    dt = f'DepTime_label_{dep_time}'
    di = f'Distance_type_{dist_type}'
    if ak in row: row[ak] = 1
    if dt in row: row[dt] = 1
    if di in row: row[di] = 1
    row.update({
        'month': month, 'day': day, 'Day_Of_Week': day_of_week,
        'is_weekend': int(is_weekend), 'Flight_Duration': duration,
        'Aicraft_age': 12, 'tavg': tavg, 'prcp': prcp,
        'wspd': wspd, 'pres': pres, 'LATITUDE': lat, 'LONGITUDE': lon,
    })
    df = pd.DataFrame([row])[svm_columns]
    X  = svm_scaler.transform(df.values)
    p  = svm_model.predict_proba(X)[0]
    return float(p[1]), float(p[0])

def regressor_predict(airline_key, dep_time, dist_type, month, day, day_of_week,
                      is_weekend, tavg, prcp, wspd, pres, lat, lon, duration):
    """Predict estimated delay minutes using the GBT Regressor."""
    if not reg_loaded:
        return None
    try:
        row = {c: 0 for c in reg_columns}
        ak = f'Airline_{airline_key}'
        dt = f'DepTime_label_{dep_time}'
        di = f'Distance_type_{dist_type}'
        if ak in row: row[ak] = 1
        if dt in row: row[dt] = 1
        if di in row: row[di] = 1
        row.update({
            'month': month, 'day': day, 'Day_Of_Week': day_of_week,
            'is_weekend': int(is_weekend), 'Flight_Duration': duration,
            'Aicraft_age': 12, 'tavg': tavg, 'prcp': prcp,
            'wspd': wspd, 'pres': pres, 'LATITUDE': lat, 'LONGITUDE': lon,
        })
        df  = pd.DataFrame([row])[[c for c in reg_columns if c in row]]
        # align to exact training columns
        df  = df.reindex(columns=reg_columns, fill_value=0)
        X   = reg_scaler.transform(df.values)
        est = float(reg_model.predict(X)[0])
        return max(0, round(est))
    except Exception as e:
        print(f"Regressor predict error: {e}")
        return None

# ==============================
# ROUTES
# ==============================
@app.route("/api/airlines")
def api_airlines():
    return jsonify(list(AIRLINE_DISPLAY.keys()))

@app.route("/api/airports")
def api_airports():
    return jsonify(airports_list)

@app.route("/api/autofill")
def api_autofill():
    dep      = request.args.get('dep', '').upper().strip()
    arr      = request.args.get('arr', '').upper().strip()
    date_str = request.args.get('date', '')
    dep_time = request.args.get('dep_time', 'Morning')

    month, day, dow, day_name, is_wknd = parse_date(date_str)
    tavg, prcp, wspd, pres = get_weather(dep, month)
    est_dur, dist_type     = get_route_stats(dep, arr)

    return jsonify({
        'day_of_week'     : day_name,
        'is_weekend'      : 'Yes' if is_wknd else 'No',
        'flight_duration' : est_dur,
        'route_type'      : dist_type,
        'tavg'            : round(tavg, 2),
        'prcp'            : round(prcp, 2),
        'wspd'            : round(wspd, 2),
        'pres'            : round(pres, 2),
        'aircraft_age'    : 12,
        'dep_airport'     : dep,
    })

@app.context_processor
def inject_globals():
    """Inject airlines + airports into every template automatically."""
    return dict(airlines=AIRLINE_DISPLAY, airports=airports_list)

@app.route("/")
def home():
    return render_template("home.html", stats=analytics_stats)

@app.route("/visualizations")
def visualizations():
    return render_template("visualizations.html", stats=analytics_stats)

@app.route("/settings")
def settings():
    return render_template("settings.html", stats=analytics_stats)

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html",
        stats=analytics_stats,
        airlines=AIRLINE_DISPLAY,
        airports=airports_list,
        model_loaded=model_loaded,
        prediction=None,
    )

@app.route("/predict", methods=["POST"])
def predict():
    airline_key = request.form.get("airline_key", "").strip()
    dep_airport = request.form.get("dep_airport", "").upper().strip()
    arr_airport = request.form.get("arr_airport", "").upper().strip()
    date_str    = request.form.get("travel_date", "")
    dep_time    = request.form.get("dep_time", "Morning")

    if not airline_key or not dep_airport or not arr_airport:
        return render_template("predict.html",
            stats=analytics_stats, airlines=AIRLINE_DISPLAY,
            airports=airports_list, model_loaded=model_loaded,
            error="Please fill in Airline, Departure and Arrival airports.",
            prediction=None,
        )

    month, day, dow, day_name, is_wknd = parse_date(date_str)
    tavg, prcp, wspd, pres = get_weather(dep_airport, month)
    lat, lon               = get_coords(dep_airport)
    est_dur, dist_type     = get_route_stats(dep_airport, arr_airport)

    autofill = {
        'day_of_week'    : day_name,
        'is_weekend'     : 'Yes' if is_wknd else 'No',
        'flight_duration': est_dur,
        'route_type'     : dist_type,
        'tavg'           : round(float(tavg), 2),
        'prcp'           : round(float(prcp), 2),
        'wspd'           : round(float(wspd), 2),
        'pres'           : round(float(pres), 2),
        'aircraft_age'   : 12,
    }

    try:
        # Try GBT first, fall back to SVM
        if gbt_loaded:
            dep_delay_avg = get_delay_avg(dep_airport)
            arr_delay_avg = get_delay_avg(arr_airport)
            schema_row = [{
                "Airline": airline_key, "DepTime_label": dep_time,
                "Distance_type": dist_type, "tavg": float(tavg),
                "prcp": float(prcp), "wspd": float(wspd), "pres": float(pres),
                "LATITUDE": float(lat), "LONGITUDE": float(lon),
                "month": int(month), "day_of_week": int(dow),
                "dep_airport_delay_avg": float(dep_delay_avg),
                "arr_airport_delay_avg": float(arr_delay_avg),
                "delay_label": 0
            }]
            pred_df    = spark.createDataFrame(schema_row)
            result_row = pipeline_model.transform(pred_df).select("prediction","probability").first()
            prob_arr   = result_row["probability"]
            delay_prob   = float(prob_arr[1])
            on_time_prob = float(prob_arr[0])
        elif svm_loaded:
            delay_prob, on_time_prob = svm_predict(
                airline_key, dep_airport, arr_airport, dep_time,
                month, day, dow, is_wknd, float(tavg), float(prcp),
                float(wspd), float(pres), float(lat), float(lon),
                est_dur, dist_type
            )
        else:
            raise Exception("No model available")

        # Risk classification
        if delay_prob >= 0.40:
            risk, icon, big_label, verdict, color = \
                "high","🔴","DELAYED","High Delay Risk","#ef4444"
        elif delay_prob >= 0.25:
            risk, icon, big_label, verdict, color = \
                "medium","⚠️","POSSIBLY DELAYED","Moderate Risk — Monitor Flight","#f59e0b"
        else:
            risk, icon, big_label, verdict, color = \
                "low","✅","ON TIME","Low Risk","#10b981"

        # ── Regression: estimate delay minutes if flight is delayed ──
        est_delay_mins = None
        if risk in ('high', 'medium'):
            est_delay_mins = regressor_predict(
                airline_key, dep_time, dist_type,
                month, day, dow, is_wknd,
                float(tavg), float(prcp), float(wspd), float(pres),
                float(lat), float(lon), est_dur
            )

        prediction = {
            'delay_prob'    : round(delay_prob * 100, 1),
            'on_time_prob'  : round(on_time_prob * 100, 1),
            'risk'          : risk,
            'icon'          : icon,
            'big_label'     : big_label,
            'verdict'       : verdict,
            'color'         : color,
            'airline_display': AIRLINE_DISPLAY.get(airline_key, airline_key),
            'route'         : f"{dep_airport} → {arr_airport}",
            'date_display'  : date_str,
            'dep_time'      : dep_time,
            'est_delay_mins': est_delay_mins,
        }

        # ── Log to MongoDB ──
        log_prediction({
            "timestamp"      : datetime.utcnow(),
            "source"         : "flask",
            "airline"        : AIRLINE_DISPLAY.get(airline_key, airline_key),
            "dep_airport"    : dep_airport,
            "arr_airport"    : arr_airport,
            "route"          : f"{dep_airport} → {arr_airport}",
            "travel_date"    : date_str,
            "dep_time"       : dep_time,
            "delay_prob"     : round(delay_prob * 100, 1),
            "on_time_prob"   : round(on_time_prob * 100, 1),
            "prediction"     : big_label,
            "risk"           : risk,
            "model_used"     : "GBT (Spark MLlib)" if gbt_loaded else "SVM (scikit-learn)",
            "flight_duration": est_dur,
            "route_type"     : dist_type,
            "weather"        : {"tavg": round(float(tavg),2), "prcp": round(float(prcp),2),
                                "wspd": round(float(wspd),2), "pres": round(float(pres),2)},
        })

        return render_template("predict.html",
            stats=analytics_stats, airlines=AIRLINE_DISPLAY,
            airports=airports_list, model_loaded=model_loaded,
            prediction=prediction, autofill=autofill,
            form_values={
                'airline_key': airline_key, 'dep_airport': dep_airport,
                'arr_airport': arr_airport, 'travel_date': date_str,
                'dep_time': dep_time,
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("predict.html",
            stats=analytics_stats, airlines=AIRLINE_DISPLAY,
            airports=airports_list, model_loaded=model_loaded,
            error=f"Prediction error: {str(e)}", prediction=None,
        )

@app.route("/history")
def history():
    records = []
    if mongo_ok:
        try:
            records = list(mongo_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(25))
        except Exception as e:
            print(f"MongoDB fetch error: {e}")
    return render_template("history.html", records=records, stats=analytics_stats,
                           mongo_ok=mongo_ok)

@app.route("/api/analytics")
def get_analytics():
    return jsonify(analytics_stats)

@app.route("/api/visualization-data")
def visualization_data():
    return jsonify(_json_ready(load_visualization_payload()))

@app.errorhandler(404)
def not_found(error):
    return render_template("home.html", stats=analytics_stats), 404

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
