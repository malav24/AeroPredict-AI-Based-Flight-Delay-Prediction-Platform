import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

try:
    flights_2023 = pd.read_csv(r"D:\Semester 6\Big Data Visualization and Data Analytics\Project\flight_delay\flight_delay\US_flights_2023.csv")
    flights_2024 = pd.read_csv(r"D:\Semester 6\Big Data Visualization and Data Analytics\Project\flight_delay\flight_delay\maj us flight - january 2024.csv")
    airports = pd.read_csv(r"D:\Semester 6\Big Data Visualization and Data Analytics\Project\flight_delay\flight_delay\airports_geolocation.csv")
    weather = pd.read_csv(r"D:\Semester 6\Big Data Visualization and Data Analytics\Project\flight_delay\flight_delay\weather_meteo_by_airport.csv")
    cancelled = pd.read_csv(r"D:\Semester 6\Big Data Visualization and Data Analytics\Project\flight_delay\flight_delay\Cancelled_Diverted_2023.csv")
    print("All datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")

# STEP 2: Combine Flight Data
flights = pd.concat([flights_2023, flights_2024], ignore_index=True)

# STEP 3: Remove Cancelled Flights
cancelled_ids = cancelled[['FlightDate', 'Airline', 'Tail_Number']]

flights = flights.merge(
    cancelled_ids,
    on=['FlightDate', 'Airline', 'Tail_Number'],
    how='left',
    indicator=True
)

flights = flights[flights['_merge'] == 'left_only'].drop(columns=['_merge'])

# STEP 4: Basic Cleaning
flights.drop_duplicates(inplace=True)

# STEP 5: Convert Dates
flights['FlightDate'] = pd.to_datetime(flights['FlightDate'])
weather.columns = weather.columns.str.strip()  # fix column spacing issue
weather['time'] = pd.to_datetime(weather['time'])

# STEP 6: Merge Airport Data
flights = flights.merge(
    airports[['IATA_CODE', 'LATITUDE', 'LONGITUDE']],
    left_on='Dep_Airport',
    right_on='IATA_CODE',
    how='left'
).drop(columns=['IATA_CODE'])

# STEP 7: Merge Weather Data
flights = flights.merge(
    weather,
    left_on=['Dep_Airport', 'FlightDate'],
    right_on=['airport_id', 'time'],
    how='left'
).drop(columns=['airport_id', 'time'])

# STEP 8: Handle Missing Weather (Smart Imputation)
weather_cols = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']

for col in weather_cols:
    flights[col] = flights.groupby('Dep_Airport')[col].transform(
        lambda x: x.fillna(x.mean())
    )

# fallback for completely missing airports
flights[weather_cols] = flights[weather_cols].fillna(flights[weather_cols].median())

# STEP 9: Fill Other Missing Values
flights['Aicraft_age'] = flights['Aicraft_age'].fillna(flights['Aicraft_age'].median())

# STEP 10: Feature Engineering
flights['is_delayed'] = (flights['Arr_Delay'] > 15).astype(int)

flights['month'] = flights['FlightDate'].dt.month
flights['day'] = flights['FlightDate'].dt.day
flights['is_weekend'] = flights['Day_Of_Week'].isin([6, 7]).astype(int)

# STEP 11: Remove Data Leakage Columns 
leakage_cols = [
    'Arr_Delay', 'Dep_Delay', 'Dep_Delay_Tag', 'Dep_Delay_Type',
    'Arr_Delay_Type', 'Delay_Carrier', 'Delay_Weather',
    'Delay_NAS', 'Delay_Security', 'Delay_LastAircraft'
]

flights.drop(columns=leakage_cols, inplace=True, errors='ignore')

# STEP 12: Drop Unnecessary Columns
flights.drop(columns=[
    'Tail_Number', 'Dep_CityName', 'Arr_CityName',
    'Manufacturer', 'Model'
], inplace=True, errors='ignore')

# STEP 13: Encode Categorical Data

# Label Encoding (High Cardinality)
le_dep = LabelEncoder()
le_arr = LabelEncoder()

flights['Dep_Airport_Encoded'] = le_dep.fit_transform(flights['Dep_Airport'])
flights['Arr_Airport_Encoded'] = le_arr.fit_transform(flights['Arr_Airport'])

# Save encoders (important for deployment)
pickle.dump(le_dep, open('dep_encoder.pkl', 'wb'))
pickle.dump(le_arr, open('arr_encoder.pkl', 'wb'))

# One-Hot Encoding (Low Cardinality)
low_card_cols = ['Airline', 'DepTime_label', 'Distance_type']
flights = pd.get_dummies(flights, columns=low_card_cols, drop_first=True)

# Drop original airport columns
flights.drop(columns=['Dep_Airport', 'Arr_Airport'], inplace=True)

# STEP 14: Final Feature Selection

base_features = [
    'Dep_Airport_Encoded', 'Arr_Airport_Encoded',
    'Day_Of_Week', 'Flight_Duration', 'Aicraft_age',
    'tavg', 'prcp', 'wspd', 'pres',
    'month', 'day', 'is_weekend',
    'LATITUDE', 'LONGITUDE'
]

# auto-detect one-hot columns
one_hot_cols = [
    col for col in flights.columns
    if col.startswith(('Airline_', 'DepTime_label_', 'Distance_type_'))
]

final_features = base_features + one_hot_cols

final_df = flights[final_features + ['is_delayed']]

# final safety fill
final_df.fillna(0, inplace=True)

# STEP 15: Train-Test Split

X = final_df.drop(columns=['is_delayed'])
y = final_df['is_delayed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Preprocessing Complete")
print(f"Training Shape: {X_train.shape}")
print(f"Total Features: {len(X.columns)}")

# STEP 16: Save Processed Data
final_df.to_csv("preprocessed_flight_data_final.csv", index=False)