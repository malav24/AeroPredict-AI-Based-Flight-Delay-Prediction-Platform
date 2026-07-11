import pandas as pd

# 1. Monthly weather averages per airport
print('Computing weather averages...')
weather = pd.read_csv('flight_delay/weather_meteo_by_airport.csv')
weather['time'] = pd.to_datetime(weather['time'])
weather['month'] = weather['time'].dt.month
weather_avg = weather.groupby(['airport_id', 'month'])[['tavg','prcp','wspd','pres']].mean().round(2)
weather_avg.to_csv('weather_averages.csv')
print('Done. Sample:')
print(weather_avg.head(5).to_string())

# 2. Typical flight duration and distance type per route
print('\nComputing route stats...')
df = pd.read_csv('flight_delay/preprocessed_flight_datav2.csv',
                 usecols=['Dep_Airport','Arr_Airport','Flight_Duration','Distance_type'])
route_stats = df.groupby(['Dep_Airport','Arr_Airport']).agg(
    avg_duration=('Flight_Duration','median'),
    distance_type=('Distance_type', lambda x: x.mode()[0])
).reset_index()
route_stats.to_csv('route_stats.csv', index=False)
print(f'Route stats saved. {len(route_stats)} unique routes found.')

# 3. Aircraft age median
df2 = pd.read_csv('flight_delay/preprocessed_flight_datav2.csv', usecols=['Aicraft_age'])
median_age = df2['Aicraft_age'].median()
print(f'Median aircraft age: {median_age}')
