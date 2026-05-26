import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# Load data
csv_path = "../Data/Energy_consumption_data.csv"
if not os.path.exists(csv_path):
    csv_path = "Data/Energy_consumption_data.csv"
data = pd.read_csv(csv_path)

# Extract Time Features from Timestamp
if 'Timestamp' in data.columns:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Hour'] = data['Timestamp'].dt.hour
    data['Day'] = data['Timestamp'].dt.day
    data['Month'] = data['Timestamp'].dt.month

day_mapping = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
data['DayOfWeek'] = data['DayOfWeek'].map(day_mapping)
holiday_mapping = {"Yes": 1, "No": 0}
data['Holiday'] = data['Holiday'].map(holiday_mapping)
binary_mapping = {"On": 1, "Off": 0}
data['HVACUsage'] = data['HVACUsage'].map(binary_mapping)
data['LightingUsage'] = data['LightingUsage'].map(binary_mapping)

def get_time_period(hour):
    if 5 <= hour < 12: return 0
    elif 12 <= hour < 17: return 1
    elif 17 <= hour < 21: return 2
    else: return 3

def get_weekend_label(dayofweek):
    return 1 if dayofweek >= 5 else 0

data['WeekendLabel'] = data['DayOfWeek'].apply(get_weekend_label)
data['TimePeriodLabel'] = data['Hour'].apply(get_time_period)

data['HourSin'] = np.sin(2 * np.pi * data['Hour'] / 24)
data['HourCos'] = np.cos(2 * np.pi * data['Hour'] / 24)
data['MonthSin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['MonthCos'] = np.cos(2 * np.pi * data['Month'] / 12)
data['DayOfWeekSin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
data['DayOfWeekCos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)

data['Temp_x_Occupancy']  = data['Temperature'] * data['Occupancy']
data['Temp_x_HVAC']       = data['Temperature'] * data['HVACUsage']
data['Temp_squared']      = data['Temperature'] ** 2
data['Occ_x_Lighting']    = data['Occupancy']   * data['LightingUsage']
data['Occ_x_HVAC']        = data['Occupancy']   * data['HVACUsage']
data['Energy_intensity']  = data['Occupancy']   / (data['SquareFootage'] + 1)
data['Renewable_ratio']   = data['RenewableEnergy'] / (data['Temperature'] + 1)

features = [
    'Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
    'HVACUsage', 'LightingUsage', 'RenewableEnergy',
    'DayOfWeek', 'Holiday', 'Hour', 'Day', 'Month',
    'WeekendLabel', 'TimePeriodLabel',
    'HourSin', 'HourCos', 'MonthSin', 'MonthCos',
    'DayOfWeekSin', 'DayOfWeekCos',
    'Temp_x_Occupancy', 'Temp_x_HVAC', 'Temp_squared',
    'Occ_x_Lighting', 'Occ_x_HVAC', 'Energy_intensity', 'Renewable_ratio'
]
target = 'EnergyConsumption'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# A set of configs to try
configs = [
    # 1. Higher n_estimators, lower learning_rate, deeper trees
    {"n_estimators": 1000, "learning_rate": 0.03, "max_depth": 6, "subsample": 0.8, "max_features": "sqrt", "min_samples_leaf": 2},
    {"n_estimators": 1500, "learning_rate": 0.02, "max_depth": 6, "subsample": 0.8, "max_features": "sqrt", "min_samples_leaf": 2},
    # 2. Deeper max_depth
    {"n_estimators": 800, "learning_rate": 0.05, "max_depth": 7, "subsample": 0.85, "max_features": "sqrt", "min_samples_leaf": 2},
    {"n_estimators": 1000, "learning_rate": 0.03, "max_depth": 7, "subsample": 0.8, "max_features": "sqrt", "min_samples_leaf": 2},
    # 3. Default features but tuned
    {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.85, "max_features": "sqrt", "min_samples_leaf": 3},
    {"n_estimators": 1200, "learning_rate": 0.03, "max_depth": 6, "subsample": 0.85, "max_features": "sqrt", "min_samples_leaf": 3},
    # 4. Standard features (no max_features limit)
    {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 5, "subsample": 0.8, "max_features": None, "min_samples_leaf": 3},
    {"n_estimators": 800, "learning_rate": 0.02, "max_depth": 5, "subsample": 0.8, "max_features": None, "min_samples_leaf": 3},
    # 5. Some extreme configs
    {"n_estimators": 1000, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8, "max_features": "sqrt", "min_samples_leaf": 2},
    {"n_estimators": 1200, "learning_rate": 0.04, "max_depth": 7, "subsample": 0.8, "max_features": "sqrt", "min_samples_leaf": 2},
]

print("Starting training...", flush=True)
for i, config in enumerate(configs):
    model = GradientBoostingRegressor(
        random_state=42,
        **config
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Config {i}: R2 = {r2:.6f} | Params: {config}", flush=True)
print("Finished all configs.", flush=True)
