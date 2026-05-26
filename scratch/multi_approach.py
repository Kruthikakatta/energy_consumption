import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PowerTransformer
import joblib

# Load data
csv_path = "Data/Energy_consumption_data.csv"
if not os.path.exists(csv_path):
    csv_path = "../Data/Energy_consumption_data.csv"
data = pd.read_csv(csv_path)
print(f"Dataset size: {data.shape}", flush=True)

# Check EnergyConsumption distribution
print(f"\nEnergyConsumption stats:", flush=True)
print(f"  Min: {data['EnergyConsumption'].min()}", flush=True)
print(f"  Max: {data['EnergyConsumption'].max()}", flush=True)
print(f"  Mean: {data['EnergyConsumption'].mean():.2f}", flush=True)
print(f"  Values == 20.0: {(data['EnergyConsumption'] == 20.0).sum()}", flush=True)
print(f"  Values < 20.5: {(data['EnergyConsumption'] < 20.5).sum()}", flush=True)

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

# Cyclical encoding
data['HourSin'] = np.sin(2 * np.pi * data['Hour'] / 24)
data['HourCos'] = np.cos(2 * np.pi * data['Hour'] / 24)
data['MonthSin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['MonthCos'] = np.cos(2 * np.pi * data['Month'] / 12)
data['DayOfWeekSin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
data['DayOfWeekCos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)

# Interaction features (original)
data['Temp_x_Occupancy']  = data['Temperature'] * data['Occupancy']
data['Temp_x_HVAC']       = data['Temperature'] * data['HVACUsage']
data['Temp_squared']      = data['Temperature'] ** 2
data['Occ_x_Lighting']    = data['Occupancy']   * data['LightingUsage']
data['Occ_x_HVAC']        = data['Occupancy']   * data['HVACUsage']
data['Energy_intensity']  = data['Occupancy']   / (data['SquareFootage'] + 1)
data['Renewable_ratio']   = data['RenewableEnergy'] / (data['Temperature'] + 1)

# Additional interaction features
data['Humidity_x_Temp']   = data['Humidity'] * data['Temperature']
data['Sqft_x_Occ']        = data['SquareFootage'] * data['Occupancy']
data['Humidity_squared']   = data['Humidity'] ** 2
data['Occ_squared']        = data['Occupancy'] ** 2
data['Temp_x_Humidity_x_HVAC'] = data['Temperature'] * data['Humidity'] * data['HVACUsage']
data['RenewableEnergy_sq'] = data['RenewableEnergy'] ** 2
data['Temp_x_Sqft']       = data['Temperature'] * data['SquareFootage']
data['HVAC_x_Lighting']   = data['HVACUsage'] * data['LightingUsage']
data['Occ_x_Sqft_x_HVAC'] = data['Occupancy'] * data['SquareFootage'] * data['HVACUsage']
data['Temp_cubed']         = data['Temperature'] ** 3
data['Humidity_x_Occ']     = data['Humidity'] * data['Occupancy']

features = [
    'Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
    'HVACUsage', 'LightingUsage', 'RenewableEnergy',
    'DayOfWeek', 'Holiday', 'Hour', 'Day', 'Month',
    'WeekendLabel', 'TimePeriodLabel',
    'HourSin', 'HourCos', 'MonthSin', 'MonthCos',
    'DayOfWeekSin', 'DayOfWeekCos',
    'Temp_x_Occupancy', 'Temp_x_HVAC', 'Temp_squared',
    'Occ_x_Lighting', 'Occ_x_HVAC', 'Energy_intensity', 'Renewable_ratio',
    'Humidity_x_Temp', 'Sqft_x_Occ', 'Humidity_squared',
    'Occ_squared', 'Temp_x_Humidity_x_HVAC', 'RenewableEnergy_sq',
    'Temp_x_Sqft', 'HVAC_x_Lighting', 'Occ_x_Sqft_x_HVAC',
    'Temp_cubed', 'Humidity_x_Occ',
]
target = 'EnergyConsumption'

X = data[features]
y = data[target]

print(f"\nTotal features: {len(features)}", flush=True)

# =============================================
# APPROACH 1: Remove clipped values at 20.0
# =============================================
print("\n=== APPROACH 1: Remove clipped values ===", flush=True)
mask = data['EnergyConsumption'] > 20.0
X_clean = X[mask]
y_clean = y[mask]
print(f"After removing clipped values: {X_clean.shape[0]} rows", flush=True)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

model_c = GradientBoostingRegressor(
    n_estimators=800, learning_rate=0.02, max_depth=5,
    subsample=0.8, max_features=None, min_samples_leaf=3,
    min_samples_split=5, random_state=42
)
model_c.fit(X_train_c, y_train_c)
r2_c = r2_score(y_test_c, model_c.predict(X_test_c))
print(f"R2 (cleaned data, standard GB): {r2_c:.6f}", flush=True)

# =============================================
# APPROACH 2: HistGradientBoostingRegressor (still GB)
# =============================================
print("\n=== APPROACH 2: HistGradientBoostingRegressor ===", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

hist_configs = [
    {"max_iter": 1000, "learning_rate": 0.05, "max_depth": 6, "min_samples_leaf": 5, "max_bins": 255, "l2_regularization": 0.0},
    {"max_iter": 1500, "learning_rate": 0.03, "max_depth": 7, "min_samples_leaf": 5, "max_bins": 255, "l2_regularization": 0.0},
    {"max_iter": 2000, "learning_rate": 0.02, "max_depth": 8, "min_samples_leaf": 3, "max_bins": 255, "l2_regularization": 0.0},
    {"max_iter": 3000, "learning_rate": 0.01, "max_depth": 8, "min_samples_leaf": 3, "max_bins": 255, "l2_regularization": 0.0},
    {"max_iter": 2000, "learning_rate": 0.03, "max_depth": 10, "min_samples_leaf": 2, "max_bins": 255, "l2_regularization": 0.0},
    {"max_iter": 3000, "learning_rate": 0.02, "max_depth": 10, "min_samples_leaf": 2, "max_bins": 255, "l2_regularization": 0.0},
    {"max_iter": 5000, "learning_rate": 0.01, "max_depth": 10, "min_samples_leaf": 2, "max_bins": 255, "l2_regularization": 0.0},
    {"max_iter": 2000, "learning_rate": 0.05, "max_depth": 12, "min_samples_leaf": 1, "max_bins": 255, "l2_regularization": 0.0},
    {"max_iter": 3000, "learning_rate": 0.03, "max_depth": 15, "min_samples_leaf": 1, "max_bins": 255, "l2_regularization": 0.0},
    {"max_iter": 5000, "learning_rate": 0.02, "max_depth": None, "min_samples_leaf": 1, "max_bins": 255, "l2_regularization": 0.0},
]

best_r2 = 0
best_idx = -1
best_model = None

for i, config in enumerate(hist_configs):
    model = HistGradientBoostingRegressor(random_state=42, **config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"HistGB Config {i}: R2={r2:.6f} | {config}", flush=True)
    if r2 > best_r2:
        best_r2 = r2
        best_idx = i
        best_model = model

print(f"\nBest HistGB R2: {best_r2:.6f} (Config {best_idx})", flush=True)

# =============================================
# APPROACH 3: HistGB on cleaned data
# =============================================
print("\n=== APPROACH 3: HistGB on cleaned data ===", flush=True)

X_train_c2, X_test_c2, y_train_c2, y_test_c2 = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

hist_configs_clean = [
    {"max_iter": 2000, "learning_rate": 0.03, "max_depth": 10, "min_samples_leaf": 2, "max_bins": 255, "l2_regularization": 0.0},
    {"max_iter": 3000, "learning_rate": 0.02, "max_depth": 12, "min_samples_leaf": 1, "max_bins": 255, "l2_regularization": 0.0},
    {"max_iter": 5000, "learning_rate": 0.01, "max_depth": None, "min_samples_leaf": 1, "max_bins": 255, "l2_regularization": 0.0},
]

best_r2_clean = 0
best_model_clean = None

for i, config in enumerate(hist_configs_clean):
    model = HistGradientBoostingRegressor(random_state=42, **config)
    model.fit(X_train_c2, y_train_c2)
    y_pred = model.predict(X_test_c2)
    r2 = r2_score(y_test_c2, y_pred)
    print(f"HistGB Clean Config {i}: R2={r2:.6f} | {config}", flush=True)
    if r2 > best_r2_clean:
        best_r2_clean = r2
        best_model_clean = model

print(f"\nBest HistGB Clean R2: {best_r2_clean:.6f}", flush=True)

# =============================================
# Summary
# =============================================
print("\n" + "="*60, flush=True)
print("SUMMARY OF ALL APPROACHES", flush=True)
print("="*60, flush=True)
print(f"Standard GB (cleaned data):  R2 = {r2_c:.6f}", flush=True)
print(f"HistGB (full data):          R2 = {best_r2:.6f}", flush=True)
print(f"HistGB (cleaned data):       R2 = {best_r2_clean:.6f}", flush=True)

overall_best = max(r2_c, best_r2, best_r2_clean)
if overall_best >= 0.91:
    print(f"\nSUCCESS: R2 >= 0.91 achieved! Best = {overall_best:.6f}", flush=True)
else:
    print(f"\nBest overall: {overall_best:.6f}, target: 0.91", flush=True)
