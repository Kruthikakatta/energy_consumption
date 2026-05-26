"""
Programmatically train the model using the same parameters as the updated notebook,
verify its performance (R^2 >= 0.80), and save it as Model/energy_model.pkl.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the newly generated dataset
DATA_PATH = r"c:\Users\SRI KRUTHIKA REDDY\Downloads\Energy Forecasting System\Energy Forecasting System\Data\Energy_consumption_data.csv"
data = pd.read_csv(DATA_PATH)

# Preprocessing & encoding identical to notebook
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

# Helper functions
def get_time_period(hour):
    if 5 <= hour < 12:
        return 0  # Morning
    elif 12 <= hour < 17:
        return 1  # Afternoon
    elif 17 <= hour < 21:
        return 2  # Evening
    else:
        return 3  # Night

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

# Interaction features
data['Temp_x_Occupancy']  = data['Temperature'] * data['Occupancy']
data['Temp_x_HVAC']       = data['Temperature'] * data['HVACUsage']
data['Temp_squared']      = data['Temperature'] ** 2
data['Occ_x_Lighting']    = data['Occupancy']   * data['LightingUsage']
data['Occ_x_HVAC']        = data['Occupancy']   * data['HVACUsage']
data['Energy_intensity']  = data['Occupancy']   / (data['SquareFootage'] + 1)
data['Renewable_ratio']   = data['RenewableEnergy'] / (data['Temperature'] + 1)

# Feature selection
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

X = data[features]
y = data['EnergyConsumption']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=3,
    min_samples_split=5,
    subsample=0.85,
    max_features='sqrt',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print("================ TRAINED MODEL PERFORMANCE ================")
print(f"R2 Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")

# Cross validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"5-Fold CV R2 : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# Save the trained model
save_path = r"c:\Users\SRI KRUTHIKA REDDY\Downloads\Energy Forecasting System\Energy Forecasting System\Model\energy_model.pkl"
joblib.dump(model, save_path)
print("SUCCESS: Model saved to " + save_path)
