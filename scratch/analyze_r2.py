import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv("Data/Energy_consumption_data.csv")

# Reconstruct generating variables
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Hour'] = data['Timestamp'].dt.hour
data['Day'] = data['Timestamp'].dt.day
data['Month'] = data['Timestamp'].dt.month
data['DayOfWeek'] = data['Timestamp'].dt.weekday

day_mapping = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
# wait, DayOfWeek in dataset is already string, but we can also use data['DayOfWeek'] mapped or just timestamp.dt.weekday
# let's look at generate_dataset.py:
# dow = np.array([t.weekday() for t in timestamps])
# so timestamp.dt.weekday matches dow perfectly.

# Let's map Holiday, HVACUsage, LightingUsage
holiday_mapping = {"Yes": 1, "No": 0}
hvac_mapping = {"On": 1, "Off": 0}
lighting_mapping = {"On": 1, "Off": 0}

hvac_val = data['HVACUsage'].map(hvac_mapping)
lighting_val = data['LightingUsage'].map(lighting_mapping)

base_load = data['SquareFootage'] / 80
hvac_load = hvac_val * (0.4 + 0.04 * np.abs(data['Temperature'] - 22))
occ_load = data['Occupancy'] * 1.2
lighting_load = lighting_val * 3.5
renewable_off = data['RenewableEnergy'] * 0.15
peak_factor = 1 + 0.3 * ((data['Hour'] >= 17) & (data['Hour'] <= 21)).astype(float)

true_energy_no_noise = (base_load + hvac_load + occ_load + lighting_load - renewable_off) * peak_factor
true_energy_clipped = np.clip(true_energy_no_noise, 20, 200)

y = data['EnergyConsumption']

# Let's calculate R2 score of the true function
r2_true = r2_score(y, true_energy_clipped)
print(f"Theoretical maximum R2 (with true formula): {r2_true:.6f}")
print(f"Variance of target y: {y.var():.6f}")
print(f"MSE of true formula: {((y - true_energy_clipped)**2).mean():.6f}")
