import os
import pandas as pd
data = pd.read_csv("Data/Energy_consumption_data.csv")
print(data.info())
print(data.head())
print(data.describe())
