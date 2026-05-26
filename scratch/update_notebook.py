"""
Script to update model.ipynb with improved model (R^2 >= 0.80).
Key improvements:
  1. Add RenewableEnergy to features
  2. Add interaction & cyclical features
  3. Switch to GradientBoostingRegressor
  4. Retrain and re-save energy_model.pkl
"""

import json
import os

NOTEBOOK_PATH = r"c:\Users\SRI KRUTHIKA REDDY\Downloads\Energy Forecasting System\Energy Forecasting System\Model\model.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ─────────────────────────────────────────────────────────────
# NEW CELL SOURCES
# ─────────────────────────────────────────────────────────────

IMPORT_SOURCE = [
    "\n",
    "# =========================================================\n",
    "# 1. IMPORT LIBRARIES\n",
    "# =========================================================\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import (\n",
    "    train_test_split,\n",
    "    cross_val_score,\n",
    "    RandomizedSearchCV\n",
    ")\n",
    "\n",
    "from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor\n",
    "\n",
    "from sklearn.metrics import (\n",
    "    r2_score,\n",
    "    mean_squared_error,\n",
    "    mean_absolute_error\n",
    ")\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
]

FEATURES_SOURCE = [
    "# =========================================================\n",
    "# 7. ENHANCED FEATURE ENGINEERING\n",
    "# =========================================================\n",
    "\n",
    "# --- Cyclical encoding of Hour ---\n",
    "data['HourSin'] = np.sin(2 * np.pi * data['Hour'] / 24)\n",
    "data['HourCos'] = np.cos(2 * np.pi * data['Hour'] / 24)\n",
    "\n",
    "# --- Cyclical encoding of Month ---\n",
    "data['MonthSin'] = np.sin(2 * np.pi * data['Month'] / 12)\n",
    "data['MonthCos'] = np.cos(2 * np.pi * data['Month'] / 12)\n",
    "\n",
    "# --- Cyclical encoding of DayOfWeek ---\n",
    "data['DayOfWeekSin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)\n",
    "data['DayOfWeekCos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)\n",
    "\n",
    "# --- Interaction features ---\n",
    "data['Temp_x_Occupancy']  = data['Temperature'] * data['Occupancy']\n",
    "data['Temp_x_HVAC']       = data['Temperature'] * data['HVACUsage']\n",
    "data['Temp_squared']      = data['Temperature'] ** 2\n",
    "data['Occ_x_Lighting']    = data['Occupancy']   * data['LightingUsage']\n",
    "data['Occ_x_HVAC']        = data['Occupancy']   * data['HVACUsage']\n",
    "data['Energy_intensity']  = data['Occupancy']   / (data['SquareFootage'] + 1)\n",
    "data['Renewable_ratio']   = data['RenewableEnergy'] / (data['Temperature'] + 1)\n",
    "\n",
    "# =========================================================\n",
    "# 8. FEATURES & TARGET\n",
    "# =========================================================\n",
    "\n",
    "features = [\n",
    "    # Core features\n",
    "    'Temperature',\n",
    "    'Humidity',\n",
    "    'SquareFootage',\n",
    "    'Occupancy',\n",
    "    'HVACUsage',\n",
    "    'LightingUsage',\n",
    "    'RenewableEnergy',\n",
    "    'DayOfWeek',\n",
    "    'Holiday',\n",
    "    'Hour',\n",
    "    'Day',\n",
    "    'Month',\n",
    "    'WeekendLabel',\n",
    "    'TimePeriodLabel',\n",
    "    # Cyclical encodings\n",
    "    'HourSin', 'HourCos',\n",
    "    'MonthSin', 'MonthCos',\n",
    "    'DayOfWeekSin', 'DayOfWeekCos',\n",
    "    # Interaction features\n",
    "    'Temp_x_Occupancy',\n",
    "    'Temp_x_HVAC',\n",
    "    'Temp_squared',\n",
    "    'Occ_x_Lighting',\n",
    "    'Occ_x_HVAC',\n",
    "    'Energy_intensity',\n",
    "    'Renewable_ratio',\n",
    "]\n",
    "\n",
    "target = 'EnergyConsumption'\n",
    "\n",
    "X = data[features]\n",
    "y = data[target]\n",
    "\n",
    "print(f\"Total features used: {len(features)}\")\n",
    "print(f\"Dataset shape: {X.shape}\")\n"
]

MODEL_TRAIN_SOURCE = [
    "# =========================================================\n",
    "# 9. TRAIN / TEST SPLIT\n",
    "# =========================================================\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X,\n",
    "    y,\n",
    "    test_size=0.2,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "\n",
    "# =========================================================\n",
    "# 10. TRAIN GRADIENT BOOSTING MODEL\n",
    "# =========================================================\n",
    "\n",
    "model = GradientBoostingRegressor(\n",
    "    n_estimators=500,\n",
    "    learning_rate=0.05,\n",
    "    max_depth=5,\n",
    "    min_samples_leaf=3,\n",
    "    min_samples_split=5,\n",
    "    subsample=0.85,\n",
    "    max_features='sqrt',\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "print(\"Model training complete!\")\n"
]

EVAL_SOURCE = [
    "# =========================================================\n",
    "# 11. PREDICTIONS & EVALUATION\n",
    "# =========================================================\n",
    "\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "r2   = r2_score(y_test, y_pred)\n",
    "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "mae  = mean_absolute_error(y_test, y_pred)\n",
    "\n",
    "print(\"\\n================ MODEL PERFORMANCE ================\")\n",
    "print(f\"R2 Score : {r2:.4f}\")\n",
    "print(f\"RMSE     : {rmse:.2f}\")\n",
    "print(f\"MAE      : {mae:.2f}\")\n",
    "\n",
    "# Cross-validation score\n",
    "cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')\n",
    "print(f\"\\n5-Fold CV R2 : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\")\n",
    "\n",
    "if r2 >= 0.80:\n",
    "    print(f\"\\nSUCCESS: Target achieved! R2 = {r2:.4f} >= 0.80\")\n",
    "else:\n",
    "    print(f\"\\nWARNING: R2 = {r2:.4f} - still below 0.80\")\n"
]

SAVE_MODEL_SOURCE = [
    "#save the model\n",
    "import joblib\n",
    "joblib.dump(model, 'energy_model.pkl')\n",
    "print(\"Model saved as energy_model.pkl\")\n"
]

FEATURE_IMP_SOURCE = [
    "# =========================================================\n",
    "# 13. FEATURE IMPORTANCE\n",
    "# =========================================================\n",
    "\n",
    "importance_df = pd.DataFrame({\n",
    "    'Feature': features,\n",
    "    'Importance': model.feature_importances_\n",
    "})\n",
    "\n",
    "importance_df = importance_df.sort_values(\n",
    "    by='Importance',\n",
    "    ascending=False\n",
    ").reset_index(drop=True)\n",
    "\n",
    "print(\"\\n================ FEATURE IMPORTANCE ================\")\n",
    "print(importance_df.to_string())\n"
]

# ─────────────────────────────────────────────────────────────
# REBUILD NOTEBOOK CELLS
# ─────────────────────────────────────────────────────────────

def make_cell(cell_id, source, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": outputs if outputs is not None else [],
        "source": source
    }

# Collect all original cells by ID for reference
orig_by_id = {c["id"]: c for c in nb["cells"]}

new_cells = []

# 1. Imports (replace cell f8c255dd)
new_cells.append(make_cell("f8c255dd", IMPORT_SOURCE))

# 2. Load Dataset (keep original)
new_cells.append(orig_by_id["3411d952"])

# 3. Feature engineering functions (keep cell 57d061a9)
new_cells.append(orig_by_id["57d061a9"])

# 4. Data Preprocessing (keep cell a9af63bc)
new_cells.append(orig_by_id["a9af63bc"])

# 5. Data display (keep cell bb95bd11)
new_cells.append(orig_by_id["bb95bd11"])

# 6. NEW: Enhanced features + feature list (replaces old cell 44982297)
new_cells.append(make_cell("44982297_v2", FEATURES_SOURCE))

# 7. Model training (replaces cell 777186f2)
new_cells.append(make_cell("777186f2_v2", MODEL_TRAIN_SOURCE))

# 8. Save model (keep cell 320e737f)
new_cells.append(make_cell("320e737f_v2", SAVE_MODEL_SOURCE))

# 9. Evaluation (replaces cell b3e0790e)
new_cells.append(make_cell("b3e0790e_v2", EVAL_SOURCE))

# 10. Feature importance (replaces cell 3ec32dcf)
new_cells.append(make_cell("3ec32dcf_v2", FEATURE_IMP_SOURCE))

# 11. User input / prediction pipeline (keep original cells)
for cid in ["24aad43b", "6f822662", "570c041d", "2240bf5a", "2ff611df", "f28df166", "aff09fd1", "e87b8457"]:
    if cid in orig_by_id:
        new_cells.append(orig_by_id[cid])

nb["cells"] = new_cells

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("SUCCESS: Notebook updated successfully!")
print("Path: " + NOTEBOOK_PATH)
print("Next: run the training script to evaluate this notebook and save the model.")
