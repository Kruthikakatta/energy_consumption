"""
FastAPI Backend - Energy Consumption Forecasting System
Production-ready API with user-friendly inputs and comprehensive predictions.
FIXED: Daily forecast endpoint now works
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Literal
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

# =========================================================
# FASTAPI APP INITIALIZATION
# =========================================================

app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="Predict energy consumption for homes and commercial buildings",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# GLOBAL VARIABLES & MODEL LOADING
# =========================================================

MODEL_PATH = Path(__file__).parent.parent / "Model" / "energy_model.pkl"
FEATURES = [
    'Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
    'HVACUsage', 'LightingUsage', 'RenewableEnergy',
    'DayOfWeek', 'Holiday', 'Hour', 'Day', 'Month',
    'WeekendLabel', 'TimePeriodLabel',
    'HourSin', 'HourCos', 'MonthSin', 'MonthCos',
    'DayOfWeekSin', 'DayOfWeekCos',
    'Temp_x_Occupancy', 'Temp_x_HVAC', 'Temp_squared',
    'Occ_x_Lighting', 'Occ_x_HVAC', 'Energy_intensity', 'Renewable_ratio'
]

# Pricing constants (INR per kWh)
DOMESTIC_RATE = 6.5
COMMERCIAL_RATE = 9.0

MODEL = None

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_time_period(hour: int) -> int:
    """Convert hour to time period label."""
    if 5 <= hour < 12:
        return 0  # Morning
    elif 12 <= hour < 17:
        return 1  # Afternoon
    elif 17 <= hour < 21:
        return 2  # Evening
    else:
        return 3  # Night


def get_weekend_label(dayofweek: int) -> int:
    """Convert day of week to weekend label."""
    return 1 if dayofweek >= 5 else 0


# =========================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =========================================================

class UserFriendlyInput(BaseModel):
    """User-friendly input schema with intuitive fields."""
    property_type: Literal["Home", "Commercial"] = Field(
        ...,
        description="Type of property: Home or Commercial"
    )
    property_size: Literal["Small", "Medium", "Large"] = Field(
        ...,
        description="Size category: Small (300 sq ft), Medium (750 sq ft), Large (1000 sq ft)"
    )
    people_count: int = Field(
        ...,
        ge=1,
        le=100,
        description="Number of people in the building"
    )
    ac_usage: Literal["Low", "Medium", "High"] = Field(
        ...,
        description="Air conditioning usage level"
    )
    lighting_usage: Literal["Low", "Medium", "High"] = Field(
        ...,
        description="Lighting usage level"
    )
    weather_condition: Literal["Cool", "Normal", "Hot"] = Field(
        ...,
        description="Current weather: Cool (22C), Normal (28C), Hot (35C)"
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day (0-23)"
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day of week (0=Monday, 6=Sunday)"
    )
    holiday: Literal[0, 1] = Field(
        ...,
        description="Is it a holiday? 0=No, 1=Yes"
    )
    day: int = Field(
        ...,
        ge=1,
        le=31,
        description="Day of month (1-31)"
    )
    month: int = Field(
        ...,
        ge=1,
        le=12,
        description="Month (1-12)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "property_type": "Home",
                "property_size": "Medium",
                "people_count": 4,
                "ac_usage": "Medium",
                "lighting_usage": "Low",
                "weather_condition": "Hot",
                "hour": 14,
                "day_of_week": 2,
                "holiday": 0,
                "day": 23,
                "month": 5
            }
        }


class TechnicalInput(BaseModel):
    """Technical input schema for advanced users with exact parameters."""
    temperature: float = Field(..., ge=-10, le=50, description="Temperature in C")
    humidity: float = Field(..., ge=0, le=100, description="Humidity in %")
    square_footage: int = Field(..., ge=100, le=10000, description="Building size in sq ft")
    occupancy: int = Field(..., ge=1, le=100, description="Number of occupants")
    hvac_usage: Literal[0, 1] = Field(..., description="HVAC status (0=Off, 1=On)")
    lighting_usage: Literal[0, 1] = Field(..., description="Lighting status (0=Off, 1=On)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    holiday: Literal[0, 1] = Field(..., description="Holiday flag (0=No, 1=Yes)")
    hour: int = Field(..., ge=0, le=23, description="Hour (0-23)")
    day: int = Field(..., ge=1, le=31, description="Day of month")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    
    class Config:
        schema_extra = {
            "example": {
                "temperature": 28.5,
                "humidity": 65,
                "square_footage": 750,
                "occupancy": 4,
                "hvac_usage": 1,
                "lighting_usage": 1,
                "day_of_week": 2,
                "holiday": 0,
                "hour": 14,
                "day": 23,
                "month": 5
            }
        }


class PredictionResponse(BaseModel):
    """Comprehensive prediction response."""
    # Core Predictions - NOW UNCOMMENTED (FIXED)
    hourly_consumption_kwh: float = Field(..., description="Predicted hourly energy consumption")
    daily_consumption_kwh: float = Field(..., description="Estimated daily consumption")
    monthly_consumption_kwh: float = Field(..., description="Estimated monthly consumption")
    
    # Cost Estimates - NOW UNCOMMENTED (FIXED)
    hourly_cost_inr: float = Field(..., description="Hourly cost")
    daily_cost_inr: float = Field(..., description="Daily cost estimate")
    monthly_cost_inr: float = Field(..., description="Monthly cost estimate")
    
    # Metadata
    property_type: str = Field(..., description="Home or Commercial")
    electricity_rate_inr: float = Field(..., description="Rate per kWh")
    time_period: str = Field(..., description="Morning/Afternoon/Evening/Night")
    is_weekend: bool = Field(..., description="Weekend flag")
    
    # Recommendations
    recommendation: str = Field(..., description="Energy-saving recommendation")
    usage_category: str = Field(..., description="Low/Moderate/High usage category")


class DailyForecastRequest(BaseModel):
    """Request for 24-hour forecast."""
    property_type: Literal["Home", "Commercial"]
    property_size: Literal["Small", "Medium", "Large"]
    people_count: int = Field(..., ge=1, le=100)
    ac_usage: Literal["Low", "Medium", "High"]
    lighting_usage: Literal["Low", "Medium", "High"]
    weather_condition: Literal["Cool", "Normal", "Hot"]
    day_of_week: int = Field(..., ge=0, le=6)
    holiday: Literal[0, 1]
    day: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12)


# =========================================================
# STARTUP EVENT - LOAD MODEL
# =========================================================

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    global MODEL
    try:
        MODEL = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Could not load model from {MODEL_PATH}")
        print(f"Error: {str(e)}")
        print("API will return mock predictions until model is loaded.")


# =========================================================
# PREDICTION UTILITIES
# =========================================================

def map_user_friendly_to_technical(user_input: UserFriendlyInput) -> Dict:
    """Convert user-friendly inputs to technical model inputs."""
    # Property Size Mapping
    size_mapping = {
        "Small": 300,
        "Medium": 750,
        "Large": 1000
    }
    square_footage = size_mapping[user_input.property_size]
    
    # Weather Mapping
    weather_mapping = {
        "Cool": {"Temperature": 22, "Humidity": 45},
        "Normal": {"Temperature": 28, "Humidity": 55},
        "Hot": {"Temperature": 35, "Humidity": 65}
    }
    temperature = weather_mapping[user_input.weather_condition]["Temperature"]
    humidity = weather_mapping[user_input.weather_condition]["Humidity"]
    
    # AC Usage Mapping
    ac_mapping = {"Low": 0, "Medium": 1, "High": 1}
    hvac_usage = ac_mapping[user_input.ac_usage]
    
    # Lighting Usage Mapping
    lighting_mapping = {"Low": 0, "Medium": 1, "High": 1}
    lighting_usage = lighting_mapping[user_input.lighting_usage]
    
    return {
        "temperature": temperature,
        "humidity": humidity,
        "square_footage": square_footage,
        "occupancy": user_input.people_count,
        "hvac_usage": hvac_usage,
        "lighting_usage": lighting_usage,
        "day_of_week": user_input.day_of_week,
        "holiday": user_input.holiday,
        "hour": user_input.hour,
        "day": user_input.day,
        "month": user_input.month,
        "property_type": user_input.property_type
    }


def estimate_renewable_energy(hour: int) -> float:
    """Estimate solar energy generation based on hour of day (peaking at noon)."""
    if 6 <= hour <= 18:
        return float(max(0.0, 15.0 * np.sin(np.pi * (hour - 6) / 12)))
    return 0.0


def prepare_model_input(technical_params: Dict) -> pd.DataFrame:
    """Prepare feature vector for model prediction."""
    # 1. Core variables
    temp = technical_params["temperature"]
    humidity = technical_params["humidity"]
    sqft = technical_params["square_footage"]
    occ = technical_params["occupancy"]
    hvac = technical_params["hvac_usage"]
    lighting = technical_params["lighting_usage"]
    dow = technical_params["day_of_week"]
    holiday = technical_params["holiday"]
    hour = technical_params["hour"]
    day = technical_params["day"]
    month = technical_params["month"]
    
    weekend_label = get_weekend_label(dow)
    time_period_label = get_time_period(hour)
    
    # Estimate renewable energy based on the hour
    renewable = estimate_renewable_energy(hour)
    
    # 2. Cyclical encodings
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    
    # 3. Interaction features
    temp_x_occ = temp * occ
    temp_x_hvac = temp * hvac
    temp_squared = temp ** 2
    occ_x_light = occ * lighting
    occ_x_hvac = occ * hvac
    energy_intensity = occ / (sqft + 1)
    renewable_ratio = renewable / (temp + 1)
    
    features_dict = {
        'Temperature': temp,
        'Humidity': humidity,
        'SquareFootage': sqft,
        'Occupancy': occ,
        'HVACUsage': hvac,
        'LightingUsage': lighting,
        'RenewableEnergy': renewable,
        'DayOfWeek': dow,
        'Holiday': holiday,
        'Hour': hour,
        'Day': day,
        'Month': month,
        'WeekendLabel': weekend_label,
        'TimePeriodLabel': time_period_label,
        'HourSin': hour_sin,
        'HourCos': hour_cos,
        'MonthSin': month_sin,
        'MonthCos': month_cos,
        'DayOfWeekSin': dow_sin,
        'DayOfWeekCos': dow_cos,
        'Temp_x_Occupancy': temp_x_occ,
        'Temp_x_HVAC': temp_x_hvac,
        'Temp_squared': temp_squared,
        'Occ_x_Lighting': occ_x_light,
        'Occ_x_HVAC': occ_x_hvac,
        'Energy_intensity': energy_intensity,
        'Renewable_ratio': renewable_ratio
    }
    
    return pd.DataFrame([features_dict])


def generate_recommendation(monthly_consumption: float) -> tuple:
    """Generate energy-saving recommendation based on consumption."""
    if monthly_consumption > 1200:
        return (
            "Very high energy usage detected. Consider: "
            "1) Reduce AC usage during peak hours, "
            "2) Switch to LED lighting, "
            "3) Regular appliance maintenance.",
            "High"
        )
    elif monthly_consumption > 600:
        return (
            "Moderate energy usage. Tips: "
            "1) Monitor AC and lighting schedules, "
            "2) Use natural ventilation when possible, "
            "3) Turn off unused devices.",
            "Moderate"
        )
    else:
        return (
            "Excellent! Your energy usage is efficient. "
            "Keep maintaining good practices.",
            "Low"
        )


def make_prediction(technical_params: Dict) -> PredictionResponse:
    """Make energy consumption prediction and calculate costs."""
    # Prepare input
    model_input = prepare_model_input(technical_params)
    
    # Make prediction
    if MODEL is None:
        # Mock prediction if model not loaded
        hourly_consumption = 2.5  # Realistic value for a home
    else:
        raw_prediction = float(MODEL.predict(model_input)[0])
        # Use raw prediction directly (already scaled and realistic in new dataset)
        hourly_consumption = max(1.0, raw_prediction)
    
    # Calculate daily and monthly estimates
    daily_consumption = hourly_consumption * 24
    monthly_consumption = daily_consumption * 30
    
    # Determine electricity rate
    property_type = technical_params.get("property_type", "Home")
    electricity_rate = DOMESTIC_RATE if property_type == "Home" else COMMERCIAL_RATE
    
    # Calculate costs
    hourly_cost = hourly_consumption * electricity_rate
    daily_cost = daily_consumption * electricity_rate
    monthly_cost = monthly_consumption * electricity_rate
    
    # Generate recommendation
    recommendation, usage_category = generate_recommendation(monthly_consumption)
    
    # Get time period name
    time_periods = ["Morning", "Afternoon", "Evening", "Night"]
    time_period = time_periods[get_time_period(technical_params["hour"])]
    is_weekend = bool(get_weekend_label(technical_params["day_of_week"]))
    
    # Return response - NOW INCLUDES hourly fields (FIXED)
    return PredictionResponse(
        hourly_consumption_kwh=round(hourly_consumption, 2),
        daily_consumption_kwh=round(daily_consumption, 2),
        monthly_consumption_kwh=round(monthly_consumption, 2),
        hourly_cost_inr=round(hourly_cost, 2),
        daily_cost_inr=round(daily_cost, 2),
        monthly_cost_inr=round(monthly_cost, 2),
        property_type=property_type,
        electricity_rate_inr=electricity_rate,
        time_period=time_period,
        is_weekend=is_weekend,
        recommendation=recommendation,
        usage_category=usage_category
    )


# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - Beautiful API Dashboard."""
    model_status = "✅ Loaded" if MODEL is not None else "⚠️ Mock Mode"
    model_badge = "badge-success" if MODEL is not None else "badge-warning"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>⚡ Energy Forecasting API</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg: #050a14;
      --surface: #0d1627;
      --surface2: #111e35;
      --border: #1e3a5f;
      --accent: #00d4ff;
      --accent2: #7c3aed;
      --accent3: #10b981;
      --warn: #f59e0b;
      --text: #e2e8f0;
      --muted: #64748b;
      --glow: 0 0 30px rgba(0, 212, 255, 0.15);
    }}
    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
    }}

    /* Animated background */
    body::before {{
      content: '';
      position: fixed;
      inset: 0;
      background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,212,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(124,58,237,0.06) 0%, transparent 60%);
      pointer-events: none;
      z-index: 0;
    }}

    .container {{ position: relative; z-index: 1; max-width: 1100px; margin: 0 auto; padding: 40px 24px; }}

    /* HEADER */
    .header {{
      text-align: center;
      padding: 60px 0 40px;
    }}
    .logo {{
      display: inline-flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 20px;
    }}
    .logo-icon {{
      width: 64px; height: 64px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      border-radius: 18px;
      display: flex; align-items: center; justify-content: center;
      font-size: 30px;
      box-shadow: 0 0 40px rgba(0,212,255,0.3);
      animation: pulse 3s ease-in-out infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ box-shadow: 0 0 40px rgba(0,212,255,0.3); }}
      50% {{ box-shadow: 0 0 60px rgba(0,212,255,0.5), 0 0 80px rgba(124,58,237,0.2); }}
    }}
    h1 {{
      font-size: 2.6rem;
      font-weight: 800;
      background: linear-gradient(135deg, #fff 0%, var(--accent) 60%, var(--accent2) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1.2;
    }}
    .tagline {{
      color: var(--muted);
      font-size: 1.05rem;
      margin-top: 10px;
      font-weight: 400;
    }}

    /* STATUS BAR */
    .status-bar {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 20px;
      flex-wrap: wrap;
      margin: 30px 0;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 18px;
      border-radius: 100px;
      font-size: 0.85rem;
      font-weight: 600;
      border: 1px solid transparent;
    }}
    .badge-success {{ background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.3); color: #34d399; }}
    .badge-warning {{ background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.3); color: #fbbf24; }}
    .badge-info    {{ background: rgba(0,212,255,0.10); border-color: rgba(0,212,255,0.25); color: var(--accent); }}
    .badge-purple  {{ background: rgba(124,58,237,0.12); border-color: rgba(124,58,237,0.3); color: #a78bfa; }}
    .dot {{ width: 8px; height: 8px; border-radius: 50%; background: currentColor; animation: blink 1.5s infinite; }}
    @keyframes blink {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:0.3 }} }}

    /* CARDS GRID */
    .grid {{ display: grid; gap: 20px; margin: 36px 0; }}
    .grid-3 {{ grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
    .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); }}

    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 28px;
      transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    }}
    .card:hover {{
      transform: translateY(-3px);
      border-color: rgba(0,212,255,0.35);
      box-shadow: var(--glow);
    }}
    .card-header {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 20px;
    }}
    .card-icon {{
      width: 42px; height: 42px;
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 20px;
      flex-shrink: 0;
    }}
    .icon-blue   {{ background: rgba(0,212,255,0.12); }}
    .icon-purple {{ background: rgba(124,58,237,0.12); }}
    .icon-green  {{ background: rgba(16,185,129,0.12); }}
    .icon-orange {{ background: rgba(245,158,11,0.12); }}
    .card-title {{ font-size: 1rem; font-weight: 600; color: var(--text); }}
    .card-sub   {{ font-size: 0.78rem; color: var(--muted); margin-top: 2px; }}

    /* ENDPOINTS LIST */
    .endpoint-list {{ display: flex; flex-direction: column; gap: 10px; }}
    .endpoint {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px;
      text-decoration: none;
      transition: border-color 0.2s, background 0.2s;
      cursor: pointer;
    }}
    .endpoint:hover {{
      border-color: rgba(0,212,255,0.4);
      background: rgba(0,212,255,0.04);
    }}
    .method {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.7rem;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 6px;
      min-width: 44px;
      text-align: center;
    }}
    .method-get  {{ background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.25); }}
    .method-post {{ background: rgba(0,212,255,0.12); color: var(--accent); border: 1px solid rgba(0,212,255,0.2); }}
    .path {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.82rem;
      color: var(--text);
      flex: 1;
    }}
    .ep-desc {{ font-size: 0.75rem; color: var(--muted); text-align: right; }}

    /* STATS */
    .stat {{ text-align: center; padding: 16px; }}
    .stat-value {{
      font-size: 2rem;
      font-weight: 800;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    .stat-label {{ font-size: 0.8rem; color: var(--muted); margin-top: 4px; }}

    /* FEATURES */
    .feature-list {{ display: flex; flex-direction: column; gap: 8px; }}
    .feature {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.88rem;
      color: var(--muted);
    }}
    .feature span {{ color: var(--accent3); font-size: 1rem; }}

    /* FOOTER */
    .footer {{
      text-align: center;
      padding: 40px 0 20px;
      color: var(--muted);
      font-size: 0.82rem;
      border-top: 1px solid var(--border);
      margin-top: 40px;
    }}
    .footer a {{ color: var(--accent); text-decoration: none; }}
    .footer a:hover {{ text-decoration: underline; }}

    /* Animated energy bar */
    .energy-bar-wrap {{ margin-top: 16px; }}
    .energy-bar-label {{ display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--muted); margin-bottom: 6px; }}
    .energy-bar {{
      height: 6px;
      border-radius: 10px;
      background: var(--surface2);
      overflow: hidden;
    }}
    .energy-bar-fill {{
      height: 100%;
      border-radius: 10px;
      animation: fillBar 2s ease-out forwards;
    }}
    @keyframes fillBar {{ from {{ width: 0 }} }}
    .fill-cyan   {{ background: linear-gradient(90deg, var(--accent), #0099bb); width: 92%; }}
    .fill-purple {{ background: linear-gradient(90deg, var(--accent2), #9f1bfa); width: 85%; }}
    .fill-green  {{ background: linear-gradient(90deg, var(--accent3), #059669); width: 78%; }}
  </style>
</head>
<body>
<div class="container">

  <!-- HEADER -->
  <div class="header">
    <div class="logo">
      <div class="logo-icon">⚡</div>
    </div>
    <h1>Energy Forecasting API</h1>
    <p class="tagline">AI-powered energy consumption prediction for homes &amp; commercial buildings</p>
    <div class="status-bar">
      <span class="badge badge-success"><span class="dot"></span> API Online</span>
      <span class="badge {model_badge}">{model_status}</span>
      <span class="badge badge-info">v1.0.0</span>
      <span class="badge badge-purple">🕒 {now}</span>
    </div>
  </div>

  <!-- STATS ROW -->
  <div class="grid grid-3">
    <div class="card">
      <div class="stat">
        <div class="stat-value">6</div>
        <div class="stat-label">API Endpoints</div>
      </div>
    </div>
    <div class="card">
      <div class="stat">
        <div class="stat-value">27</div>
        <div class="stat-label">ML Features</div>
      </div>
    </div>
    <div class="card">
      <div class="stat">
        <div class="stat-value">24h</div>
        <div class="stat-label">Forecast Window</div>
      </div>
    </div>
  </div>

  <!-- ENDPOINTS + MODEL INFO -->
  <div class="grid grid-2">

    <!-- ENDPOINTS -->
    <div class="card">
      <div class="card-header">
        <div class="card-icon icon-blue">🔌</div>
        <div>
          <div class="card-title">API Endpoints</div>
          <div class="card-sub">All available routes</div>
        </div>
      </div>
      <div class="endpoint-list">
        <a class="endpoint" href="/docs" target="_blank">
          <span class="method method-get">GET</span>
          <span class="path">/docs</span>
          <span class="ep-desc">Interactive Docs</span>
        </a>
        <div class="endpoint">
          <span class="method method-get">GET</span>
          <span class="path">/health</span>
          <span class="ep-desc">Health Check</span>
        </div>
        <div class="endpoint">
          <span class="method method-post">POST</span>
          <span class="path">/predict/simple</span>
          <span class="ep-desc">Easy Prediction</span>
        </div>
        <div class="endpoint">
          <span class="method method-post">POST</span>
          <span class="path">/predict/technical</span>
          <span class="ep-desc">Advanced Inputs</span>
        </div>
        <div class="endpoint">
          <span class="method method-post">POST</span>
          <span class="path">/predict/daily</span>
          <span class="ep-desc">24-Hour Forecast</span>
        </div>
        <div class="endpoint">
          <span class="method method-post">POST</span>
          <span class="path">/compare</span>
          <span class="ep-desc">Scenario Compare</span>
        </div>
        <div class="endpoint">
          <span class="method method-get">GET</span>
          <span class="path">/model/info</span>
          <span class="ep-desc">Model Metadata</span>
        </div>
      </div>
    </div>

    <!-- MODEL INFO -->
    <div class="card">
      <div class="card-header">
        <div class="card-icon icon-purple">🤖</div>
        <div>
          <div class="card-title">Model Information</div>
          <div class="card-sub">ML model details &amp; accuracy metrics</div>
        </div>
      </div>
      <div class="feature-list">
        <div class="feature"><span>▸</span> Algorithm: Random Forest Regressor</div>
        <div class="feature"><span>▸</span> Features: 27 engineered inputs</div>
        <div class="feature"><span>▸</span> Cyclical encodings (hour, month, day)</div>
        <div class="feature"><span>▸</span> Interaction &amp; polynomial features</div>
        <div class="feature"><span>▸</span> Domestic rate: ₹6.50 / kWh</div>
        <div class="feature"><span>▸</span> Commercial rate: ₹9.00 / kWh</div>
      </div>
      <div class="energy-bar-wrap">
        <div class="energy-bar-label"><span>Accuracy (R²)</span><span>~92%</span></div>
        <div class="energy-bar"><div class="energy-bar-fill fill-cyan"></div></div>
        <div class="energy-bar-label" style="margin-top:10px"><span>Coverage</span><span>Homes &amp; Commercial</span></div>
        <div class="energy-bar"><div class="energy-bar-fill fill-purple"></div></div>
        <div class="energy-bar-label" style="margin-top:10px"><span>Forecast Horizon</span><span>24 hours</span></div>
        <div class="energy-bar"><div class="energy-bar-fill fill-green"></div></div>
      </div>
    </div>

  </div>

  <!-- QUICK START -->
  <div class="card">
    <div class="card-header">
      <div class="card-icon icon-green">🚀</div>
      <div>
        <div class="card-title">Quick Start</div>
        <div class="card-sub">Try a prediction right now</div>
      </div>
    </div>
    <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:20px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#a78bfa;margin-bottom:8px;">POST /predict/simple</div>
      <pre style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#94a3b8;line-height:1.8;overflow-x:auto;">{{
  "property_type": "Home",
  "property_size": "Medium",
  "people_count": 4,
  "ac_usage": "Medium",
  "lighting_usage": "Low",
  "weather_condition": "Hot",
  "hour": 14,
  "day_of_week": 2,
  "holiday": 0,
  "day": 26,
  "month": 5
}}</pre>
    </div>
    <p style="margin-top:16px;font-size:0.85rem;color:var(--muted);">👉 Visit <a href="/docs" style="color:var(--accent);text-decoration:none;font-weight:600;">/docs</a> for the full interactive Swagger UI, or <a href="/redoc" style="color:var(--accent2);text-decoration:none;font-weight:600;">/redoc</a> for ReDoc documentation.</p>
  </div>

  <div class="footer">
    <p>⚡ <strong>Energy Consumption Forecasting System</strong> &nbsp;|&nbsp; FastAPI v1.0.0 &nbsp;|&nbsp; Built for precision energy prediction</p>
    <p style="margin-top:8px;">Swagger UI: <a href="/docs">/docs</a> &nbsp;•&nbsp; ReDoc: <a href="/redoc">/redoc</a> &nbsp;•&nbsp; Health: <a href="/health">/health</a></p>
  </div>

</div>
</body>
</html>
    """
    return HTMLResponse(content=html)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": MODEL is not None
    }


@app.post("/predict/simple", response_model=PredictionResponse)
async def predict_simple(input_data: UserFriendlyInput):
    """Make prediction using user-friendly inputs."""
    try:
        technical_params = map_user_friendly_to_technical(input_data)
        result = make_prediction(technical_params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/technical", response_model=PredictionResponse)
async def predict_technical(input_data: TechnicalInput):
    """Make prediction using technical/exact parameters."""
    try:
        technical_params = input_data.dict()
        technical_params["property_type"] = "Home"
        result = make_prediction(technical_params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/daily")
async def predict_daily_forecast(request: DailyForecastRequest):
    """Generate 24-hour energy consumption forecast."""
    try:
        hourly_predictions = []
        
        for hour in range(24):
            # Create input for this hour
            user_input = UserFriendlyInput(
                property_type=request.property_type,
                property_size=request.property_size,
                people_count=request.people_count,
                ac_usage=request.ac_usage,
                lighting_usage=request.lighting_usage,
                weather_condition=request.weather_condition,
                hour=hour,
                day_of_week=request.day_of_week,
                holiday=request.holiday,
                day=request.day,
                month=request.month
            )
            
            # Convert and predict
            technical_params = map_user_friendly_to_technical(user_input)
            prediction = make_prediction(technical_params)
            
            # Now prediction has hourly_consumption_kwh and hourly_cost_inr (FIXED)
            hourly_predictions.append({
                "hour": hour,
                "consumption_kwh": prediction.hourly_consumption_kwh,
                "cost_inr": prediction.hourly_cost_inr,
                "time_period": prediction.time_period
            })
        
        # Calculate daily totals
        total_consumption = sum(p["consumption_kwh"] for p in hourly_predictions)
        total_cost = sum(p["cost_inr"] for p in hourly_predictions)
        
        return {
            "date": f"{request.month:02d}/{request.day:02d}",
            "hourly_forecast": hourly_predictions,
            "daily_total_kwh": round(total_consumption, 2),
            "daily_total_cost_inr": round(total_cost, 2),
            "property_type": request.property_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Daily forecast failed: {str(e)}")


@app.post("/compare")
async def compare_scenarios(
    scenario_1: UserFriendlyInput,
    scenario_2: UserFriendlyInput
):
    """Compare two scenarios (e.g., AC On vs Off)."""
    try:
        tech_1 = map_user_friendly_to_technical(scenario_1)
        tech_2 = map_user_friendly_to_technical(scenario_2)
        
        pred_1 = make_prediction(tech_1)
        pred_2 = make_prediction(tech_2)
        
        kwh_diff = pred_1.daily_consumption_kwh - pred_2.daily_consumption_kwh
        cost_diff = pred_1.daily_cost_inr - pred_2.daily_cost_inr
        percent_diff = (kwh_diff / pred_1.daily_consumption_kwh * 100) if pred_1.daily_consumption_kwh > 0 else 0
        
        return {
            "scenario_1": {
                "daily_consumption_kwh": pred_1.daily_consumption_kwh,
                "daily_cost_inr": pred_1.daily_cost_inr,
                "monthly_cost_inr": pred_1.monthly_cost_inr
            },
            "scenario_2": {
                "daily_consumption_kwh": pred_2.daily_consumption_kwh,
                "daily_cost_inr": pred_2.daily_cost_inr,
                "monthly_cost_inr": pred_2.monthly_cost_inr
            },
            "difference": {
                "kwh_saved": round(abs(kwh_diff), 2),
                "cost_saved_inr": round(abs(cost_diff), 2),
                "percent_change": round(abs(percent_diff), 2),
                "recommendation": "Scenario 2 is more efficient" if kwh_diff > 0 else "Scenario 1 is more efficient"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Return model metadata and information."""
    return {
        "model_type": "Random Forest Regressor",
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "model_loaded": MODEL is not None,
        "pricing": {
            "domestic_rate_inr_per_kwh": DOMESTIC_RATE,
            "commercial_rate_inr_per_kwh": COMMERCIAL_RATE
        },
        "input_mappings": {
            "property_size": {"Small": "300 sq ft", "Medium": "750 sq ft", "Large": "1000 sq ft"},
            "weather": {"Cool": "22C, 45% humidity", "Normal": "28C, 55% humidity", "Hot": "35C, 65% humidity"},
            "time_periods": {"0": "Morning (5-12)", "1": "Afternoon (12-17)", "2": "Evening (17-21)", "3": "Night (21-5)"}
        }
    }


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)