import React, { useState } from 'react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorAlert from '../components/ErrorAlert';
import { predictDaily } from '../services/api';
import './DailyForecast.css';

function DailyForecast() {
  const [formData, setFormData] = useState({
    property_type: 'Home',
    property_size: 'Medium',
    people_count: 4,
    ac_usage: 'Medium',
    lighting_usage: 'Low',
    weather_condition: 'Normal',
    day_of_week: new Date().getDay(),
    holiday: 0,
    day: new Date().getDate(),
    month: new Date().getMonth() + 1
  });
  
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const result = await predictDaily(formData);
      setForecast(result);
    } catch (err) {
      setError(err.detail || 'Daily forecast failed');
    } finally {
      setLoading(false);
    }
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="tooltip-hour">Hour: {label}:00</p>
          <p className="tooltip-value">Consumption: {payload[0].value} kWh</p>
          <p className="tooltip-cost">Cost: ₹{payload[0].payload.cost}</p>
          <p className="tooltip-period">Period: {payload[0].payload.time_period}</p>
        </div>
      );
    }
    return null;
  };

  const getPeakHour = () => {
    if (!forecast) return null;
    const max = Math.max(...forecast.hourly_forecast.map(h => h.consumption_kwh));
    const peak = forecast.hourly_forecast.find(h => h.consumption_kwh === max);
    return peak;
  };

  const getLowestHour = () => {
    if (!forecast) return null;
    const min = Math.min(...forecast.hourly_forecast.map(h => h.consumption_kwh));
    const lowest = forecast.hourly_forecast.find(h => h.consumption_kwh === min);
    return lowest;
  };

  const peakHour = getPeakHour();
  const lowestHour = getLowestHour();

  return (
    <div className="daily-forecast-page">
      <div className="page-header">
        <h1>24-Hour Energy Forecast</h1>
        <p>See how your energy consumption varies throughout the day</p>
      </div>

      <div className="forecast-form-container">
        <form onSubmit={handleSubmit} className="forecast-form">
          <div className="form-row">
            <div className="form-group">
              <label>Property Type</label>
              <select name="property_type" value={formData.property_type} onChange={handleChange}>
                <option value="Home">Home</option>
                <option value="Commercial">Commercial</option>
              </select>
            </div>

            <div className="form-group">
              <label>Property Size</label>
              <select name="property_size" value={formData.property_size} onChange={handleChange}>
                <option value="Small">Small</option>
                <option value="Medium">Medium</option>
                <option value="Large">Large</option>
              </select>
            </div>

            <div className="form-group">
              <label>People Count</label>
              <input type="number" name="people_count" value={formData.people_count} onChange={handleChange} min="1" />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>AC Usage</label>
              <select name="ac_usage" value={formData.ac_usage} onChange={handleChange}>
                <option value="Low">Low</option>
                <option value="Medium">Medium</option>
                <option value="High">High</option>
              </select>
            </div>

            <div className="form-group">
              <label>Lighting Usage</label>
              <select name="lighting_usage" value={formData.lighting_usage} onChange={handleChange}>
                <option value="Low">Low</option>
                <option value="Medium">Medium</option>
                <option value="High">High</option>
              </select>
            </div>

            <div className="form-group">
              <label>Weather</label>
              <select name="weather_condition" value={formData.weather_condition} onChange={handleChange}>
                <option value="Cool">Cool</option>
                <option value="Normal">Normal</option>
                <option value="Hot">Hot</option>
              </select>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Day of Week</label>
              <input type="number" name="day_of_week" value={formData.day_of_week} onChange={handleChange} min="0" max="6" />
            </div>

            <div className="form-group">
              <label>Holiday</label>
              <select name="holiday" value={formData.holiday} onChange={handleChange}>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>

            <div className="form-group">
              <label>Date</label>
              <div className="date-group">
                <input type="number" name="day" value={formData.day} onChange={handleChange} min="1" max="31" placeholder="Day" />
                <input type="number" name="month" value={formData.month} onChange={handleChange} min="1" max="12" placeholder="Month" />
              </div>
            </div>
          </div>

          <button type="submit" disabled={loading} className="generate-btn">
            {loading ? 'Generating...' : 'Generate 24-Hour Forecast'}
          </button>
        </form>
      </div>

      {error && <ErrorAlert message={error} onRetry={handleSubmit} />}

      {loading && <LoadingSpinner />}

      {forecast && !loading && (
        <div className="forecast-results">
          <div className="summary-cards">
            <div className="summary-card">
              <div className="summary-icon"></div>
              <div className="summary-content">
                <div className="summary-label">Daily Total</div>
                <div className="summary-value">{forecast.daily_total_kwh} kWh</div>
              </div>
            </div>
            
            <div className="summary-card">
              <div className="summary-icon"></div>
              <div className="summary-content">
                <div className="summary-label">Total Cost</div>
                <div className="summary-value">₹{forecast.daily_total_cost_inr}</div>
              </div>
            </div>
            
            {peakHour && (
              <div className="summary-card">
                <div className="summary-icon"></div>
                <div className="summary-content">
                  <div className="summary-label">Peak Hour</div>
                  <div className="summary-value">{peakHour.hour}:00 ({peakHour.consumption_kwh} kWh)</div>
                </div>
              </div>
            )}
            
            {lowestHour && (
              <div className="summary-card">
                <div className="summary-icon"></div>
                <div className="summary-content">
                  <div className="summary-label">Lowest Hour</div>
                  <div className="summary-value">{lowestHour.hour}:00 ({lowestHour.consumption_kwh} kWh)</div>
                </div>
              </div>
            )}
          </div>

          <div className="chart-container">
            <h3>Hourly Consumption Pattern</h3>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={forecast.hourly_forecast}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" label={{ value: 'Hour', position: 'bottom' }} />
                <YAxis label={{ value: 'kWh', angle: -90, position: 'left' }} />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="consumption_kwh"
                  stroke="#667eea"
                  fill="url(#colorGradient)"
                />
                <defs>
                  <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#667eea" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#667eea" stopOpacity={0}/>
                  </linearGradient>
                </defs>
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="hourly-table-container">
            <h3>Hourly Breakdown</h3>
            <div className="table-wrapper">
              <table className="hourly-table">
                <thead>
                  <tr>
                    <th>Hour</th>
                    <th>Time Period</th>
                    <th>Consumption (kWh)</th>
                    <th>Cost (₹)</th>
                  </tr>
                </thead>
                <tbody>
                  {forecast.hourly_forecast.map((hour) => (
                    <tr key={hour.hour} className={hour.consumption_kwh === peakHour?.consumption_kwh ? 'peak-row' : ''}>
                      <td>{hour.hour}:00</td>
                      <td>{hour.time_period}</td>
                      <td>{hour.consumption_kwh}</td>
                      <td>₹{hour.cost_inr}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default DailyForecast;
