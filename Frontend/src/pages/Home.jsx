import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import PredictionForm from '../components/PredictionForm';
import ResultCard from '../components/ResultCard';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorAlert from '../components/ErrorAlert';
import { predictSimple } from '../services/api';
import { saveToHistory } from '../utils/storage';
import './Home.css';

function Home() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    property_type: 'Home',
    property_size: 'Medium',
    people_count: 4,
    ac_usage: 'Medium',
    lighting_usage: 'Low',
    weather_condition: 'Normal',
    hour: new Date().getHours(),
    day_of_week: new Date().getDay(),
    holiday: 0,
    day: new Date().getDate(),
    month: new Date().getMonth() + 1
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFormChange = (name, value) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const result = await predictSimple(formData);
      setPrediction(result);
      saveToHistory(result);
    } catch (err) {
      setError(err.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="home-page">
      <div className="hero-section">
        <h1>Smart Energy Consumption Forecasting</h1>
        <p>Predict, Compare, and Optimize Your Energy Usage</p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon"></div>
          <div className="stat-content">
            <div className="stat-label">Today's Prediction</div>
            <div className="stat-value">
              {prediction ? `${prediction.daily_consumption_kwh} kWh` : '--'}
            </div>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon"></div>
          <div className="stat-content">
            <div className="stat-label">This Month</div>
            <div className="stat-value">
              {prediction ? `${prediction.monthly_consumption_kwh} kWh` : '--'}
            </div>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon"></div>
          <div className="stat-content">
            <div className="stat-label">Avg Cost/Day</div>
            <div className="stat-value">
              {prediction ? `₹${prediction.daily_cost_inr}` : '--'}
            </div>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon"></div>
          <div className="stat-content">
            <div className="stat-label">Usage Category</div>
            <div className="stat-value">
              {prediction ? prediction.usage_category : '--'}
            </div>
          </div>
        </div>
      </div>

      <div className="content-grid">
        <div className="form-container">
          <PredictionForm
            formData={formData}
            onChange={handleFormChange}
            onSubmit={handleSubmit}
            isLoading={loading}
          />
        </div>
        
        <div className="results-container">
          {loading && <LoadingSpinner />}
          {error && <ErrorAlert message={error} onRetry={handleSubmit} />}
          {prediction && !loading && <ResultCard prediction={prediction} />}
        </div>
      </div>

      <div className="action-buttons">
        <button onClick={() => navigate('/daily-forecast')} className="action-btn">
          View Daily Forecast
        </button>
        <button onClick={() => navigate('/compare')} className="action-btn">
          Compare Scenarios
        </button>
        <button onClick={() => navigate('/history')} className="action-btn">
          View History
        </button>
      </div>
    </div>
  );
}

export default Home;