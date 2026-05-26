import React, { useState, useEffect } from 'react';
import { getModelInfo } from '../services/api';
import './About.css';

function About() {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const info = await getModelInfo();
      setModelInfo(info);
    } catch (error) {
      console.error('Failed to fetch model info:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="about-page">
      <div className="page-header">
        <h1>About This Tool</h1>
        <p>Learn how our energy consumption forecasting system works</p>
      </div>

      <div className="about-grid">
        <div className="about-card">
          <h3>How It Works</h3>
          <p>Our system uses machine learning to predict energy consumption based on multiple factors:</p>
          <ul>
            <li>Building characteristics (size, type, occupancy)</li>
            <li>Usage patterns (AC, lighting)</li>
            <li>Environmental conditions (temperature, humidity)</li>
            <li>Temporal factors (hour, day, month, holidays)</li>
          </ul>
        </div>

        <div className="about-card">
          <h3>Model Details</h3>
          {loading ? (
            <p>Loading model information...</p>
          ) : modelInfo ? (
            <>
              <p><strong>Model Type:</strong> {modelInfo.model_type}</p>
              <p><strong>Features:</strong> {modelInfo.feature_count} parameters</p>
              <p><strong>Status:</strong> {modelInfo.model_loaded ? 'Active' : 'Using fallback'}</p>
            </>
          ) : (
            <p>Model information unavailable</p>
          )}
        </div>

        <div className="about-card">
          <h3>Pricing Information</h3>
          <p>Electricity rates used for cost calculations:</p>
          <ul>
            <li><strong>Domestic (Home):</strong> ₹6.5 per kWh</li>
            <li><strong>Commercial:</strong> ₹9.0 per kWh</li>
          </ul>
          <p className="note">*Based on average Indian electricity tariffs</p>
        </div>

        <div className="about-card">
          <h3>Tips for Best Results</h3>
          <ul>
            <li>Select the correct property size for your building</li>
            <li>Be honest about AC and lighting usage patterns</li>
            <li>Consider seasonal variations in weather</li>
            <li>Use scenario comparison for "what-if" analysis</li>
          </ul>
        </div>

        <div className="about-card">
          <h3>Features Used</h3>
          <div className="features-grid">
            <span className="feature-tag">Temperature</span>
            <span className="feature-tag">Humidity</span>
            <span className="feature-tag">Square Footage</span>
            <span className="feature-tag">Occupancy</span>
            <span className="feature-tag">HVAC Usage</span>
            <span className="feature-tag">Lighting Usage</span>
            <span className="feature-tag">Day of Week</span>
            <span className="feature-tag">Holiday</span>
            <span className="feature-tag">Hour</span>
            <span className="feature-tag">Day</span>
            <span className="feature-tag">Month</span>
          </div>
        </div>

        <div className="about-card">
          <h3>Tech Stack</h3>
          <ul>
            <li><strong>Backend:</strong> FastAPI, Python, scikit-learn</li>
            <li><strong>Frontend:</strong> React, Axios, Recharts</li>
            <li><strong>Model:</strong> Random Forest Regressor</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default About;
