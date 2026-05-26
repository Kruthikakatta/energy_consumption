import React, { useState } from 'react';
import PredictionForm from '../components/PredictionForm';
import ErrorAlert from '../components/ErrorAlert';
import { predictSimple } from '../services/api';
import './Compare.css';

function Compare() {
  const [scenario1, setScenario1] = useState({
    property_type: 'Home',
    property_size: 'Medium',
    people_count: 4,
    ac_usage: 'High',
    lighting_usage: 'Low',
    weather_condition: 'Hot',
    hour: 14,
    day_of_week: 2,
    holiday: 0,
    day: 15,
    month: 5
  });
  
  const [scenario2, setScenario2] = useState({
    property_type: 'Home',
    property_size: 'Medium',
    people_count: 4,
    ac_usage: 'Low',
    lighting_usage: 'Low',
    weather_condition: 'Hot',
    hour: 14,
    day_of_week: 2,
    holiday: 0,
    day: 15,
    month: 5
  });
  
  const [comparison, setComparison] = useState(null);
  const [loading1, setLoading1] = useState(false);
  const [loading2, setLoading2] = useState(false);
  const [error, setError] = useState(null);

  const handleScenario1Change = (name, value) => {
    setScenario1(prev => ({ ...prev, [name]: value }));
  };

  const handleScenario2Change = (name, value) => {
    setScenario2(prev => ({ ...prev, [name]: value }));
  };

  const handleCompare = async (e) => {
    e.preventDefault();
    setLoading1(true);
    setLoading2(true);
    setError(null);
    
    try {
      const [pred1, pred2] = await Promise.all([
        predictSimple(scenario1),
        predictSimple(scenario2)
      ]);
      
      const kwhDiff = pred1.daily_consumption_kwh - pred2.daily_consumption_kwh;
      const costDiff = pred1.daily_cost_inr - pred2.daily_cost_inr;
      const percentDiff = (kwhDiff / pred1.daily_consumption_kwh) * 100;
      
      setComparison({
        scenario1: pred1,
        scenario2: pred2,
        kwhSaved: Math.abs(kwhDiff),
        costSaved: Math.abs(costDiff),
        percentChange: Math.abs(percentDiff),
        winner: kwhDiff > 0 ? 'Scenario 2' : 'Scenario 1',
        message: kwhDiff > 0 
          ? `Scenario 2 saves ${Math.abs(kwhDiff).toFixed(2)} kWh (₹${Math.abs(costDiff).toFixed(2)}) per day`
          : `Scenario 1 saves ${Math.abs(kwhDiff).toFixed(2)} kWh (₹${Math.abs(costDiff).toFixed(2)}) per day`
      });
    } catch (err) {
      setError(err.detail || 'Comparison failed');
    } finally {
      setLoading1(false);
      setLoading2(false);
    }
  };

  const loadPreset = (preset) => {
    if (preset === 'ac_on_off') {
      setScenario1(prev => ({ ...prev, ac_usage: 'High' }));
      setScenario2(prev => ({ ...prev, ac_usage: 'Low' }));
    } else if (preset === 'peak_offpeak') {
      setScenario1(prev => ({ ...prev, hour: 18 }));
      setScenario2(prev => ({ ...prev, hour: 6 }));
    } else if (preset === 'weekday_weekend') {
      setScenario1(prev => ({ ...prev, day_of_week: 2, holiday: 0 }));
      setScenario2(prev => ({ ...prev, day_of_week: 6, holiday: 1 }));
    }
  };

  return (
    <div className="compare-page">
      <div className="page-header">
        <h1>Scenario Comparison</h1>
        <p>Compare different usage patterns to find the most efficient option</p>
      </div>

      <div className="presets">
        <button onClick={() => loadPreset('ac_on_off')} className="preset-btn">
          AC High vs AC Low
        </button>
        <button onClick={() => loadPreset('peak_offpeak')} className="preset-btn">
          Peak Hour vs Off-Peak
        </button>
        <button onClick={() => loadPreset('weekday_weekend')} className="preset-btn">
          Weekday vs Weekend
        </button>
      </div>

      <form onSubmit={handleCompare} className="compare-form">
        <div className="scenarios-container">
          <div className="scenario-card">
            <h3>Scenario 1</h3>
            <PredictionForm
              formData={scenario1}
              onChange={handleScenario1Change}
              onSubmit={(e) => e.preventDefault()}
              isLoading={loading1}
            />
          </div>

          <div className="vs-divider">
            <span className="vs-badge">VS</span>
          </div>

          <div className="scenario-card">
            <h3>Scenario 2</h3>
            <PredictionForm
              formData={scenario2}
              onChange={handleScenario2Change}
              onSubmit={(e) => e.preventDefault()}
              isLoading={loading2}
            />
          </div>
        </div>

        <button type="submit" disabled={loading1 || loading2} className="compare-btn">
          {loading1 || loading2 ? 'Comparing...' : 'Compare Scenarios'}
        </button>
      </form>

      {error && <ErrorAlert message={error} onRetry={handleCompare} />}

      {comparison && !loading1 && !loading2 && (
        <div className="comparison-results">
          <div className="winner-banner">
            Winner: {comparison.winner} is more efficient!
          </div>

          <div className="comparison-grid">
            <div className="comparison-card">
              <h4>Scenario 1</h4>
              <div className="metric">
                <span>Daily Consumption</span>
                <strong>{comparison.scenario1.daily_consumption_kwh} kWh</strong>
              </div>
              <div className="metric">
                <span>Daily Cost</span>
                <strong>₹{comparison.scenario1.daily_cost_inr}</strong>
              </div>
              <div className="metric">
                <span>Monthly Cost</span>
                <strong>₹{comparison.scenario1.monthly_cost_inr}</strong>
              </div>
              <div className="usage-badge" style={{ backgroundColor: 
                comparison.scenario1.usage_category === 'Low' ? '#4caf50' :
                comparison.scenario1.usage_category === 'Moderate' ? '#ff9800' : '#f44336'
              }}>
                {comparison.scenario1.usage_category}
              </div>
            </div>

            <div className="difference-card">
              <div className="difference-value">
                <span className="diff-label">Difference</span>
                <span className="diff-number">-{comparison.kwhSaved} kWh</span>
                <span className="diff-cost">-₹{comparison.costSaved}/day</span>
                <span className="diff-percent">{comparison.percentChange}% reduction</span>
              </div>
              <div className="savings-bar">
                <div 
                  className="savings-fill" 
                  style={{ width: `${Math.min(comparison.percentChange, 100)}%` }}
                ></div>
              </div>
              <p className="savings-message">{comparison.message}</p>
            </div>

            <div className="comparison-card">
              <h4>Scenario 2</h4>
              <div className="metric">
                <span>Daily Consumption</span>
                <strong>{comparison.scenario2.daily_consumption_kwh} kWh</strong>
              </div>
              <div className="metric">
                <span>Daily Cost</span>
                <strong>₹{comparison.scenario2.daily_cost_inr}</strong>
              </div>
              <div className="metric">
                <span>Monthly Cost</span>
                <strong>₹{comparison.scenario2.monthly_cost_inr}</strong>
              </div>
              <div className="usage-badge" style={{ backgroundColor: 
                comparison.scenario2.usage_category === 'Low' ? '#4caf50' :
                comparison.scenario2.usage_category === 'Moderate' ? '#ff9800' : '#f44336'
              }}>
                {comparison.scenario2.usage_category}
              </div>
            </div>
          </div>

          <div className="recommendation-card">
            <strong>Recommendation</strong>
            <p>{comparison.message}. Choose {comparison.winner} for optimal savings.</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default Compare;
