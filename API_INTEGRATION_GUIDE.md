# DIAFLUX - API Integration Guide & Code Examples
## For Frontend Development Integration

---

## 🚀 QUICK START API EXAMPLES

### 1. Basic Prediction Call (JavaScript/React)

```javascript
// Example: Making a prediction request
async function getDiabetesPrediction(healthData) {
  try {
    const response = await fetch('http://localhost:5000/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        age: healthData.age,
        gender: healthData.gender,
        hypertension: healthData.hypertension ? 1 : 0,
        heart_disease: healthData.heartDisease ? 1 : 0,
        smoking_history: healthData.smokingHistory,
        bmi: healthData.bmi,
        HbA1c_level: healthData.hba1c,
        blood_glucose_level: healthData.bloodGlucose
      })
    });

    const result = await response.json();
    
    if (result.success) {
      return {
        probability: result.probability,
        riskLevel: result.risk_level,
        confidence: result.confidence,
        recommendations: result.recommendations
      };
    } else {
      console.error('API Error:', result.error);
      throw new Error(result.error);
    }
  } catch (error) {
    console.error('Fetch Error:', error);
    throw error;
  }
}

// Usage Example:
const healthData = {
  age: 45,
  gender: 'Female',
  hypertension: true,
  heartDisease: false,
  smokingHistory: 'former',
  bmi: 28.5,
  hba1c: 6.8,
  bloodGlucose: 145
};

getDiabetesPrediction(healthData)
  .then(prediction => console.log(prediction))
  .catch(error => console.error(error));
```

---

### 2. Using Axios (Recommended)

```javascript
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 5000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Prediction Service
export const predictionService = {
  async predict(healthMetrics) {
    const response = await api.post('/api/predict', {
      age: healthMetrics.age,
      gender: healthMetrics.gender,
      hypertension: healthMetrics.hypertension,
      heart_disease: healthMetrics.heartDisease,
      smoking_history: healthMetrics.smokingHistory,
      bmi: healthMetrics.bmi,
      HbA1c_level: healthMetrics.hba1c,
      blood_glucose_level: healthMetrics.bloodGlucose
    });

    if (response.data.success) {
      return response.data;
    }
    throw new Error(response.data.error || 'Prediction failed');
  },

  async simulate(currentData, modifications) {
    const response = await api.post('/api/simulate', {
      current_data: currentData,
      modifications: modifications
    });

    if (response.data.success) {
      return response.data;
    }
    throw new Error(response.data.error || 'Simulation failed');
  },

  async getRecommendations(riskProfile) {
    const response = await api.post('/api/recommendations', riskProfile);

    if (response.data.success) {
      return response.data;
    }
    throw new Error(response.data.error || 'Failed to get recommendations');
  },

  async checkHealth() {
    const response = await api.get('/api/health');
    return response.data;
  }
};

// Usage in React Component:
import { predictionService } from './services/predictionService';
import { useState } from 'react';

function PredictionForm() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (formData) => {
    setLoading(true);
    setError(null);

    try {
      const prediction = await predictionService.predict(formData);
      setResult(prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    // Component JSX...
  );
}
```

---

### 3. React Hook for Predictions

```javascript
import { useState } from 'react';
import { predictionService } from './services/predictionService';

export function usePrediction() {
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const predict = async (healthMetrics) => {
    setLoading(true);
    setError(null);

    try {
      const result = await predictionService.predict(healthMetrics);
      setPrediction(result);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const simulate = async (currentData, modifications) => {
    setLoading(true);
    setError(null);

    try {
      const result = await predictionService.simulate(currentData, modifications);
      setPrediction(result);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return {
    predict,
    simulate,
    loading,
    prediction,
    error
  };
}

// Usage in Component:
function PredictionComponent() {
  const { predict, loading, prediction, error } = usePrediction();

  const handleFormSubmit = async (formData) => {
    try {
      await predict(formData);
    } catch (err) {
      // Error handled by hook
    }
  };

  return (
    <div>
      {loading && <p>Analyzing your health data...</p>}
      {error && <p>Error: {error}</p>}
      {prediction && (
        <div>
          <h2>Your Risk Assessment</h2>
          <p>Risk Level: {prediction.risk_level}</p>
          <p>Confidence: {prediction.confidence}%</p>
        </div>
      )}
    </div>
  );
}
```

---

### 4. Form Data Handling

```javascript
import { useState } from 'react';

function HealthAssessmentForm({ onSubmit }) {
  const [formData, setFormData] = useState({
    age: '',
    gender: 'Female',
    hypertension: 0,
    heartDisease: 0,
    smokingHistory: 'No Info',
    bmi: '',
    hba1c: '',
    bloodGlucose: ''
  });

  const [errors, setErrors] = useState({});

  // Validation function
  const validateForm = () => {
    const newErrors = {};

    if (!formData.age || formData.age < 18 || formData.age > 100) {
      newErrors.age = 'Age must be between 18 and 100';
    }
    if (!formData.bmi || formData.bmi < 10 || formData.bmi > 60) {
      newErrors.bmi = 'BMI must be between 10 and 60';
    }
    if (!formData.hba1c || formData.hba1c < 3 || formData.hba1c > 14) {
      newErrors.hba1c = 'HbA1c must be between 3 and 14';
    }
    if (!formData.bloodGlucose || formData.bloodGlucose < 40 || formData.bloodGlucose > 400) {
      newErrors.bloodGlucose = 'Blood glucose must be between 40 and 400';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? (checked ? 1 : 0) : value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (validateForm()) {
      onSubmit(formData);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* Age Input */}
      <div>
        <label htmlFor="age">Age:</label>
        <input
          type="number"
          id="age"
          name="age"
          value={formData.age}
          onChange={handleChange}
          min="18"
          max="100"
          required
        />
        {errors.age && <span className="error">{errors.age}</span>}
      </div>

      {/* Gender Select */}
      <div>
        <label htmlFor="gender">Gender:</label>
        <select
          id="gender"
          name="gender"
          value={formData.gender}
          onChange={handleChange}
        >
          <option value="Female">Female</option>
          <option value="Male">Male</option>
        </select>
      </div>

      {/* Hypertension Checkbox */}
      <div>
        <label htmlFor="hypertension">
          <input
            type="checkbox"
            id="hypertension"
            name="hypertension"
            checked={formData.hypertension === 1}
            onChange={handleChange}
          />
          Do you have hypertension?
        </label>
      </div>

      {/* Smoking History */}
      <div>
        <label htmlFor="smokingHistory">Smoking History:</label>
        <select
          id="smokingHistory"
          name="smokingHistory"
          value={formData.smokingHistory}
          onChange={handleChange}
        >
          <option value="never">Never</option>
          <option value="former">Former</option>
          <option value="current">Current</option>
          <option value="ever">Ever</option>
          <option value="No Info">No Info</option>
        </select>
      </div>

      {/* BMI Input */}
      <div>
        <label htmlFor="bmi">BMI:</label>
        <input
          type="number"
          id="bmi"
          name="bmi"
          value={formData.bmi}
          onChange={handleChange}
          min="10"
          max="60"
          step="0.1"
          required
        />
        {errors.bmi && <span className="error">{errors.bmi}</span>}
      </div>

      {/* HbA1c Input */}
      <div>
        <label htmlFor="hba1c">HbA1c Level (%):</label>
        <input
          type="number"
          id="hba1c"
          name="hba1c"
          value={formData.hba1c}
          onChange={handleChange}
          min="3"
          max="14"
          step="0.1"
          required
        />
        {errors.hba1c && <span className="error">{errors.hba1c}</span>}
      </div>

      {/* Blood Glucose Input */}
      <div>
        <label htmlFor="bloodGlucose">Blood Glucose Level (mg/dL):</label>
        <input
          type="number"
          id="bloodGlucose"
          name="bloodGlucose"
          value={formData.bloodGlucose}
          onChange={handleChange}
          min="40"
          max="400"
          required
        />
        {errors.bloodGlucose && <span className="error">{errors.bloodGlucose}</span>}
      </div>

      <button type="submit">Analyze My Risk</button>
      <button type="reset">Clear</button>
    </form>
  );
}

export default HealthAssessmentForm;
```

---

### 5. Results Display Component

```javascript
import React from 'react';
import { CircleChart } from './CircleChart'; // Chart component

function ResultsDisplay({ prediction, loading, error }) {
  if (loading) {
    return <div className="loading">Analyzing your health data...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  if (!prediction) {
    return null;
  }

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'LOW':
        return '#4CAF50';
      case 'MEDIUM':
        return '#FF9800';
      case 'HIGH':
        return '#F44336';
      default:
        return '#2196F3';
    }
  };

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel) {
      case 'LOW':
        return '✓';
      case 'MEDIUM':
        return '⚠';
      case 'HIGH':
        return '⚠';
      default:
        return 'i';
    }
  };

  return (
    <div className="results-container">
      <h2>Your Diabetes Risk Assessment</h2>

      {/* Risk Score Display */}
      <div className="risk-score-section">
        <CircleChart
          percentage={prediction.confidence}
          color={getRiskColor(prediction.risk_level)}
          label={prediction.risk_level}
        />
        <div className="score-details">
          <p className="score-value">{prediction.confidence}%</p>
          <p className="score-label">Risk Score</p>
          <p className="confidence">
            {prediction.probability > 0.7
              ? 'High Confidence'
              : 'Moderate Confidence'}
          </p>
        </div>
      </div>

      {/* Summary Card */}
      <div className="summary-card">
        <h3>Your Health Metrics Summary</h3>
        {prediction.patient_summary && (
          <div className="metrics-grid">
            <div className="metric">
              <span className="label">Age:</span>
              <span className="value">{prediction.patient_summary.age}</span>
            </div>
            <div className="metric">
              <span className="label">BMI:</span>
              <span className="value">{prediction.patient_summary.bmi}</span>
            </div>
            <div className="metric">
              <span className="label">HbA1c:</span>
              <span className="value">{prediction.patient_summary.hba1c}%</span>
            </div>
            <div className="metric">
              <span className="label">Blood Glucose:</span>
              <span className="value">{prediction.patient_summary.blood_glucose} mg/dL</span>
            </div>
          </div>
        )}
      </div>

      {/* Risk Factors */}
      {prediction.risk_factors && prediction.risk_factors.length > 0 && (
        <div className="risk-factors">
          <h3>Risk Factors Identified</h3>
          <ul>
            {prediction.risk_factors.map((factor, index) => (
              <li key={index}>
                <span className="icon">✗</span>
                {factor}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Positive Factors */}
      {prediction.positive_factors && prediction.positive_factors.length > 0 && (
        <div className="positive-factors">
          <h3>Positive Factors</h3>
          <ul>
            {prediction.positive_factors.map((factor, index) => (
              <li key={index}>
                <span className="icon">✓</span>
                {factor}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Recommendations */}
      {prediction.recommendations && prediction.recommendations.length > 0 && (
        <div className="recommendations">
          <h3>Recommended Actions</h3>
          <ol>
            {prediction.recommendations.map((rec, index) => (
              <li key={index}>{rec}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

export default ResultsDisplay;
```

---

### 6. Lifestyle Simulator Component

```javascript
import React, { useState } from 'react';
import { predictionService } from './services/predictionService';

function LifestyleSimulator({ originalData, originalPrediction }) {
  const [modifications, setModifications] = useState({
    bmi: originalData.bmi,
    hba1c: originalData.hba1c,
    bloodGlucose: originalData.bloodGlucose
  });

  const [simResult, setSimResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSliderChange = (metric, value) => {
    setModifications(prev => ({
      ...prev,
      [metric]: parseFloat(value)
    }));
  };

  const runSimulation = async () => {
    setLoading(true);
    try {
      const currentData = {
        ...originalData,
        HbA1c_level: originalData.hba1c,
        blood_glucose_level: originalData.bloodGlucose
      };

      const modifiedData = {
        HbA1c_level: modifications.hba1c,
        blood_glucose_level: modifications.bloodGlucose,
        bmi: modifications.bmi
      };

      const result = await predictionService.simulate(currentData, modifiedData);
      setSimResult(result);
    } catch (error) {
      console.error('Simulation failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const resetToOriginal = () => {
    setModifications({
      bmi: originalData.bmi,
      hba1c: originalData.hba1c,
      bloodGlucose: originalData.bloodGlucose
    });
    setSimResult(null);
  };

  return (
    <div className="simulator-container">
      <h2>Lifestyle Change Simulator</h2>
      <p>Adjust your health metrics to see how lifestyle changes could affect your risk</p>

      <div className="sliders-section">
        {/* BMI Slider */}
        <div className="slider-group">
          <label>
            BMI: {modifications.bmi.toFixed(1)}
            <br />
            <small>Current: {originalData.bmi.toFixed(1)} → Target: {modifications.bmi.toFixed(1)}</small>
          </label>
          <input
            type="range"
            min="15"
            max="45"
            step="0.1"
            value={modifications.bmi}
            onChange={(e) => handleSliderChange('bmi', e.target.value)}
            className="slider"
          />
          <div className="range-labels">
            <span>15</span>
            <span>Healthy (25)</span>
            <span>45</span>
          </div>
        </div>

        {/* HbA1c Slider */}
        <div className="slider-group">
          <label>
            HbA1c Level: {modifications.hba1c.toFixed(1)}%
            <br />
            <small>Current: {originalData.hba1c.toFixed(1)}% → Target: {modifications.hba1c.toFixed(1)}%</small>
          </label>
          <input
            type="range"
            min="4"
            max="10"
            step="0.1"
            value={modifications.hba1c}
            onChange={(e) => handleSliderChange('hba1c', e.target.value)}
            className="slider"
          />
          <div className="range-labels">
            <span>4%</span>
            <span>Normal (5.7%)</span>
            <span>10%</span>
          </div>
        </div>

        {/* Blood Glucose Slider */}
        <div className="slider-group">
          <label>
            Blood Glucose: {modifications.bloodGlucose.toFixed(0)} mg/dL
            <br />
            <small>Current: {originalData.bloodGlucose} → Target: {modifications.bloodGlucose.toFixed(0)}</small>
          </label>
          <input
            type="range"
            min="80"
            max="250"
            step="1"
            value={modifications.bloodGlucose}
            onChange={(e) => handleSliderChange('bloodGlucose', e.target.value)}
            className="slider"
          />
          <div className="range-labels">
            <span>80</span>
            <span>Normal (100)</span>
            <span>250</span>
          </div>
        </div>
      </div>

      <div className="action-buttons">
        <button onClick={runSimulation} disabled={loading} className="btn-primary">
          {loading ? 'Simulating...' : 'See Impact'}
        </button>
        <button onClick={resetToOriginal} className="btn-secondary">
          Reset to Current
        </button>
      </div>

      {simResult && (
        <div className="simulation-results">
          <h3>Simulation Results</h3>
          <div className="comparison">
            <div className="original">
              <h4>Current Risk</h4>
              <p className="risk-score">{(simResult.original_prediction * 100).toFixed(1)}%</p>
            </div>
            <div className="arrow">→</div>
            <div className="simulated">
              <h4>With Changes</h4>
              <p className="risk-score">{(simResult.simulated_prediction * 100).toFixed(1)}%</p>
            </div>
          </div>

          <div className="improvement-summary">
            <p className="improvement">
              <strong>Improvement: {simResult.improvement_percentage}%</strong>
            </p>
            <p>{simResult.impact_summary}</p>
          </div>

          <div className="achievability">
            <h4>Can These Changes Be Achieved?</h4>
            <p><strong>Difficulty:</strong> {simResult.scenario_achievability}</p>
            <p><strong>Timeframe:</strong> 3-6 months with consistent effort</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default LifestyleSimulator;
```

---

## 📋 DATA MAPPING EXAMPLES

### Frontend Field → Backend Field Mapping

```javascript
const fieldMapping = {
  'age' → 'age',
  'gender' → 'gender',
  'hypertension' → 'hypertension' (0 or 1),
  'heart_disease' → 'heart_disease' (0 or 1),
  'smoking_history' → 'smoking_history',
  'bmi' → 'bmi',
  'hba1c' or 'HbA1c_level' → 'HbA1c_level',
  'bloodGlucose' or 'blood_glucose' → 'blood_glucose_level'
}
```

### Response Mapping Example

```javascript
// API Response → Component Props
const predictionResponse = {
  'prediction' → Used for binary classification display
  'probability' → Converted to percentage for confidence display
  'risk_level' → Used for color coding and alert type
  'confidence' → Displayed as percentage (multiply by 100)
  'recommendations' → Mapped to recommendation list items
  'patient_summary' → Used to populate metrics summary
  'risk_factors' → Displayed in red/warning style
  'positive_factors' → Displayed in green/success style
}
```

---

## 🔄 ERROR HANDLING PATTERN

```javascript
async function makeApiCall(endpoint, data) {
  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    // Check HTTP status
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();

    // Check API response status
    if (!result.success) {
      throw new Error(result.error || 'API request failed');
    }

    return result;
  } catch (error) {
    // Handle different error types
    if (error instanceof TypeError) {
      console.error('Network error:', error.message);
      return { success: false, error: 'Network connection failed' };
    } else if (error instanceof SyntaxError) {
      console.error('Invalid response:', error.message);
      return { success: false, error: 'Invalid server response' };
    } else {
      console.error('API error:', error.message);
      return { success: false, error: error.message };
    }
  }
}
```

---

## 📊 CHART VISUALIZATION EXAMPLES

### Risk Score Gauge

```javascript
import Chart from 'chart.js/auto';

function RiskGaugeChart({ probability, riskLevel }) {
  const percentage = Math.round(probability * 100);
  
  const ctx = document.getElementById('riskChart').getContext('2d');
  const riskChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Risk', 'Safe'],
      datasets: [{
        data: [percentage, 100 - percentage],
        backgroundColor: [
          getRiskColor(riskLevel),
          '#e0e0e0'
        ],
        borderWidth: 2,
        borderColor: '#fff'
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: true }
      }
    }
  });

  return riskChart;
}
```

### Before/After Comparison

```javascript
function BeforeAfterChart({ originalRisk, simulatedRisk }) {
  const ctx = document.getElementById('comparisonChart').getContext('2d');
  const comparisonChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Current', 'With Changes'],
      datasets: [{
        label: 'Risk Percentage',
        data: [
          Math.round(originalRisk * 100),
          Math.round(simulatedRisk * 100)
        ],
        backgroundColor: ['#F44336', '#4CAF50']
      }]
    },
    options: {
      indexAxis: 'y',
      scales: {
        x: { max: 100 }
      }
    }
  });

  return comparisonChart;
}
```

---

## 🧪 TESTING CODE EXAMPLES

### Unit Test Example (Jest)

```javascript
import { predictionService } from '../services/predictionService';

describe('Prediction Service', () => {
  test('should make successful prediction', async () => {
    const mockData = {
      age: 45,
      gender: 'Female',
      bmi: 28.5,
      hba1c: 6.8,
      bloodGlucose: 145,
      hypertension: 1,
      heartDisease: 0,
      smokingHistory: 'former'
    };

    const result = await predictionService.predict(mockData);

    expect(result.success).toBe(true);
    expect(result.probability).toBeGreaterThanOrEqual(0);
    expect(result.probability).toBeLessThanOrEqual(1);
    expect(['LOW', 'MEDIUM', 'HIGH']).toContain(result.risk_level);
  });

  test('should handle validation errors', async () => {
    const invalidData = {
      age: 150, // Invalid: exceeds max
      gender: 'Female'
    };

    await expect(predictionService.predict(invalidData))
      .rejects
      .toThrow();
  });
});
```

---

## 🎯 ENVIRONMENT CONFIGURATION

### .env.example

```
REACT_APP_API_URL=http://localhost:5000
REACT_APP_ENV=development
REACT_APP_LOG_LEVEL=debug
```

### Environment-based API URL

```javascript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_URL,
  timeout: 5000
});

export default api;
```

---

## 📱 Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

---

**Document Version:** 1.0  
**Last Updated:** June 3, 2026  
**Status:** Ready for Implementation
