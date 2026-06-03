# DIAFLUX - Technical Specification for Frontend Development
## AI-Assisted Frontend Generation Guide

---

## 📐 SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER (Browser)                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP/HTTPS
        ┌─────────────▼───────────────────┐
        │   FRONTEND APPLICATION           │
        │  (React/Vue/Angular)             │
        │  ┌──────────────────────────┐   │
        │  │ Components:              │   │
        │  │ - Input Form             │   │
        │  │ - Results Dashboard      │   │
        │  │ - Lifestyle Simulator    │   │
        │  │ - Recommendations View   │   │
        │  │ - Educational Content    │   │
        │  └──────────────────────────┘   │
        └─────────────┬────────────────────┘
                      │ REST API Calls
        ┌─────────────▼───────────────────┐
        │   BACKEND API SERVER             │
        │  (Flask/FastAPI/Node.js)         │
        │  ┌──────────────────────────┐   │
        │  │ Endpoints:               │   │
        │  │ - /api/predict           │   │
        │  │ - /api/simulate          │   │
        │  │ - /api/recommendations   │   │
        │  │ - /api/health            │   │
        │  └──────────────────────────┘   │
        └─────────────┬────────────────────┘
                      │
        ┌─────────────▼───────────────────┐
        │   ML MODEL INFERENCE             │
        │  (Loaded from models/*.pkl)      │
        │  - diabetes_model.pkl            │
        │  - scaler.pkl                    │
        └─────────────────────────────────┘
```

---

## 🎨 USER INTERFACE COMPONENTS

### 1. Main Dashboard/Home Page
```
┌──────────────────────────────────────────────────────┐
│                    DiaFlux                           │
│          Diabetes Risk Assessment Tool               │
├──────────────────────────────────────────────────────┤
│                                                      │
│  "Know Your Risk, Control Your Health"              │
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │ [Get Started] [About] [Resources]           │   │
│  └─────────────────────────────────────────────┘   │
│                                                      │
│  Features:                                           │
│  ✓ Instant diabetes risk prediction                 │
│  ✓ Lifestyle impact simulation                      │
│  ✓ Personalized recommendations                     │
│  ✓ Educational resources                            │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 2. Assessment Form Component
```
FORM LAYOUT
─────────────────────────────────────

Personal Information
├─ Age: [slider or number input: 18-100]
├─ Gender: [Female ▼] [Male ▼]
└─ Smoking Status: [dropdown - never/former/current/No Info/ever]

Health Metrics
├─ Height: [input] cm
├─ Weight: [input] kg
├─ BMI: [auto-calculated or input] (18.5-40+)
├─ Blood Pressure: 
│  ├─ Hypertension: [Toggle Yes/No]
│  └─ Heart Disease: [Toggle Yes/No]
└─ Recent Blood Tests:
   ├─ HbA1c Level: [input] (3-9%)
   └─ Blood Glucose Level: [input] mg/dL (70-300)

[ANALYZE MY RISK] [CLEAR FORM]
```

### 3. Results Page Layout
```
┌─────────────────────────────────────────────────────┐
│            YOUR DIABETES RISK ASSESSMENT             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Overall Risk Level                                 │
│  ┌───────────────────────────────────┐             │
│  │  ●●●●○                              │   HIGH    │
│  │  Risk Score: 78%                    │   RISK    │
│  └───────────────────────────────────┘             │
│                                                     │
│  Your Metrics Summary                               │
│  ┌─────────────────────────────────┐              │
│  │ Age: 45 years old                │              │
│  │ BMI: 28.5 (Overweight)           │              │
│  │ HbA1c: 6.8% (Pre-diabetic range) │              │
│  │ Blood Glucose: 145 mg/dL (High)  │              │
│  │ Hypertension: Yes                │              │
│  │ Heart Disease: No                │              │
│  │ Smoking: Former                  │              │
│  └─────────────────────────────────┘              │
│                                                     │
│  What This Means                                    │
│  ⚠️ Your current health metrics indicate an        │
│  elevated risk for diabetes. Early intervention    │
│  is important.                                      │
│                                                     │
│  [Simulate Lifestyle Changes] [Get Recommendations]│
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 4. Lifestyle Simulator Component
```
INTERACTIVE SLIDERS
─────────────────────────────────

Try Different Scenarios:

BMI Adjustment
├─ Current: 28.5
├─ [━━○─────] Set to: 25.0
└─ Impact: 78% → 45% Risk (↓ 33%)

HbA1c Level Adjustment
├─ Current: 6.8%
├─ [━━━━━○──] Set to: 5.8%
└─ Impact: 78% → 52% Risk (↓ 26%)

Blood Glucose Adjustment
├─ Current: 145 mg/dL
├─ [━━━○────] Set to: 120 mg/dL
└─ Impact: 78% → 61% Risk (↓ 17%)

Combined Impact: 78% → 28% Risk (↓ 50%)
[Save Scenario] [Reset to Current]
```

### 5. Recommendations Component
```
PERSONALIZED RECOMMENDATIONS
──────────────────────────────

Based on Your Risk Profile

🍎 Dietary Recommendations
├─ Reduce daily sugar intake to < 25g
├─ Increase fiber intake (25-30g daily)
├─ Choose whole grains over refined
├─ Limit processed foods
└─ Drink plenty of water

🏃 Physical Activity
├─ Aim for 150 minutes moderate exercise per week
├─ Include strength training 2x per week
├─ Take short walks after meals
└─ Reduce sedentary time

⚕️ Medical Recommendations
├─ Schedule annual diabetes screening
├─ Monitor blood pressure regularly
├─ Check cholesterol levels
└─ Consult with endocrinologist

✅ Quick Wins
├─ Lose 5-10% body weight
├─ Increase daily steps to 8,000+
├─ Reduce stress levels
└─ Improve sleep quality (7-9 hours)
```

---

## 🔧 FORM FIELD SPECIFICATIONS

### Input Fields Detailed

```
┌─ DEMOGRAPHIC FIELDS
│
├─ age
│  Type: Number Input or Slider
│  Range: 18-100
│  Default: 30
│  Unit: years
│  Validation: Must be positive integer
│  Placeholder: "Enter your age"
│
├─ gender
│  Type: Radio Button or Dropdown Select
│  Options: ["Female", "Male"]
│  Default: "Female"
│  Required: Yes
│
└─ smoking_history
   Type: Dropdown Select
   Options: ["never", "former", "current", "No Info", "ever"]
   Default: "No Info"
   Required: Yes

┌─ HEALTH STATUS FIELDS
│
├─ hypertension
│  Type: Toggle/Checkbox/Radio
│  Values: 0 (No), 1 (Yes)
│  Label: "Do you have hypertension?"
│  Default: 0
│
└─ heart_disease
   Type: Toggle/Checkbox/Radio
   Values: 0 (No), 1 (Yes)
   Label: "Do you have heart disease?"
   Default: 0

┌─ BIOMETRIC FIELDS
│
├─ height
│  Type: Number Input
│  Range: 100-250 cm (or feet/inches)
│  Unit: cm (or ft-in)
│  Help: "Used to calculate BMI"
│
├─ weight
│  Type: Number Input
│  Range: 30-300 kg
│  Unit: kg (or lbs)
│  Help: "Used to calculate BMI"
│
├─ bmi
│  Type: Auto-calculated or Manual Input
│  Range: 10-60
│  Unit: kg/m²
│  Formula: weight(kg) / (height(m))²
│  Categories:
│    - Underweight: < 18.5
│    - Normal: 18.5-24.9
│    - Overweight: 25-29.9
│    - Obese: ≥ 30
│
└─ height + weight can be auto-calculated into BMI
   OR user can enter BMI directly

┌─ CLINICAL TEST RESULTS
│
├─ HbA1c_level
│  Type: Number Input
│  Range: 3.0-14.0
│  Unit: %
│  Reference Ranges:
│    - Normal: < 5.7%
│    - Pre-diabetic: 5.7-6.4%
│    - Diabetic: ≥ 6.5%
│  Placeholder: "6.5"
│  Help: "3-month average blood glucose"
│
└─ blood_glucose_level
   Type: Number Input
   Range: 40-400
   Unit: mg/dL
   Reference Ranges:
     - Normal (fasting): 70-100 mg/dL
     - Pre-diabetic: 100-125 mg/dL
     - Diabetic: ≥ 126 mg/dL
   Placeholder: "120"
   Help: "Fasting blood glucose level"
```

---

## 📊 RESPONSE DATA STRUCTURES

### Prediction Response
```json
{
  "success": true,
  "prediction": 1,
  "probability": 0.78,
  "risk_level": "HIGH",
  "confidence": 78,
  "risk_interpretation": {
    "title": "High Risk",
    "description": "Your current health metrics indicate elevated diabetes risk",
    "color": "#FF6B6B",
    "icon": "alert-circle"
  },
  "patient_summary": {
    "age": 45,
    "gender": "Female",
    "bmi": 28.5,
    "hba1c": 6.8,
    "blood_glucose": 145,
    "hypertension": true,
    "heart_disease": false
  },
  "risk_factors": [
    "Elevated HbA1c level (pre-diabetic range)",
    "Blood glucose above normal range",
    "BMI in overweight category",
    "Hypertension present"
  ],
  "positive_factors": [
    "No heart disease",
    "Former smoker (reduced risk)",
    "Relatively young age"
  ],
  "recommendations": [
    "Reduce daily sugar intake to <25g",
    "Aim for 150 minutes of moderate exercise per week",
    "Schedule diabetes screening appointment",
    "Lose 5-10% of body weight"
  ]
}
```

### Simulation Response
```json
{
  "success": true,
  "original_scenario": {
    "probability": 0.78,
    "risk_level": "HIGH"
  },
  "simulated_scenario": {
    "modifications": {
      "bmi": 25.0,
      "hba1c_level": 5.8,
      "blood_glucose_level": 120
    },
    "probability": 0.28,
    "risk_level": "LOW"
  },
  "improvement": {
    "probability_reduction": 0.50,
    "percentage_improvement": 50,
    "new_risk_level": "LOW"
  },
  "impact_summary": "Achieving optimal BMI, HbA1c, and glucose levels could reduce your diabetes risk by 50%",
  "scenario_name": "Healthy Lifestyle Changes",
  "achievability": {
    "difficulty": "moderate",
    "timeframe": "6-12 months",
    "tips": [
      "BMI reduction: Requires weight loss of ~10-15 kg through diet and exercise",
      "HbA1c reduction: Achievable through consistent blood sugar management",
      "Glucose reduction: Can be achieved through dietary changes"
    ]
  }
}
```

---

## 🌐 API ENDPOINT SPECIFICATIONS

### Base URL
```
Backend: http://localhost:5000
or
Backend: https://api.diaflux.com
```

### 1. POST /api/predict
**Purpose:** Generate diabetes risk prediction

**Request:**
```json
{
  "age": 45,
  "gender": "Female",
  "hypertension": 1,
  "heart_disease": 0,
  "smoking_history": "former",
  "bmi": 28.5,
  "HbA1c_level": 6.8,
  "blood_glucose_level": 145
}
```

**Response (200):**
```json
{
  "success": true,
  "prediction": 1,
  "probability": 0.78,
  "risk_level": "HIGH",
  "confidence": 78,
  "recommendations": [...]
}
```

**Error Response (400):**
```json
{
  "success": false,
  "error": "Invalid input: age must be between 18 and 100"
}
```

### 2. POST /api/simulate
**Purpose:** Simulate lifestyle changes and predict new risk

**Request:**
```json
{
  "current_data": {
    "age": 45,
    "gender": "Female",
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "former",
    "bmi": 28.5,
    "HbA1c_level": 6.8,
    "blood_glucose_level": 145
  },
  "modifications": {
    "bmi": 25.0,
    "HbA1c_level": 5.8,
    "blood_glucose_level": 120
  }
}
```

**Response (200):**
```json
{
  "success": true,
  "original_prediction": 0.78,
  "simulated_prediction": 0.28,
  "improvement_percentage": 50,
  "impact_summary": "...",
  "scenario_achievability": "moderate"
}
```

### 3. POST /api/recommendations
**Purpose:** Get personalized health recommendations

**Request:**
```json
{
  "risk_level": "HIGH",
  "age": 45,
  "bmi": 28.5,
  "HbA1c_level": 6.8,
  "smoking_history": "former"
}
```

**Response (200):**
```json
{
  "success": true,
  "dietary": [...],
  "physical_activity": [...],
  "medical": [...],
  "lifestyle_changes": [...]
}
```

### 4. GET /api/health
**Purpose:** Check backend health status

**Response (200):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "version": "1.0.0"
}
```

---

## 🎯 VALIDATION RULES

### Frontend Input Validation

```javascript
const validationRules = {
  age: {
    type: "number",
    min: 18,
    max: 100,
    required: true,
    errorMessage: "Age must be between 18 and 100"
  },
  gender: {
    type: "enum",
    values: ["Female", "Male"],
    required: true,
    errorMessage: "Gender must be selected"
  },
  bmi: {
    type: "number",
    min: 10,
    max: 60,
    required: true,
    errorMessage: "BMI must be between 10 and 60"
  },
  HbA1c_level: {
    type: "number",
    min: 3,
    max: 14,
    required: true,
    errorMessage: "HbA1c must be between 3% and 14%"
  },
  blood_glucose_level: {
    type: "number",
    min: 40,
    max: 400,
    required: true,
    errorMessage: "Blood glucose must be between 40-400 mg/dL"
  },
  smoking_history: {
    type: "enum",
    values: ["never", "former", "current", "No Info", "ever"],
    required: true,
    errorMessage: "Smoking status must be selected"
  },
  hypertension: {
    type: "binary",
    values: [0, 1],
    required: true
  },
  heart_disease: {
    type: "binary",
    values: [0, 1],
    required: true
  }
}
```

---

## 🎨 DESIGN SYSTEM

### Color Palette
```
Risk Levels:
├─ LOW RISK: #4CAF50 (Green)
├─ MEDIUM RISK: #FF9800 (Orange)
└─ HIGH RISK: #F44336 (Red)

Primary Colors:
├─ Primary Blue: #2196F3
├─ Secondary Blue: #0D47A1
├─ Accent: #00BCD4
└─ Background: #F5F5F5

Text:
├─ Primary Text: #212121
├─ Secondary Text: #757575
└─ Light Text: #BDBDBD

Status Indicators:
├─ Success: #4CAF50
├─ Warning: #FF9800
├─ Error: #F44336
└─ Info: #2196F3
```

### Typography
```
Headings:
├─ H1: 32px, Bold, Primary Text
├─ H2: 24px, Bold, Primary Text
├─ H3: 20px, Bold, Primary Text
└─ H4: 16px, Bold, Primary Text

Body:
├─ Regular: 14px, Regular, Primary Text
├─ Small: 12px, Regular, Secondary Text
└─ Micro: 10px, Regular, Secondary Text
```

---

## 📱 RESPONSIVE DESIGN BREAKPOINTS

```
Mobile: < 640px (single column, stacked components)
Tablet: 640px - 1024px (two columns when appropriate)
Desktop: > 1024px (full multi-column layout)
```

---

## ⚡ PERFORMANCE REQUIREMENTS

- API response time: < 500ms
- Page load time: < 2 seconds
- Chart rendering: < 1 second
- Form submission: < 1 second
- Mobile optimization: < 3 seconds

---

## 🔐 SECURITY REQUIREMENTS

1. **Input Sanitization:** All user inputs validated and sanitized
2. **HTTPS:** All communications encrypted
3. **CORS:** Proper CORS headers configured
4. **API Key:** Optional authentication for production
5. **Rate Limiting:** Max 100 requests per minute per IP
6. **Data Privacy:** No PII stored; only predictions logged with consent

---

## 📦 DEPENDENCIES (Recommended)

### Frontend Stack
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.4.0",
    "chart.js": "^4.3.0",
    "react-chartjs-2": "^5.2.0",
    "zustand": "^4.3.0"
  },
  "devDependencies": {
    "typescript": "^5.1.0",
    "tailwindcss": "^3.3.0",
    "eslint": "^8.44.0"
  }
}
```

### Backend Stack (Python)
```
flask==2.3.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
joblib==1.2.0
python-dotenv==1.0.0
flask-cors==4.0.0
```

---

## 📋 DEVELOPMENT CHECKLIST

### UI Components to Build
- [ ] Header/Navigation Bar
- [ ] Assessment Form
- [ ] Results Display Card
- [ ] Risk Gauge/Visual Indicator
- [ ] Lifestyle Simulator with Sliders
- [ ] Recommendations List
- [ ] Footer/Navigation

### Functionality to Implement
- [ ] Form validation
- [ ] API integration
- [ ] Error handling
- [ ] Loading states
- [ ] Result caching
- [ ] Responsive design
- [ ] Accessibility (WCAG 2.1)

### Testing Requirements
- [ ] Unit tests (80%+ coverage)
- [ ] Integration tests
- [ ] E2E tests
- [ ] Accessibility audit
- [ ] Performance testing
- [ ] Cross-browser testing

---

## 📞 SUPPORT INFORMATION

**Backend Model Details:**
- Model Type: Ensemble (Best performing among trained models)
- Input Features: 8
- Output: Binary classification (0: No Diabetes, 1: Diabetes)
- Probability Range: 0-1 (0% to 100%)

**Feature Encoding:**
- gender: One-hot encoded (Female=0, Male=1) or Label encoded
- smoking_history: One-hot encoded (5 categories)
- All numerical features: StandardScaler normalized

**Data Source:**
- Reference Notebook: `notebooks/diabetes_prediction_dataset.ipynb`
- Raw Data: `data/raw/diabetes_prediction_dataset.csv`
- Processed Models: `models/diabetes_model.pkl`, `models/scaler.pkl`

---

**Document Version:** 1.0  
**Last Updated:** June 3, 2026  
**Status:** Ready for Frontend Development
