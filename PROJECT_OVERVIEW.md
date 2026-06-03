# DIAFLUX - Project Overview
## Diabetes Risk Prediction with Lifestyle Impact Simulation Using Machine Learning

---

## 📋 PROJECT SUMMARY

**Project Name:** DiaFlux  
**Project Type:** Machine Learning Application - Diabetes Risk Prediction  
**Purpose:** Predict diabetes risk in patients based on health metrics and lifestyle factors, with capability to simulate impact of lifestyle changes  
**Status:** Backend ML Models Trained | Frontend Pending Development  

---

## 🎯 PROJECT OBJECTIVES

1. **Diabetes Risk Prediction:** Build a machine learning model to predict whether a patient has or will develop diabetes
2. **Lifestyle Impact Simulation:** Allow users to adjust lifestyle parameters and see how those changes affect their diabetes risk
3. **User-Friendly Interface:** Provide an intuitive frontend for patients and healthcare providers to interact with the model
4. **Health Insights:** Generate actionable insights and recommendations based on prediction results

---

## 📊 DATASET INFORMATION

### Dataset Details
- **Name:** Diabetes Prediction Dataset
- **File Location:** `data/raw/diabetes_prediction_dataset.csv`
- **Size:** Comprehensive dataset with multiple health metrics
- **Format:** CSV (Comma-Separated Values)
- **Processing:** Already analyzed and processed in notebooks for ML model training

### Dataset Features (Input Variables)

| Feature Name | Data Type | Description | Range/Values |
|---|---|---|---|
| **gender** | Categorical | Patient's gender | Female, Male |
| **age** | Numerical | Patient's age in years | Integer (typically 18-80) |
| **hypertension** | Binary | Whether patient has hypertension | 0 (No), 1 (Yes) |
| **heart_disease** | Binary | Whether patient has heart disease | 0 (No), 1 (Yes) |
| **smoking_history** | Categorical | Smoking status | never, former, current, No Info, ever |
| **bmi** | Numerical | Body Mass Index | Float (typically 18.5-40+) |
| **HbA1c_level** | Numerical | Hemoglobin A1c level (3-month glucose average) | Float (typically 4-9%) |
| **blood_glucose_level** | Numerical | Fasting blood glucose level | Integer (typically 70-300 mg/dL) |

### Target Variable

| Variable | Description | Values |
|---|---|---|
| **diabetes** | Whether patient has diabetes | 0 (No Diabetes), 1 (Diabetes) |

### Data Statistics
- **Total Records:** 100,000+ samples (including synthetic test data)
- **Total Features:** 8 input features + 1 target variable
- **Feature Types:** 3 categorical, 4 numerical, 1 binary target
- **Missing Values:** Analyzed and handled in preprocessing
- **Data Quality:** Validated for impossible values, outliers, and duplicates

---

## 🤖 MACHINE LEARNING MODELS

### Models Trained & Evaluated

Multiple machine learning algorithms were trained and compared:

1. **Logistic Regression**
   - Classical statistical approach
   - Good for binary classification
   - Interpretable results

2. **Random Forest Classifier**
   - Ensemble method using decision trees
   - Handles non-linear relationships
   - Provides feature importance scores

3. **Support Vector Machine (SVM)**
   - Kernel-based classification
   - Effective for high-dimensional data
   - Uses RBF kernel for non-linear separation

4. **Gradient Boosting Classifier**
   - Advanced ensemble method
   - Sequential tree building
   - Strong predictive performance

### Model Selection Process
- **Metric:** Composite scoring based on Accuracy (50%) + F1-Score (50%)
- **Best Model:** Selected based on highest composite score
- **Training:** 80-20 train-test split
- **Evaluation:** Accuracy, F1-Score, ROC-AUC Score

### Model Files
- **Location:** `models/`
- **diabetes_model.pkl** - Trained best-performing model (serialized)
- **scaler.pkl** - StandardScaler for feature normalization

---

## 🏗️ BACKEND ARCHITECTURE

### Technology Stack
- **Language:** Python 3
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib
- **Web Framework:** Streamlit (for demo/deployment)

### Project Structure
```
diaflux/
├── data/
│   ├── raw/
│   │   ├── diabetes_prediction_dataset.csv
│   │   └── diabetes.csv
│   └── processed/
├── models/
│   ├── diabetes_model.pkl          # Best trained model
│   └── scaler.pkl                  # Feature scaler
├── notebooks/
│   ├── diabetes_prediction_dataset.ipynb    # Main analysis & exploration
│   └── diabetes.ipynb                       # Secondary analysis
├── src/                             # Backend source code (pending)
├── frontend/                        # Frontend code (pending development)
├── test_model_saving.py            # Model training & serialization script
├── requirements.txt                # Python dependencies
└── README.md                        # Project documentation
```

### Dependencies
```
pandas              # Data manipulation and analysis
numpy               # Numerical computing
scikit-learn        # Machine learning algorithms
matplotlib          # Data visualization
seaborn             # Statistical visualization
joblib              # Model serialization
streamlit           # Web app framework
```

---

## 🔮 PREDICTION WORKFLOW

### Input Processing
1. User provides health metrics through frontend interface
2. Input features validated and normalized using the saved scaler
3. Features transformed to match training data format

### Prediction Steps
1. **Feature Scaling:** Raw input values normalized using StandardScaler
2. **Model Inference:** Scaled features passed to trained diabetes_model.pkl
3. **Probability Output:** Model returns probability of diabetes (0-1 scale)
4. **Risk Classification:** 
   - Low Risk: Probability < 0.3
   - Medium Risk: Probability 0.3-0.7
   - High Risk: Probability > 0.7

### Output Format
```json
{
  "prediction": 1 or 0,
  "probability": 0.0 - 1.0,
  "risk_level": "Low/Medium/High",
  "confidence": "percentage",
  "recommendations": ["array of health recommendations"],
  "lifestyle_impact": {
    "if_improve_bmi": "new_prediction",
    "if_increase_exercise": "new_prediction",
    "if_improve_diet": "new_prediction"
  }
}
```

---

## 🎨 FRONTEND REQUIREMENTS

### Frontend Purpose
Interactive web application allowing users to:
- Input their health metrics
- Receive diabetes risk predictions
- Simulate lifestyle changes and see impact on predictions
- View health recommendations
- Access educational resources about diabetes

### Key Frontend Pages/Components

1. **User Input Form**
   - Age (slider or input)
   - Gender (dropdown: Female, Male)
   - Hypertension (checkbox/toggle)
   - Heart Disease (checkbox/toggle)
   - Smoking History (dropdown: never, former, current, No Info, ever)
   - BMI (input or calculator)
   - HbA1c Level (input with range guide)
   - Blood Glucose Level (input)

2. **Results Dashboard**
   - Risk score visualization (gauge/progress bar)
   - Risk level display (Low/Medium/High with color coding)
   - Confidence percentage
   - Key health metrics summary

3. **Lifestyle Simulator**
   - Sliders to adjust BMI, HbA1c, blood glucose
   - Real-time prediction updates
   - Before/After comparison
   - Impact visualization

4. **Health Recommendations**
   - Personalized dietary recommendations
   - Exercise suggestions
   - Medical checkup recommendations
   - Risk reduction strategies

5. **Educational Resources**
   - Information about diabetes
   - Understanding test results
   - Lifestyle improvement tips
   - Healthcare provider contact info

### Suggested Frontend Technologies
- **React.js** or **Vue.js** - Interactive UI
- **Chart.js** or **D3.js** - Data visualization
- **Tailwind CSS** - Styling
- **Axios** - API communication
- **React Query** - State management

---

## 📡 API ENDPOINTS (Backend Requirements)

The frontend will need to communicate with these backend API endpoints:

### 1. Predict Endpoint
```
POST /api/predict
Content-Type: application/json

Request Body:
{
  "gender": "Female",
  "age": 45,
  "hypertension": 0,
  "heart_disease": 0,
  "smoking_history": "never",
  "bmi": 27.5,
  "HbA1c_level": 5.8,
  "blood_glucose_level": 140
}

Response:
{
  "success": true,
  "prediction": 1,
  "probability": 0.75,
  "risk_level": "High",
  "confidence": 75,
  "recommendations": [...]
}
```

### 2. Simulate Endpoint
```
POST /api/simulate
Content-Type: application/json

Request Body:
{
  "original_data": {...},
  "modifications": {
    "bmi": 25.0,
    "HbA1c_level": 5.5,
    "blood_glucose_level": 120
  }
}

Response:
{
  "original_prediction": 0.75,
  "simulated_prediction": 0.45,
  "improvement_percentage": 40,
  "impact_summary": "..."
}
```

### 3. Recommendations Endpoint
```
POST /api/recommendations
Content-Type: application/json

Request Body:
{
  "risk_level": "High",
  "health_metrics": {...}
}

Response:
{
  "dietary_recommendations": [...],
  "exercise_recommendations": [...],
  "medical_recommendations": [...]
}
```

---

## 📈 DATA PREPROCESSING PIPELINE

The models were trained using the following preprocessing steps:

1. **Missing Value Handling:** Identified and treated null values
2. **Duplicate Removal:** Removed duplicate records
3. **Outlier Detection:** Used IQR method to identify anomalies
4. **Feature Encoding:** Categorical variables encoded appropriately
5. **Feature Scaling:** StandardScaler applied to normalize numerical features
6. **Class Balancing:** Dataset has ~88% negative (no diabetes) and ~12% positive (diabetes)

---

## 📊 MODEL PERFORMANCE METRICS

### Evaluation Metrics Used
- **Accuracy:** Percentage of correct predictions
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC Score:** Area under the ROC curve
- **Sensitivity/Recall:** True positive rate
- **Specificity:** True negative rate

### Model Selection Criteria
Best model selected based on:
- Highest composite score (Accuracy 50% + F1-Score 50%)
- Balanced performance across all metrics
- Ability to handle imbalanced dataset (12% positive class)

---

## 🚀 DEPLOYMENT CONSIDERATIONS

### Backend Deployment
- Can be deployed using Streamlit for quick prototype
- Production deployment using Flask/FastAPI + Gunicorn + Nginx
- Docker containerization recommended

### Model Serving
- Load model from `models/diabetes_model.pkl` at startup
- Load scaler from `models/scaler.pkl` for feature normalization
- Ensure consistent Python versions and dependencies

### Frontend Deployment
- Can be deployed on Vercel, Netlify, or GitHub Pages (static sites)
- Use environment variables for API endpoint configuration
- Implement proper error handling and loading states

---

## 📋 DEVELOPMENT ROADMAP

### Phase 1: Frontend Development (Current)
- [ ] Design UI mockups
- [ ] Develop input form component
- [ ] Implement results visualization
- [ ] Build lifestyle simulator
- [ ] Create recommendations engine

### Phase 2: Backend API
- [ ] Build Flask/FastAPI server
- [ ] Implement prediction endpoint
- [ ] Add simulation functionality
- [ ] Create recommendations generator
- [ ] Add error handling and validation

### Phase 3: Integration & Testing
- [ ] Connect frontend to backend APIs
- [ ] End-to-end testing
- [ ] User acceptance testing
- [ ] Performance optimization

### Phase 4: Deployment
- [ ] Set up production environment
- [ ] Configure CI/CD pipeline
- [ ] Deploy to cloud platform
- [ ] Set up monitoring and logging

---

## 🔒 Security & Privacy Considerations

1. **Data Privacy:** No patient data should be stored without consent
2. **Input Validation:** All frontend inputs must be validated
3. **HTTPS:** Use HTTPS for all API communications
4. **Rate Limiting:** Implement rate limiting to prevent abuse
5. **Error Messages:** Don't expose sensitive information in errors

---

## 📚 ADDITIONAL RESOURCES

### Files to Reference
- **Main Analysis Notebook:** `notebooks/diabetes_prediction_dataset.ipynb` - Comprehensive data exploration and visualization
- **Model Training Script:** `test_model_saving.py` - Model selection and serialization logic
- **Data Dictionary:** Defined in this document

### Key Insights from Analysis
- Dataset includes both demographic and clinical health metrics
- Clear class imbalance (88% no diabetes, 12% diabetes) - handled during training
- Multiple ML algorithms compared for optimal performance
- Feature scaling essential for model performance

---

## 📞 CONTACT & SUPPORT

For questions about:
- **Dataset structure:** Refer to notebooks and data exploration output
- **Model details:** Check test_model_saving.py
- **Data processing:** Review notebooks for preprocessing steps
- **Frontend integration:** Use API specifications provided above

---

**Generated on:** June 3, 2026  
**Project Status:** Backend Complete | Frontend Pending Development  
**Last Updated:** Current Analysis Session
