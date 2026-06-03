export interface HealthMetrics {
  gender: 'Female' | 'Male';
  age: number;
  hypertension: 0 | 1;
  heart_disease: 0 | 1;
  smoking_history: 'never' | 'former' | 'current' | 'No Info' | 'ever';
  bmi: number;
  HbA1c_level: number;
  blood_glucose_level: number;
}

export interface PredictionResult {
  success: boolean;
  prediction: number;
  probability: number;
  risk_level: 'Low' | 'Medium' | 'High';
  confidence: number;
  recommendations: {
    dietary: string[];
    exercise: string[];
    medical: string[];
  };
  explanation: string;
}

export interface SimulationRequest {
  original_data: HealthMetrics;
  modifications: {
    bmi: number;
    HbA1c_level: number;
    blood_glucose_level: number;
  };
}

export interface SimulationResult {
  original_prediction: number;
  simulated_prediction: number;
  improvement_percentage: number;
  impact_summary: string;
}
