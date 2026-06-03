import React, { useState } from 'react';
import { HealthMetrics } from '../types';
import { Calculator, HelpCircle, Activity, Info, ChevronRight } from 'lucide-react';

interface RiskFormProps {
  onSubmit: (metrics: HealthMetrics) => void;
  isLoading: boolean;
  initialMetrics?: HealthMetrics;
}

export default function RiskForm({ onSubmit, isLoading, initialMetrics }: RiskFormProps) {
  const [metrics, setMetrics] = useState<HealthMetrics>(initialMetrics || {
    gender: 'Female',
    age: 45,
    hypertension: 0,
    heart_disease: 0,
    smoking_history: 'never',
    bmi: 24.5,
    HbA1c_level: 5.5,
    blood_glucose_level: 95
  });

  const [showBmiCalc, setShowBmiCalc] = useState(false);
  const [weight, setWeight] = useState('70');
  const [height, setHeight] = useState('170');

  const handleInputChange = (key: keyof HealthMetrics, value: any) => {
    setMetrics(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const calculateBmi = () => {
    const w = parseFloat(weight);
    const h = parseFloat(height) / 100; // cm to meters
    if (w > 0 && h > 0) {
      const bmiVal = parseFloat((w / (h * h)).toFixed(1));
      handleInputChange('bmi', bmiVal);
      setShowBmiCalc(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(metrics);
  };

  return (
    <form id="risk-assessment-form" onSubmit={handleFormSubmit} className="space-y-6">
      <div className="bg-[#121214] p-6 sm:p-8 rounded-none border border-white/10 space-y-8">
        
        {/* Step Header */}
        <div className="flex justify-between items-center border-b border-white/10 pb-4">
          <div>
            <h2 className="text-xs font-black uppercase tracking-widest text-blue-400">01 // Patient Physiological parameters</h2>
            <p className="text-[10px] text-white/50 uppercase tracking-widest mt-1">Provide clinic-measured values for optimal prediction fidelity</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
          
          {/* Gender */}
          <div className="space-y-2">
            <label className="text-[10px] uppercase tracking-widest text-white/60 font-black block" id="label-gender">Biological Sex</label>
            <div className="grid grid-cols-2 gap-3" id="input-gender-group">
              <button
                type="button"
                id="btn-gender-female"
                onClick={() => handleInputChange('gender', 'Female')}
                className={`py-3.5 px-4 rounded-none border text-xs font-black uppercase tracking-wider transition-all cursor-pointer text-center ${
                  metrics.gender === 'Female'
                    ? 'border-blue-500 bg-blue-600/20 text-white'
                    : 'border-white/15 hover:border-white/30 text-white/60 bg-white/5'
                }`}
              >
                Female
              </button>
              <button
                type="button"
                id="btn-gender-male"
                onClick={() => handleInputChange('gender', 'Male')}
                className={`py-3.5 px-4 rounded-none border text-xs font-black uppercase tracking-wider transition-all cursor-pointer text-center ${
                  metrics.gender === 'Male'
                    ? 'border-blue-500 bg-blue-600/20 text-white'
                    : 'border-white/15 hover:border-white/30 text-white/60 bg-white/5'
                }`}
              >
                Male
              </button>
            </div>
          </div>

          {/* Age range slider with precise bold indicators */}
          <div className="space-y-2">
            <div className="flex justify-between items-baseline">
              <label className="text-[10px] uppercase tracking-widest text-white/60 font-black" id="label-age">Patient Age</label>
              <span className="text-lg font-black font-mono text-blue-400">{metrics.age} <span className="text-[10px] uppercase tracking-widest font-black text-white/50">YRS</span></span>
            </div>
            <div className="pt-2">
              <input
                type="range"
                id="input-age-range"
                min="18"
                max="80"
                value={metrics.age}
                onChange={(e) => handleInputChange('age', parseInt(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer bg-white/10"
              />
              <div className="flex justify-between text-[9px] text-white/40 font-mono tracking-widest mt-1">
                <span>18_</span>
                <span>49_</span>
                <span>80_</span>
              </div>
            </div>
          </div>

          {/* Hypertension status toggles */}
          <div className="space-y-2">
            <label className="text-[10px] uppercase tracking-widest text-white/60 font-black block" id="label-hypertension">Hypertension Status</label>
            <div className="grid grid-cols-2 gap-3" id="input-hypertension-group">
              <button
                type="button"
                id="btn-hyper-no"
                onClick={() => handleInputChange('hypertension', 0)}
                className={`py-3.5 px-4 rounded-none border text-xs font-black uppercase tracking-wider transition-all cursor-pointer text-center ${
                  metrics.hypertension === 0
                    ? 'border-blue-500 bg-blue-600/20 text-white'
                    : 'border-white/15 hover:border-white/30 text-white/60 bg-white/5'
                }`}
              >
                Negative
              </button>
              <button
                type="button"
                id="btn-hyper-yes"
                onClick={() => handleInputChange('hypertension', 1)}
                className={`py-3.5 px-4 rounded-none border text-xs font-black uppercase tracking-wider transition-all cursor-pointer text-center ${
                  metrics.hypertension === 1
                    ? 'border-rose-500 bg-rose-950/40 text-rose-400'
                    : 'border-white/15 hover:border-white/30 text-white/60 bg-white/5'
                }`}
              >
                Positive (Active)
              </button>
            </div>
          </div>

          {/* Heart disease status toggles */}
          <div className="space-y-2">
            <label className="text-[10px] uppercase tracking-widest text-white/60 font-black block" id="label-heart-disease">Heart Disease Status</label>
            <div className="grid grid-cols-2 gap-3" id="input-heart-group">
              <button
                type="button"
                id="btn-heart-no"
                onClick={() => handleInputChange('heart_disease', 0)}
                className={`py-3.5 px-4 rounded-none border text-xs font-black uppercase tracking-wider transition-all cursor-pointer text-center ${
                  metrics.heart_disease === 0
                    ? 'border-blue-500 bg-blue-600/20 text-white'
                    : 'border-white/15 hover:border-white/30 text-white/60 bg-white/5'
                }`}
              >
                Negative
              </button>
              <button
                type="button"
                id="btn-heart-yes"
                onClick={() => handleInputChange('heart_disease', 1)}
                className={`py-3.5 px-4 rounded-none border text-xs font-black uppercase tracking-wider transition-all cursor-pointer text-center ${
                  metrics.heart_disease === 1
                    ? 'border-rose-500 bg-rose-950/40 text-rose-400'
                    : 'border-white/15 hover:border-white/30 text-white/60 bg-white/5'
                }`}
              >
                Positive (Active)
              </button>
            </div>
          </div>

          {/* Smoking history status selection drop-down */}
          <div className="space-y-2">
            <label className="text-[10px] uppercase tracking-widest text-white/60 font-black block" id="label-smoking">Smoking History Category</label>
            <select
              id="select-smoking-history"
              value={metrics.smoking_history}
              onChange={(e) => handleInputChange('smoking_history', e.target.value)}
              className="w-full bg-[#1A1A20] border border-white/15 rounded-none px-4 py-3.5 text-white text-xs font-black uppercase tracking-wider focus:outline-none focus:border-blue-500 transition-all cursor-pointer"
            >
              <option value="never">Never Smoked (Optimal)</option>
              <option value="former">Former Smoker</option>
              <option value="current">Current Smoker (High Threshold)</option>
              <option value="ever">Ever Smoked (Intermittent)</option>
              <option value="No Info">No Clinical Diagnosis</option>
            </select>
          </div>

          {/* BMI (Body Mass Index) Numerical Input with micro-calculators */}
          <div className="space-y-2 relative">
            <div className="flex justify-between items-baseline">
              <label className="text-[10px] uppercase tracking-widest text-white/60 font-black" id="label-bmi">Body Mass Index (BMI)</label>
              <button
                type="button"
                id="btn-toggle-bmi-calc"
                onClick={() => setShowBmiCalc(!showBmiCalc)}
                className="text-[10px] text-blue-400 hover:text-blue-300 flex items-center gap-1 font-black uppercase tracking-widest transition-colors"
                title="Estimate BMI"
              >
                <Calculator className="w-3.5 h-3.5" />
                // RUN ESTIMATOR
              </button>
            </div>
            
            <div className="relative">
              <input
                type="number"
                id="input-bmi-number"
                step="0.1"
                min="10"
                max="60"
                value={metrics.bmi}
                onChange={(e) => handleInputChange('bmi', parseFloat(e.target.value) || 24)}
                className="w-full bg-[#1A1A20] border border-white/15 rounded-none pl-4 pr-16 py-3 text-white text-xl font-black focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 transition-all font-mono"
              />
              <span className="absolute right-4 top-1/2 -translate-y-1/2 text-[10px] font-black uppercase tracking-widest text-white/40">KG/M²</span>
            </div>

            {/* Premium brutalist range markings */}
            <div className="flex justify-between text-[9px] uppercase tracking-widest font-black select-none opacity-60">
              <span className={metrics.bmi < 18.5 ? 'text-amber-400 font-extrabold' : 'text-white/40'}>Under</span>
              <span className={metrics.bmi >= 18.5 && metrics.bmi < 25 ? 'text-emerald-400 font-extrabold' : 'text-white/40'}>Normal</span>
              <span className={metrics.bmi >= 25 && metrics.bmi < 30 ? 'text-amber-400 font-extrabold' : 'text-white/40'}>Over</span>
              <span className={metrics.bmi >= 30 ? 'text-rose-500 font-extrabold' : 'text-white/40'}>Obese</span>
            </div>

            {/* Estimated BMI micro popover */}
            {showBmiCalc && (
              <div id="bmi-calculator-modal" className="absolute z-10 top-18 left-0 right-0 p-5 bg-[#16161A] border border-white/15 rounded-none shadow-2xl space-y-4 animate-in fade-in slide-in-from-top-2 duration-200">
                <h4 className="text-[10px] font-black tracking-widest text-blue-400 uppercase flex items-center gap-1.5">
                  <Calculator className="w-4 h-4" /> Quick metric converter
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="text-[9px] font-bold text-white/50 uppercase tracking-widest block">Mass (kg)</label>
                    <input
                      type="number"
                      value={weight}
                      onChange={(e) => setWeight(e.target.value)}
                      className="w-full bg-white/5 border border-white/15 rounded-none p-2 text-sm font-black text-white font-mono"
                      placeholder="e.g. 70"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] font-bold text-white/50 uppercase tracking-widest block">Stature (cm)</label>
                    <input
                      type="number"
                      value={height}
                      onChange={(e) => setHeight(e.target.value)}
                      className="w-full bg-white/5 border border-white/15 rounded-none p-2 text-sm font-black text-white font-mono"
                      placeholder="e.g. 170"
                    />
                  </div>
                </div>
                <div className="flex gap-2 justify-end pt-2">
                  <button
                    type="button"
                    onClick={() => setShowBmiCalc(false)}
                    className="px-3 py-1.5 text-[10px] font-black uppercase tracking-widest text-white/60 hover:text-white"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={calculateBmi}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-[10px] font-black uppercase tracking-widest rounded-none shadow-md transition-all"
                  >
                    Calculate bmi
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* HbA1c Level */}
          <div className="space-y-2">
            <div className="flex justify-between items-baseline">
              <label className="text-[10px] uppercase tracking-widest text-white/60 font-black flex items-center gap-1" id="label-hba1c">
                HbA1c Level (%)
                <span className="group relative text-white/30 hover:text-white/60 cursor-help">
                  <HelpCircle className="w-3.5 h-3.5" />
                  <span className="absolute bottom-[130%] left-1/2 -translate-x-1/2 w-48 p-3 bg-slate-900 border border-white/10 text-white text-[9px] tracking-normal font-normal leading-normal opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity z-20 shadow-xl">
                    HbA1c represents your average blood glucose levels over the past 2 to 3 months.
                  </span>
                </span>
              </label>
              <span className={`text-sm font-black font-mono transition-colors ${
                metrics.HbA1c_level >= 6.5 ? 'text-rose-500' :
                metrics.HbA1c_level >= 5.7 ? 'text-amber-500' :
                'text-emerald-400'
              }`}>{metrics.HbA1c_level} %</span>
            </div>
            
            <div className="relative">
              <input
                type="number"
                id="input-hba1c-number"
                step="0.1"
                min="3"
                max="12"
                value={metrics.HbA1c_level}
                onChange={(e) => handleInputChange('HbA1c_level', parseFloat(e.target.value) || 5.0)}
                className="w-full bg-[#1A1A20] border border-white/15 rounded-none px-4 py-3 text-white text-xl font-black focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 transition-all font-mono"
              />
            </div>

            <div className="flex justify-between text-[9px] uppercase tracking-widest font-black select-none opacity-60">
              <span className={metrics.HbA1c_level < 5.7 ? 'text-emerald-400 font-extrabold' : 'text-white/40'}>Norm (&lt;5.7)</span>
              <span className={metrics.HbA1c_level >= 5.7 && metrics.HbA1c_level < 6.5 ? 'text-amber-400 font-extrabold' : 'text-white/40'}>Pre (5.7-6.4)</span>
              <span className={metrics.HbA1c_level >= 6.5 ? 'text-rose-500 font-extrabold' : 'text-white/40'}>Diabetes (&ge;6.5)</span>
            </div>
          </div>

          {/* Blood Glucose Level Indicator */}
          <div className="space-y-2">
            <div className="flex justify-between items-baseline">
              <label className="text-[10px] uppercase tracking-widest text-white/60 font-black flex items-center gap-1" id="label-glucose">
                Blood Glucose level
                <span className="group relative text-white/30 hover:text-white/60 cursor-help">
                  <HelpCircle className="w-3.5 h-3.5" />
                  <span className="absolute bottom-[130%] left-1/2 -translate-x-1/2 w-48 p-3 bg-slate-900 border border-white/10 text-white text-[9px] tracking-normal font-normal leading-normal opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity z-20 shadow-xl">
                    Concentration of free glucose circulating in fasting blood serum, measured in mg/dL.
                  </span>
                </span>
              </label>
              <span className={`text-sm font-black font-mono transition-colors ${
                metrics.blood_glucose_level >= 126 ? 'text-rose-500' :
                metrics.blood_glucose_level >= 100 ? 'text-amber-500' :
                'text-emerald-400'
              }`}>{metrics.blood_glucose_level} <span className="text-[10px] font-black text-white/50">MG/DL</span></span>
            </div>
            
            <div className="relative">
              <input
                type="number"
                id="input-glucose-number"
                step="1"
                min="50"
                max="400"
                value={metrics.blood_glucose_level}
                onChange={(e) => handleInputChange('blood_glucose_level', parseInt(e.target.value) || 80)}
                className="w-full bg-[#1A1A20] border border-white/15 rounded-none px-4 py-3 text-white text-xl font-black focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 transition-all font-mono"
              />
            </div>

            <div className="flex justify-between text-[9px] uppercase tracking-widest font-black select-none opacity-60">
              <span className={metrics.blood_glucose_level < 100 ? 'text-emerald-400 font-extrabold' : 'text-white/40'}>Normal (&lt;100)</span>
              <span className={metrics.blood_glucose_level >= 100 && metrics.blood_glucose_level < 126 ? 'text-amber-400 font-extrabold' : 'text-white/40'}>Pre (100-125)</span>
              <span className={metrics.blood_glucose_level >= 126 ? 'text-rose-500 font-extrabold' : 'text-white/40'}>Diabetes (&ge;126)</span>
            </div>
          </div>
        </div>

        {/* Informative Disclaimer Card */}
        <div className="p-4 bg-white/5 border border-white/10 rounded-none flex items-start gap-3">
          <Info className="w-4 h-4 text-white/50 shrink-0 mt-0.5" />
          <p className="text-[10px] text-white/60 leading-relaxed uppercase tracking-wider">
            <strong>System warning anchor:</strong> Clinical measurements are evaluated against current WHO / ADA criteria. Values dictate downstream predictive matrices inside the neural classifier.
          </p>
        </div>
      </div>

      <button
        type="submit"
        id="btn-submit-assessment"
        disabled={isLoading}
        className="w-full py-4.5 px-6 bg-blue-600 hover:bg-blue-700 border border-transparent disabled:bg-white/10 transition-all font-black uppercase tracking-widest text-white text-sm rounded-none cursor-pointer flex items-center justify-center gap-3 transition-colors active:translate-y-0.5 shadow-lg shadow-blue-500/10"
      >
        {isLoading ? (
          <div className="flex items-center justify-center gap-2">
            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
            Syncing predictive arrays...
          </div>
        ) : (
          <>
            <span>Run ML Risk Appraisal</span>
            <ChevronRight className="w-5 h-5 text-blue-300" />
          </>
        )}
      </button>
    </form>
  );
}
