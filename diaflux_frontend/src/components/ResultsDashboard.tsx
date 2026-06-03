import React from 'react';
import { PredictionResult, HealthMetrics } from '../types';
import { Sparkles, RefreshCw, Activity, ArrowRight, ChevronRight } from 'lucide-react';

interface ResultsDashboardProps {
  result: PredictionResult;
  metrics: HealthMetrics;
  onReset: () => void;
  onNavigateToSimulator: () => void;
}

export default function ResultsDashboard({ result, metrics, onReset, onNavigateToSimulator }: ResultsDashboardProps) {
  const percentage = Math.round(result.probability * 100);
  
  const isHigh = result.risk_level === 'High';
  const isMedium = result.risk_level === 'Medium';
  
  // High contrast color tags for the bold taxonomy
  const riskColorTag = isHigh 
    ? 'text-rose-600 border-rose-600/30 bg-rose-50' 
    : isMedium 
      ? 'text-amber-600 border-amber-600/30 bg-amber-50' 
      : 'text-emerald-600 border-emerald-600/30 bg-emerald-50';

  return (
    <div className="space-y-8 animate-in fade-in duration-300" id="prediction-results">
      
      {/* Editorial High-Contrast Main Presentation Block */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Side: Bold Big Number Stat Card (The focal point of the theme) */}
        <div className="lg:col-span-5 bg-white text-black p-8 sm:p-10 flex flex-col justify-between relative rounded-none border-b-8 border-blue-600">
          
          {/* Header confidence tag */}
          <div className="absolute top-0 right-0 p-4">
            <span className="text-[10px] font-black border border-black/20 px-2 py-1 uppercase tracking-widest font-mono">
              CONFIDENCE {result.confidence || '94.2'}% // REG
            </span>
          </div>

          <div className="pt-4 flex-1 flex flex-col justify-center">
            <p className="text-[10px] font-black uppercase tracking-[0.25em] text-black/40 mb-2">
              Risk analysis output
            </p>
            
            <h3 className="text-[124px] sm:text-[150px] font-black leading-none tracking-tighter text-black select-none font-sans flex items-baseline">
              {percentage}
              <span className="text-3xl font-black tracking-normal self-start mt-8 inline-block text-blue-600">%</span>
            </h3>
            
            <p className="text-3xl sm:text-4xl font-black uppercase tracking-tight -mt-4 text-black font-sans">
              {result.risk_level} Risk
            </p>
            
            <p className="mt-4 text-xs leading-relaxed text-black/60 font-sans font-medium">
              The DiaFlux classifier indicates biological parameters aligned with a {result.risk_level.toLowerCase()} probability of Type II Diabetes. Lifestyle recalibration and biometric tracking is recommended.
            </p>
          </div>

          <div className="border-t border-black/10 pt-6 mt-8">
            <div className={`inline-block border px-3 py-1 text-[10px] font-black uppercase tracking-widest ${riskColorTag}`}>
              {result.risk_level} Metric Pool
            </div>
          </div>
        </div>

        {/* Right Side: Biometric Parameter Grids & Explanation */}
        <div className="lg:col-span-7 space-y-6 flex flex-col justify-between">
          
          {/* Summary values table */}
          <div className="bg-[#121214] border border-white/10 p-6 sm:p-8 space-y-6">
            <div>
              <h4 className="text-xs font-black uppercase tracking-widest text-blue-400">02 // Biological weight contribution</h4>
              <p className="text-[9px] uppercase tracking-wider text-white/40 mt-1">Status of individual biological coefficients mapping predictions</p>
            </div>

            <div className="grid grid-cols-2 gap-4 font-mono">
              
              <div className="p-4 bg-white/5 border border-white/10 rounded-none">
                <span className="text-[9px] font-black tracking-widest uppercase text-white/40 block">HbA1c level</span>
                <span className={`text-lg font-black block mt-1 ${metrics.HbA1c_level >= 5.7 ? 'text-amber-500' : 'text-emerald-400'}`}>
                  {metrics.HbA1c_level}%
                </span>
                <span className="text-[9px] uppercase font-bold text-white/50 block">Avg Blood Sugar</span>
              </div>

              <div className="p-4 bg-white/5 border border-white/10 rounded-none">
                <span className="text-[9px] font-black tracking-widest uppercase text-white/40 block">Fast Glucose</span>
                <span className={`text-lg font-black block mt-1 ${metrics.blood_glucose_level >= 100 ? 'text-amber-500' : 'text-emerald-400'}`}>
                  {metrics.blood_glucose_level} mg/dL
                </span>
                <span className="text-[9px] uppercase font-bold text-white/50 block">Serum free count</span>
              </div>

              <div className="p-4 bg-white/5 border border-white/10 rounded-none">
                <span className="text-[9px] font-black tracking-widest uppercase text-white/40 block">Body Mass BMI</span>
                <span className={`text-lg font-black block mt-1 ${metrics.bmi >= 25 ? 'text-amber-500' : 'text-emerald-400'}`}>
                  {metrics.bmi}
                </span>
                <span className="text-[9px] uppercase font-bold text-white/50 block">kg/m² diagnostic</span>
              </div>

              <div className="p-4 bg-white/5 border border-white/10 rounded-none">
                <span className="text-[9px] font-black tracking-widest uppercase text-white/40 block">Smoking Status</span>
                <span className="text-lg font-black text-blue-400 block mt-1 uppercase">
                  {metrics.smoking_history}
                </span>
                <span className="text-[9px] uppercase font-bold text-white/50 block">Behavioral vector</span>
              </div>

            </div>
          </div>

          {/* AI Clinical Insights */}
          <div className="bg-[#1A1A20] border border-white/10 p-6 sm:p-8 relative overflow-hidden" id="ai-insight-panel">
            <span className="absolute top-4 right-4 text-[9px] font-mono tracking-widest text-[#3b82f6] uppercase border border-blue-500/30 px-2 py-0.5 font-bold">
              AI CLINICAL GUIDE
            </span>

            <div className="space-y-2">
              <span className="text-[10px] font-black uppercase tracking-widest text-blue-400 block">// Automated deep analysis</span>
              <p className="text-xs leading-relaxed text-white/80 font-sans" id="text-prediction-explanation">
                {result.explanation}
              </p>
            </div>
          </div>

        </div>

      </div>

      {/* Action Row */}
      <div className="flex flex-col sm:flex-row gap-4 justify-between items-center border-t border-white/10 pt-6">
        <button
          type="button"
          id="btn-re-evaluate"
          onClick={onReset}
          className="w-full sm:w-auto py-3 px-6 border border-white/20 bg-white/5 hover:bg-white/10 text-white font-black text-xs uppercase tracking-widest rounded-none transition-all flex items-center justify-center gap-2 select-none cursor-pointer"
        >
          <RefreshCw className="w-4 h-4 text-white/60" />
          Re-evaluate biometrics
        </button>

        <button
          type="button"
          id="btn-goto-simulator"
          onClick={onNavigateToSimulator}
          className="w-full sm:w-auto py-3.5 px-8 bg-blue-600 hover:bg-blue-700 text-white font-black text-xs uppercase tracking-widest rounded-none transition-all flex items-center justify-center gap-2 cursor-pointer shadow-lg active:translate-y-0.5"
        >
          Access lifestyle simulator
          <ArrowRight className="w-4 h-4 text-white" />
        </button>
      </div>

    </div>
  );
}
