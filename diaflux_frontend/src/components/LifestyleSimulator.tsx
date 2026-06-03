import React, { useState, useEffect } from 'react';
import { HealthMetrics, SimulationResult } from '../types';
import { Sliders, Sparkles, AlertCircle, ArrowRight, CheckCircle, Flame, Plus } from 'lucide-react';

interface LifestyleSimulatorProps {
  originalMetrics: HealthMetrics;
  originalProbability: number;
}

export default function LifestyleSimulator({ originalMetrics, originalProbability }: LifestyleSimulatorProps) {
  // Simulator inputs initialized to the original metric values
  const [sibmi, setBmi] = useState(originalMetrics.bmi);
  const [sihba1c, setHba1c] = useState(originalMetrics.HbA1c_level);
  const [siglucose, setGlucose] = useState(originalMetrics.blood_glucose_level);

  // Live client-calculated state
  const [liveProb, setLiveProb] = useState(originalProbability);
  const [improvementPercent, setImprovementPercent] = useState(0);

  // Gemini deep simulation impact text state
  const [impactSummary, setImpactSummary] = useState('');
  const [isGeminiLoading, setIsGeminiLoading] = useState(false);

  // Re-score the modified metrics against the real ML backend. Shared by the
  // live (debounced) readout and the detailed impact explanation request.
  const runSimulation = async (): Promise<SimulationResult | null> => {
    try {
      const response = await fetch('/api/simulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          original_data: originalMetrics,
          modifications: {
            bmi: sibmi,
            HbA1c_level: sihba1c,
            blood_glucose_level: siglucose
          }
        })
      });
      if (!response.ok) throw new Error('Simulation request failed');
      return (await response.json()) as SimulationResult;
    } catch (err) {
      console.error("Simulation error:", err);
      return null;
    }
  };

  // Debounced live re-scoring whenever a slider moves, powered by the real model.
  useEffect(() => {
    const handle = setTimeout(async () => {
      const data = await runSimulation();
      if (data) {
        setLiveProb(data.simulated_prediction);
        setImprovementPercent(data.improvement_percentage);
      }
    }, 350);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sibmi, sihba1c, siglucose, originalMetrics]);

  // Fetch the detailed clinical impact narrative for the current slider state.
  const requestGeminiSimulation = async () => {
    setIsGeminiLoading(true);
    setImpactSummary('');
    const data = await runSimulation();
    if (data) {
      setLiveProb(data.simulated_prediction);
      setImprovementPercent(data.improvement_percentage);
      setImpactSummary(data.impact_summary);
    } else {
      setImpactSummary("Failed to fetch custom clinical modeling summary. Maintain optimized ranges under formal dietetic supervision.");
    }
    setIsGeminiLoading(false);
  };

  const origPercent = Math.round(originalProbability * 100);
  const targetPercent = Math.round(liveProb * 100);

  // Colors based on risk levels inside white/black cards
  const getRiskColor = (prob: number) => {
    if (prob >= 0.7) return 'text-rose-600';
    if (prob >= 0.3) return 'text-amber-600';
    return 'text-emerald-600';
  };

  const getRiskBg = (prob: number) => {
    if (prob >= 0.7) return 'bg-rose-500';
    if (prob >= 0.3) return 'bg-amber-500';
    return 'bg-emerald-500';
  };

  const getRiskTextLabel = (prob: number) => {
    if (prob >= 0.7) return 'High Risk';
    if (prob >= 0.3) return 'Moderate';
    return 'Optimal';
  };

  return (
    <div className="space-y-8 uppercase tracking-wide animate-in fade-in duration-300" id="lifestyle-simulator-module">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
        
        {/* Sliders Input Panel with dark background */}
        <div className="lg:col-span-7 bg-[#121214] p-6 sm:p-8 rounded-none border border-white/10 space-y-8 flex flex-col justify-between">
          <div className="space-y-6">
            <div className="flex items-center gap-3 border-b border-white/10 pb-4">
              <div className="p-2.5 bg-blue-600 rounded-none text-white">
                <Sliders className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="text-xs font-black uppercase tracking-widest text-blue-400">03 // Biometric metrics variable deck</h2>
                <p className="text-[10px] text-white/50 lowercase mt-1 tracking-normal font-sans">Slide metrics to model different physical states and observe simulated risk fluctuation</p>
              </div>
            </div>

            <div className="space-y-8">
              {/* BMI Slider */}
              <div className="space-y-2">
                <div className="flex justify-between items-baseline text-xs">
                  <span className="font-extrabold uppercase tracking-wider text-white/70">Body Mass Index (BMI)</span>
                  <span className="font-mono text-2xl font-black text-blue-400">
                    {sibmi} <span className="text-[10px] text-white/50 tracking-widest">KG/M²</span>
                  </span>
                </div>
                <input
                  type="range"
                  id="slider-sim-bmi"
                  min="15"
                  max="45"
                  step="0.1"
                  value={sibmi}
                  onChange={(e) => setBmi(parseFloat(e.target.value))}
                  className="w-full accent-blue-500 cursor-pointer bg-white/10"
                />
                <div className="flex justify-between text-[10px] text-white/40 font-mono">
                  <span>Target Anchor: 22.0</span>
                  <span>Baseline: {originalMetrics.bmi}</span>
                </div>
              </div>

              {/* HbA1c Slider */}
              <div className="space-y-2">
                <div className="flex justify-between items-baseline text-xs">
                  <span className="font-extrabold uppercase tracking-wider text-white/70">Glycated Hemoglobin (HbA1c)</span>
                  <span className="font-mono text-2xl font-black text-blue-400">
                    {sihba1c} <span className="text-[10px] text-white/50 tracking-widest">%</span>
                  </span>
                </div>
                <input
                  type="range"
                  id="slider-sim-hba1c"
                  min="4"
                  max="10"
                  step="0.1"
                  value={sihba1c}
                  onChange={(e) => setHba1c(parseFloat(e.target.value))}
                  className="w-full accent-blue-500 cursor-pointer bg-white/10"
                />
                <div className="flex justify-between text-[10px] text-white/40 font-mono">
                  <span>Target Anchor: &lt;5.7%</span>
                  <span>Baseline: {originalMetrics.HbA1c_level}%</span>
                </div>
              </div>

              {/* Glucose Slider */}
              <div className="space-y-2">
                <div className="flex justify-between items-baseline text-xs">
                  <span className="font-extrabold uppercase tracking-wider text-white/70">Fasting Blood Glucose</span>
                  <span className="font-mono text-2xl font-black text-blue-400">
                    {siglucose} <span className="text-[10px] text-white/50 tracking-widest">MG/DL</span>
                  </span>
                </div>
                <input
                  type="range"
                  id="slider-sim-glucose"
                  min="60"
                  max="250"
                  step="1"
                  value={siglucose}
                  onChange={(e) => setGlucose(parseInt(e.target.value))}
                  className="w-full accent-blue-500 cursor-pointer bg-white/10"
                />
                <div className="flex justify-between text-[10px] text-white/40 font-mono">
                  <span>Target Anchor: &lt;100 mg/dL</span>
                  <span>Baseline: {originalMetrics.blood_glucose_level}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="p-4 bg-white/5 border border-white/10 rounded-none space-y-1 mt-4">
            <span className="text-[10px] font-black text-blue-400 uppercase tracking-widest block">// Simulation guideline</span>
            <p className="text-[10px] text-white/60 leading-relaxed tracking-wider">
              Lowering clinical integers creates logarithmic drops in risk estimates. Adjust variables to safe zones to model patient outcomes.
            </p>
          </div>
        </div>

        {/* Real-time Display and Compare Panel (White Editorial styling) */}
        <div className="lg:col-span-5 bg-white text-black p-8 flex flex-col justify-between rounded-none border-b-8 border-blue-600 h-full">
          
          <div className="space-y-6">
            <span className="text-[10px] uppercase font-black tracking-widest text-black/50 block">Biometric modeling output</span>
            
            <div className="grid grid-cols-2 gap-4 divide-x divide-black/10">
              
              {/* Baseline state */}
              <div className="pr-2 space-y-1">
                <span className="text-[10px] font-black text-black/40 uppercase tracking-widest block font-sans">Baseline risk</span>
                <div className={`text-4xl sm:text-5xl font-black font-mono tracking-tight ${getRiskColor(originalProbability)}`}>
                  {origPercent}%
                </div>
                <span className="text-[10px] text-black/60 font-black uppercase tracking-wider block mt-1">{getRiskTextLabel(originalProbability)} Pool</span>
              </div>

              {/* Simulated target state */}
              <div className="pl-4 space-y-1">
                <span className="text-[10px] font-black text-black/40 uppercase tracking-widest block font-sans">Simulated risk</span>
                <div className={`text-4xl sm:text-5xl font-black font-mono tracking-tight ${getRiskColor(liveProb)}`}>
                  {targetPercent}%
                </div>
                <span className="text-[10px] text-black/80 font-black uppercase tracking-wider block mt-1">{getRiskTextLabel(liveProb)} target</span>
              </div>

            </div>

            {/* Simulated bar reduction indicator */}
            <div className="space-y-2 pt-4">
              <span className="text-[9px] font-black text-black/50 block uppercase tracking-widest">Reduction progression monitor</span>
              <div className="h-4 bg-black/10 rounded-none overflow-hidden flex p-0.5">
                <div className={`transition-all duration-300 ${getRiskBg(liveProb)}`} style={{ width: `${targetPercent}%` }}></div>
              </div>
            </div>
          </div>

          {/* Change readout circle and explanation triggers */}
          <div className="mt-8 pt-6 border-t border-black/15 flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
            <div className="space-y-0.5">
              <span className="text-[10px] text-black/40 font-black uppercase tracking-widest block">Predicted change</span>
              <div className="text-sm font-black text-black uppercase tracking-wider flex items-center gap-1.5">
                {improvementPercent > 0 ? (
                  <>
                    <Flame className="w-4 h-4 text-emerald-600 fill-emerald-250 shrink-0" />
                    <span className="text-emerald-700 font-black">-{improvementPercent}% Reduction</span>
                  </>
                ) : improvementPercent < 0 ? (
                  <span className="text-rose-600 font-black">+{Math.abs(improvementPercent)}% Risk amplification</span>
                ) : (
                  <span className="text-black/50 font-black">No change detected</span>
                )}
              </div>
            </div>

            <button
              type="button"
              id="btn-simulate-gemini"
              onClick={requestGeminiSimulation}
              disabled={isGeminiLoading}
              className="w-full sm:w-auto py-3 px-4 bg-black hover:bg-slate-900 disabled:bg-black/20 text-white text-[10px] font-black uppercase tracking-widest rounded-none transition-all flex items-center justify-center gap-2 cursor-pointer shadow"
            >
              {isGeminiLoading ? (
                <>
                  <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                  Processing...
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5 text-blue-400" />
                  Model impact explanation
                </>
              )}
            </button>
          </div>

        </div>
      </div>

      {/* Dynamic Explanation Results Box from Gemini with Bold styled container */}
      { (impactSummary || isGeminiLoading) && (
        <div className="bg-white text-black p-6 sm:p-8 rounded-none border-b-8 border-blue-600 space-y-4 animate-in fade-in duration-200" id="simulated-explanation-panel">
          <div className="flex items-center gap-2 text-black font-black text-[10px] uppercase tracking-widest border-b border-black/10 pb-3">
            <Sparkles className="w-4.5 h-4.5 text-blue-650" />
            Meta-Level Clinical Impact metrics summary
          </div>
          {isGeminiLoading ? (
            <div className="space-y-2 animate-pulse">
              <div className="h-4 bg-black/10 w-5/6"></div>
              <div className="h-4 bg-black/10 w-2/3"></div>
              <div className="h-4 bg-black/10 w-3/4"></div>
            </div>
          ) : (
            <p className="text-xs font-medium text-black/80 leading-relaxed font-sans cursor-text select-text" style={{ textTransform: 'none' }}>
              {impactSummary}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
