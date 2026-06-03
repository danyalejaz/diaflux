import React, { useState } from 'react';
import { HealthMetrics, PredictionResult } from './types';
import RiskForm from './components/RiskForm';
import ResultsDashboard from './components/ResultsDashboard';
import LifestyleSimulator from './components/LifestyleSimulator';
import RecommendationsTab from './components/RecommendationsTab';
import EducationTab from './components/EducationTab';
import { ShieldAlert, Sparkles, Activity, FileText, Sliders, CheckSquare, GraduationCap, ArrowRight, HeartPulse } from 'lucide-react';

type TabId = 'assess' | 'results' | 'simulate' | 'recommendations' | 'education';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('assess');
  const [metrics, setMetrics] = useState<HealthMetrics | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFormSubmit = async (formData: HealthMetrics) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      });
      
      if (!response.ok) {
        throw new Error('Server was unable to securely process physiological values.');
      }
      
      const resData = await response.json();
      setMetrics(formData);
      setResult(resData);
      setActiveTab('results');
    } catch (err: any) {
      console.error(err);
      setError(err?.message || 'Failed to analyze biometric variables. Check connection status.');
    } finally {
      setIsLoading(false);
    }
  };

  const hasResult = result !== null && metrics !== null;

  return (
    <div className="min-h-screen bg-[#0A0A0B] text-white flex flex-col font-sans transition-all selection:bg-blue-600 selection:text-white pb-12">
      
      {/* Upper Navigation Header bar in Bold Typography theme */}
      <header className="sticky top-0 bg-[#0A0A0B]/90 backdrop-blur-md border-b border-white/15 z-50 py-6" id="main-header">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row sm:items-end sm:justify-between gap-6">
          
          {/* Logo Brand */}
          <div className="flex items-start gap-4">
            <div className="p-3 bg-blue-650 bg-blue-600 rounded-none text-white shadow-md flex items-center justify-center shrink-0">
              <Activity className="w-7 h-7 text-white" id="header-logo-icon" />
            </div>
            <div>
              <div className="flex items-baseline gap-2">
                <h1 className="text-4xl sm:text-5xl font-black tracking-tighter uppercase leading-none font-sans">
                  DiaFlux<span className="text-blue-500">.</span>
                </h1>
                <span className="text-[9px] font-mono tracking-widest text-blue-400 uppercase font-bold border border-blue-500/30 px-2 py-0.5 rounded-none select-none">
                  SYSTEM ACTIVE
                </span>
              </div>
              <p className="text-[10px] tracking-[0.25em] uppercase text-white/50 mt-1 font-sans">
                Diabetes Risk Intelligence & Lifestyle Simulation
              </p>
            </div>
          </div>

          {/* Patient ID & Quick Stats Indicator */}
          <div className="flex items-center gap-6 text-right sm:text-right" id="quick-stats-panel">
            {hasResult ? (
              <div className="flex items-center gap-6 border border-white/10 p-3 bg-white/5 rounded-none">
                <div className="text-left py-0.5">
                  <span className="text-[9px] font-bold text-white/40 uppercase tracking-widest block">Appraisal risk</span>
                  <span className={`text-base font-black font-mono leading-none tracking-tight ${
                    result.risk_level === 'High' ? 'text-rose-500' :
                    result.risk_level === 'Medium' ? 'text-amber-500' :
                    'text-emerald-500'
                  }`}>
                    {Math.round(result.probability * 100)}% // {result.risk_level}
                  </span>
                </div>
                <div className="w-px h-8 bg-white/15"></div>
                <div className="text-left py-0.5 pr-2">
                  <span className="text-[9px] font-bold text-white/40 uppercase tracking-widest block">Metrics anchor</span>
                  <span className="text-base font-black text-white font-mono leading-none">{metrics.age}Y // {metrics.bmi} BMI</span>
                </div>
              </div>
            ) : (
              <div className="border border-white/10 p-3 bg-white/5 rounded-none text-left">
                <p className="text-[9px] uppercase tracking-widest text-white/40">Patient session</p>
                <p className="font-mono text-xs font-semibold text-blue-400">#PX-9920-A (STANDBY)</p>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Primary Layout Frame */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-8 space-y-8" id="dashboard-layout">
        
        {/* Navigation Tabs bar - sharp brutalist buttons */}
        <div className="bg-[#121214] p-1 border border-white/10 flex flex-wrap gap-1 rounded-none" id="navigation-tabs">
          
          <button
            type="button"
            id="tab-assess"
            onClick={() => setActiveTab('assess')}
            className={`py-3 px-4 sm:px-6 flex items-center gap-2 text-xs font-black uppercase tracking-widest transition-all rounded-none cursor-pointer select-none border border-transparent ${
              activeTab === 'assess'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-white/60 hover:bg-white/5 hover:text-white'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            01 // Assessment
          </button>

          <button
            type="button"
            id="tab-results"
            onClick={() => setActiveTab('results')}
            className={`py-3 px-4 sm:px-6 flex items-center gap-2 text-xs font-black uppercase tracking-widest transition-all rounded-none cursor-pointer select-none border border-transparent ${
              activeTab === 'results'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-white/60 hover:bg-white/5 hover:text-white'
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            02 // Risk Report
          </button>

          <button
            type="button"
            id="tab-simulate"
            onClick={() => setActiveTab('simulate')}
            className={`py-3 px-4 sm:px-6 flex items-center gap-2 text-xs font-black uppercase tracking-widest transition-all rounded-none cursor-pointer select-none border border-transparent ${
              activeTab === 'simulate'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-white/60 hover:bg-white/5 hover:text-white'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            03 // Live Simulator
          </button>

          <button
            type="button"
            id="tab-recommendations"
            onClick={() => setActiveTab('recommendations')}
            className={`py-3 px-4 sm:px-6 flex items-center gap-2 text-xs font-black uppercase tracking-widest transition-all rounded-none cursor-pointer select-none border border-transparent ${
              activeTab === 'recommendations'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-white/60 hover:bg-white/5 hover:text-white'
            }`}
          >
            <CheckSquare className="w-3.5 h-3.5" />
            04 // Action Guidelines
          </button>

          <button
            type="button"
            id="tab-education"
            onClick={() => setActiveTab('education')}
            className={`py-3 px-4 sm:px-6 flex items-center gap-2 text-xs font-black uppercase tracking-widest transition-all rounded-none cursor-pointer select-none border border-transparent ${
              activeTab === 'education'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-white/60 hover:bg-white/5 hover:text-white'
            }`}
          >
            <GraduationCap className="w-3.5 h-3.5" />
            05 // Reference Library
          </button>
        </div>

        {/* Global Error Banner */}
        {error && (
          <div className="p-4 bg-rose-950/40 border border-rose-900 rounded-none flex items-start gap-3 text-rose-200 font-sans" id="app-error-banner">
            <ShieldAlert className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-extrabold uppercase tracking-wider text-xs block">Analytical error detected</span>
              <p className="text-xs text-rose-305 text-white/80">{error}</p>
            </div>
          </div>
        )}

        {/* Primary Content Container based on selected tab */}
        <div className="space-y-8" id="dashboard-main-content">
          
          {activeTab === 'assess' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">
              
              {/* Form Input Block */}
              <div className="lg:col-span-7">
                <RiskForm
                  onSubmit={handleFormSubmit}
                  isLoading={isLoading}
                  initialMetrics={metrics || undefined}
                />
              </div>

              {/* Informative Welcome / ML Background sidebar */}
              <div className="lg:col-span-5 bg-white/5 p-8 rounded-none border border-white/10 space-y-8">
                
                <div className="space-y-2">
                  <span className="text-[10px] font-black text-blue-400 uppercase tracking-widest flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4 text-blue-500" />
                    Predictive heuristics architecture
                  </span>
                  <h3 className="font-black text-2xl text-white uppercase tracking-tight leading-none pt-1">
                    Biometric Weight Map
                  </h3>
                  <p className="text-xs text-white/60 leading-relaxed font-sans pt-2">
                    DiaFlux coordinates raw patient physiological parameters through a high-precision multi-variable mathematical model mapped from a reference archive of 100,000+ patient test vectors.
                  </p>
                </div>

                <div className="space-y-6 pt-2 divide-y divide-white/10 font-sans">
                  
                  <div className="space-y-2 pb-1">
                    <span className="text-xs font-bold uppercase tracking-wider text-blue-400 block">// Primary Glycemic Weights</span>
                    <p className="text-xs text-white/70 leading-relaxed">
                      Biometric regressions show that <strong className="text-white">HbA1c levels</strong> and <strong className="text-white">fasting blood sugars</strong> function as the heavy primary vectors inside the calculation model.
                    </p>
                  </div>

                  <div className="space-y-2 pt-6 pb-1">
                    <span className="text-xs font-bold uppercase tracking-wider text-blue-400 block">// Multi-Variable Interlock</span>
                    <p className="text-xs text-white/70 leading-relaxed">
                      Secondary clinical components (age, hypertension, and heart indicators) act multiplicatively on high body masses (BMI) to predict microvascular insulin resistance.
                    </p>
                  </div>

                  <div className="space-y-2 pt-6">
                    <span className="text-xs font-bold uppercase tracking-wider text-blue-450 block text-blue-400 flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5" />
                      // Gemini LLM Co-Pilot Context
                    </span>
                    <p className="text-xs text-indigo-200/80 leading-relaxed">
                      During metric evaluations, DiaFlux coordinates results with structural Gemini prompts to deliver highly nuanced clinic-grade nutrition protocols, fitness plans, and clinical checkup milestones on demand.
                    </p>
                  </div>
                </div>

              </div>
            </div>
          )}

          {activeTab === 'results' && (
            hasResult ? (
              <ResultsDashboard
                result={result}
                metrics={metrics}
                onReset={() => setActiveTab('assess')}
                onNavigateToSimulator={() => setActiveTab('simulate')}
              />
            ) : (
              <PlaceholderCallout onClick={() => setActiveTab('assess')} />
            )
          )}

          {activeTab === 'simulate' && (
            hasResult ? (
              <LifestyleSimulator
                originalMetrics={metrics}
                originalProbability={result.probability}
              />
            ) : (
              <PlaceholderCallout onClick={() => setActiveTab('assess')} />
            )
          )}

          {activeTab === 'recommendations' && (
            hasResult ? (
              <RecommendationsTab prediction={result} />
            ) : (
              <PlaceholderCallout onClick={() => setActiveTab('assess')} />
            )
          )}

          {activeTab === 'education' && (
            <EducationTab />
          )}

        </div>
      </main>

      {/* Sticky platform footer - consistent with the bottom metadata rules */}
      <footer className="mt-auto py-8 bg-[#0A0A0B] border-t border-white/10 text-xs text-white/40 font-sans px-6" id="main-footer">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4 text-center">
          <p className="tracking-wider uppercase text-[10px]">
            © 2026 DIAFLUX MEDICAL SYSTEMS // ML MODEL: DEEP LOGISTIC HEURISTICS
          </p>
          <p className="tracking-wider uppercase text-[10px] font-mono">
            System Synchronization: Constant Fasting Standard
          </p>
        </div>
      </footer>
    </div>
  );
}

// Visual placeholder prompt shown when attempting to access metric-dependent tabs prematurely
function PlaceholderCallout({ onClick }: { onClick: () => void }) {
  return (
    <div className="bg-white text-black p-12 rounded-none border border-transparent shadow-sm text-center max-w-xl mx-auto space-y-6 flex flex-col items-center justify-center font-sans">
      <div className="p-4 bg-slate-100 text-slate-400 rounded-none border border-slate-200">
        <Activity className="w-8 h-8 text-blue-600 animate-pulse" />
      </div>
      
      <div className="space-y-2 max-w-sm">
        <h3 className="font-black text-2xl uppercase tracking-tighter text-slate-900">Biometric appraisal required</h3>
        <p className="text-xs text-slate-600 leading-relaxed">
          Please complete and run your metabolic bio-appraisal under the first tab to configure metrics for custom clinical analysis, real-time lifestyle simulators, and action plans.
        </p>
      </div>

      <button
        type="button"
        id="btn-trigger-placeholder-appraisal"
        onClick={onClick}
        className="py-3.5 px-6 bg-[#0A0A0B] hover:bg-slate-900 border border-transparent font-black uppercase text-xs tracking-widest text-white rounded-none shadow transition-all text-center select-none cursor-pointer flex items-center gap-2"
      >
        Configure Appraisal Metrics
        <ArrowRight className="w-4 h-4 text-blue-400" />
      </button>
    </div>
  );
}

