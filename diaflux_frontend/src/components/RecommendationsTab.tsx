import React, { useState } from 'react';
import { PredictionResult } from '../types';
import { Apple, Dumbbell, ShieldCheck, CheckSquare, Stethoscope } from 'lucide-react';

interface RecommendationsTabProps {
  prediction: PredictionResult;
}

export default function RecommendationsTab({ prediction }: RecommendationsTabProps) {
  const { recommendations } = prediction;
  const [completedItems, setCompletedItems] = useState<Record<string, boolean>>({});

  const toggleItem = (category: string, index: number) => {
    const key = `${category}-${index}`;
    setCompletedItems(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-300" id="recommendations-container">
      
      {/* Intro Summary Banner */}
      <div className="p-6 bg-[#121214] border border-white/10 rounded-none space-y-2">
        <h3 className="text-xs font-black uppercase tracking-widest text-blue-400 flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-blue-500" />
          Proactive Clinical Action Plan ({prediction.risk_level} Risk Category)
        </h3>
        <p className="text-xs text-white/60 lowercase tracking-wide leading-relaxed font-sans" style={{ textTransform: 'none' }}>
          Based on the evaluated biological variables, the predictive models compiled these specific metabolic lifestyle action steps. Check the items off as you discuss or integrate them into your metabolic care plan.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* 1. Dietary Interventions */}
        <div className="bg-white text-black p-6 rounded-none border-b-6 border-blue-600 flex flex-col justify-between space-y-6">
          <div className="space-y-5">
            <div className="flex items-center gap-2 pb-3 border-b border-black/10">
              <div className="p-1.5 bg-black text-white rounded-none">
                <Apple className="w-4 h-4 text-white" />
              </div>
              <h4 className="font-black text-xs uppercase tracking-widest text-black">Dietary & Nutrition</h4>
            </div>

            <div className="space-y-3 font-sans">
              {recommendations.dietary.map((rec, i) => {
                const key = `dietary-${i}`;
                const isDone = !!completedItems[key];
                return (
                  <div
                    key={i}
                    onClick={() => toggleItem('dietary', i)}
                    className={`p-3.5 border transition-all cursor-pointer flex gap-3 text-xs font-bold uppercase tracking-wider leading-normal select-none ${
                      isDone
                        ? 'bg-slate-100 border-black/5 text-black/35 line-through'
                        : 'bg-black/5 hover:bg-black/10 border-black/10 text-black'
                    }`}
                  >
                    <CheckSquare className={`w-4 h-4 shrink-0 mt-0.5 ${isDone ? 'text-black/30' : 'text-blue-600'}`} />
                    <span>{rec}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* 2. Physical & Sports Protocols */}
        <div className="bg-white text-black p-6 rounded-none border-b-6 border-blue-600 flex flex-col justify-between space-y-6">
          <div className="space-y-5">
            <div className="flex items-center gap-2 pb-3 border-b border-black/10">
              <div className="p-1.5 bg-black text-white rounded-none">
                <Dumbbell className="w-4 h-4 text-white" />
              </div>
              <h4 className="font-black text-xs uppercase tracking-widest text-black">Fitness progressions</h4>
            </div>

            <div className="space-y-3 font-sans">
              {recommendations.exercise.map((rec, i) => {
                const key = `exercise-${i}`;
                const isDone = !!completedItems[key];
                return (
                  <div
                    key={i}
                    onClick={() => toggleItem('exercise', i)}
                    className={`p-3.5 border transition-all cursor-pointer flex gap-3 text-xs font-bold uppercase tracking-wider leading-normal select-none ${
                      isDone
                        ? 'bg-slate-100 border-black/5 text-black/35 line-through'
                        : 'bg-black/5 hover:bg-black/10 border-black/10 text-black'
                    }`}
                  >
                    <CheckSquare className={`w-4 h-4 shrink-0 mt-0.5 ${isDone ? 'text-black/30' : 'text-blue-600'}`} />
                    <span>{rec}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* 3. Clinical & Physician Procedures */}
        <div className="bg-white text-black p-6 rounded-none border-b-6 border-blue-600 flex flex-col justify-between space-y-6">
          <div className="space-y-5">
            <div className="flex items-center gap-2 pb-3 border-b border-black/10">
              <div className="p-1.5 bg-black text-white rounded-none">
                <Stethoscope className="w-4 h-4 text-white" />
              </div>
              <h4 className="font-black text-xs uppercase tracking-widest text-black">Clinical procedures</h4>
            </div>

            <div className="space-y-3 font-sans">
              {recommendations.medical.map((rec, i) => {
                const key = `medical-${i}`;
                const isDone = !!completedItems[key];
                return (
                  <div
                    key={i}
                    onClick={() => toggleItem('medical', i)}
                    className={`p-3.5 border transition-all cursor-pointer flex gap-3 text-xs font-bold uppercase tracking-wider leading-normal select-none ${
                      isDone
                        ? 'bg-slate-100 border-black/5 text-black/35 line-through'
                        : 'bg-black/5 hover:bg-black/10 border-black/10 text-black'
                    }`}
                  >
                    <CheckSquare className={`w-4 h-4 shrink-0 mt-0.5 ${isDone ? 'text-black/30' : 'text-blue-600'}`} />
                    <span>{rec}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

      </div>

      {/* Action Disclaimer */}
      <div className="text-[10px] text-white/40 tracking-wider text-center max-w-lg mx-auto leading-normal uppercase">
        * Always coordinate lifestyle or diet changes under formal primary physician or specialized clinical supervision.
      </div>
    </div>
  );
}
