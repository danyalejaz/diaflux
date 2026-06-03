import React from 'react';
import { BookOpen, Award, FileText, Check, Landmark, ArrowRight } from 'lucide-react';

export default function EducationTab() {
  const markerGuides = [
    { name: 'HbA1c (Glycated hemoglobin)', normal: '< 5.7%', pre: '5.7% – 6.4%', diab: '≥ 6.5%', desc: 'Measures your average blood sugar levels over the past 3 months.' },
    { name: 'Fasting Blood Glucose', normal: '< 100 mg/dL', pre: '100 – 125 mg/dL', diab: '≥ 126 mg/dL', desc: 'Measures free-circulating blood sugar in a fasted state of 8-12 hours.' },
    { name: 'Postprandial Glucose (2h)', normal: '< 140 mg/dL', pre: '140 – 199 mg/dL', diab: '≥ 200 mg/dL', desc: 'Measured exactly 2 hours after standard meal initiation.' }
  ];

  return (
    <div className="space-y-8 uppercase tracking-wide animate-in fade-in duration-300" id="education-resource-module">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Metabolic Pathophysiology Column */}
        <div className="lg:col-span-2 space-y-8">
          
          {/* Main Context Card */}
          <div className="bg-[#121214] p-6 sm:p-8 rounded-none border border-white/10 space-y-6">
            <h3 className="text-sm font-black uppercase tracking-widest text-blue-400 flex items-center gap-2 border-b border-white/10 pb-4">
              <BookOpen className="w-5 h-5 text-blue-500" />
              01 // Pathophysiology of glycemia management
            </h3>
            
            <p className="text-xs text-white/70 leading-relaxed font-sans" style={{ textTransform: 'none' }}>
              <strong className="text-white">Diabetes Mellitus</strong> is a chronic metabolic condition where the body either cannot manufacture adequate insulin, or is unable to effectively utilize the hormone it produces. Elevated circulative glucose levels cause systemic microstructural decay across vascular lines, nerves, and neural networks.
            </p>

            <div className="space-y-4 pt-2 font-sans" style={{ textTransform: 'none' }}>
              
              <div className="p-4 bg-white/5 border border-white/10 rounded-none space-y-1.5">
                <span className="text-[11px] font-black tracking-widest text-blue-400 uppercase block">The mechanics of insulin fatigue:</span>
                <p className="text-xs text-white/60 leading-relaxed">
                  In pre-diabetes, cells in muscle and fat systems begin responding poorly to insulin signals. This requires pancreatic beta-cells to over-secrete insulin (hyperinsulinemia) to balance systemic loads. Over time, beta-cells experience prolonged fatigue, pushing glycemic indexes upward.
                </p>
              </div>

              <div className="p-4 bg-blue-950/20 border border-blue-900/50 rounded-none space-y-1.5">
                <span className="text-[11px] font-black tracking-widest text-[#5ca1ff] uppercase block">Importance of early detection:</span>
                <p className="text-xs text-white/75 leading-relaxed">
                  The pre-diabetic window remains highly reversible. Early physical metrics adjustments clear lipid caches inside metabolic units, effectively restoring normal glycemic markers to safe health baselines.
                </p>
              </div>

            </div>
          </div>

          {/* Biomarkers Table - Elegant White Editorial Block */}
          <div className="bg-white text-black p-6 sm:p-8 rounded-none border-b-8 border-blue-600 space-y-6">
            <h3 className="text-xs font-black uppercase tracking-widest text-black/50 flex items-center gap-2 border-b border-black/10 pb-4">
              <FileText className="w-5 h-5 text-blue-600" />
              Target ranges for standard clinical biomarkers
            </h3>
            
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-black/10 text-black/45 uppercase tracking-widest font-black text-[9px]">
                    <th className="py-3">Biomarker</th>
                    <th className="py-3 px-2 text-emerald-700 bg-emerald-50">Normal</th>
                    <th className="py-3 px-2 text-amber-700 bg-amber-50">Prediabetes</th>
                    <th className="py-3 px-2 text-rose-700 bg-rose-50">Diabetic limit</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-black/10 text-black/80 font-black font-sans text-[11px] tracking-wider uppercase">
                  {markerGuides.map((item, idx) => (
                    <tr key={idx} className="hover:bg-black/5 transition-colors">
                      <td className="py-4 pr-3 font-extrabold text-black max-w-[150px]">{item.name}</td>
                      <td className="py-4 px-2 font-mono font-black text-emerald-600">{item.normal}</td>
                      <td className="py-4 px-2 font-mono font-black text-amber-600">{item.pre}</td>
                      <td className="py-4 px-2 font-mono font-black text-rose-600">{item.diab}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="text-[10px] text-black/45 tracking-wider leading-relaxed pt-2 font-sans font-bold uppercase">
              * Reference limits mapped according to instructions set forth by the American Diabetes Association (ADA) and WHO.
            </div>
          </div>

        </div>

        {/* Clinical reference guides sidebar */}
        <div className="space-y-8">
          
          {/* Lifestyle Pillars */}
          <div className="bg-[#121214] p-6 rounded-none border border-white/10 space-y-6">
            <h3 className="text-xs font-black uppercase tracking-widest text-blue-400 flex items-center gap-2 border-b border-white/10 pb-4">
              <Award className="w-5 h-5 text-blue-500" />
              Optimal lifestyle pillars
            </h3>
            
            <div className="space-y-6">
              
              <div className="flex gap-4">
                <span className="font-mono font-black text-blue-400 text-sm mt-0.5 shrink-0 select-none">[01]</span>
                <div className="space-y-1">
                  <span className="text-[11px] font-black uppercase tracking-wider text-white">Sleep Consistency</span>
                  <p className="text-[10px] text-white/50 tracking-wide leading-relaxed font-sans" style={{ textTransform: 'none' }}>
                    Irregular sleep schedules elevate micro-sympathetic cortisol and catecholamine spikes, causing elevated fasting glucose states.
                  </p>
                </div>
              </div>

              <div className="flex gap-4">
                <span className="font-mono font-black text-blue-400 text-sm mt-0.5 shrink-0 select-none">[02]</span>
                <div className="space-y-1">
                  <span className="text-[11px] font-black uppercase tracking-wider text-white">Strength Conditioning</span>
                  <p className="text-[10px] text-white/50 tracking-wide leading-relaxed font-sans" style={{ textTransform: 'none' }}>
                    Skeletal cellular tracks absorb the vast majority of circulative glucose. Resistance training builds insulin-independent pathways.
                  </p>
                </div>
              </div>

              <div className="flex gap-4">
                <span className="font-mono font-black text-blue-400 text-sm mt-0.5 shrink-0 select-none">[03]</span>
                <div className="space-y-1">
                  <span className="text-[11px] font-black uppercase tracking-wider text-white">High Soluble Fibers</span>
                  <p className="text-[10px] text-white/50 tracking-wide leading-relaxed font-sans" style={{ textTransform: 'none' }}>
                    Viscous soluble fiber gels slow down enzymatic breakdown rates, eliminating extreme systemic glucose spikes after meal intakes.
                  </p>
                </div>
              </div>

            </div>
          </div>

          {/* Warning Symptoms block */}
          <div className="bg-rose-950/40 text-rose-200 border border-rose-900 p-6 rounded-none space-y-4">
            <h4 className="font-black text-xs text-rose-400 uppercase tracking-widest block font-sans">// Severe Warning Indicators</h4>
            <p className="text-[10px] tracking-wide text-white/70 leading-relaxed font-sans" style={{ textTransform: 'none' }}>
              Should you observe any of the following persistent physical trends, please coordinate medical examinations immediately with standard clinical care networks:
            </p>
            <ul className="text-xs text-white/80 space-y-2 list-disc pl-5 leading-relaxed font-sans uppercase tracking-wider text-[10px]">
              <li>Frequent excessive hydration needs (polydipsia)</li>
              <li>Chronic, unexplained metabolic exhaustion</li>
              <li>Rapid, unintended weight regression</li>
              <li>Elevated nocturnal urination frequencies</li>
              <li>Tectonic delays in minor cut healing speeds</li>
            </ul>
          </div>
        </div>

      </div>
    </div>
  );
}
