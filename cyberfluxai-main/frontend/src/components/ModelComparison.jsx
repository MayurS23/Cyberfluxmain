import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from 'recharts';

const ModelComparison = ({ models, alerts }) => {
  // Use real model metrics from backend
  const getModelScores = () => {
    if (!models || models.length === 0) return [];
    
    return models.map(model => ({
      name: model.name,
      score: model.accuracy || 0,
      accuracy: model.accuracy?.toFixed(1) || '0.0',
      precision: model.precision?.toFixed(1) || '0.0',
      recall: model.recall?.toFixed(1) || '0.0',
      f1_score: model.f1_score?.toFixed(1) || '0.0'
    }));
  };

  const modelScores = getModelScores();

  const radarData = models.map(m => ({
    model: m.name,
    Accuracy: m.accuracy || 0,
    Precision: m.precision || 0,
    Recall: m.recall || 0,
    'F1-Score': m.f1_score || 0
  }));

  return (
    <div className="space-y-6" data-testid="model-comparison">
      {/* Model Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {models.map((model) => (
          <Card key={model.name} className="bg-slate-900/50 border-slate-800 hover:border-slate-700 transition-colors" data-testid={`model-card-${model.name}`}>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg text-slate-100" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>{model.name}</CardTitle>
                <Badge 
                  className="bg-emerald-500/20 text-emerald-400 border-emerald-500/50 border"
                  data-testid={`model-status-${model.name}`}
                >
                  Active
                </Badge>
              </div>
              <CardDescription className="text-slate-400">{model.type}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {/* Accuracy */}
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Accuracy</span>
                    <span className="text-slate-100 font-semibold">{model.accuracy?.toFixed(1)}%</span>
                  </div>
                  <Progress 
                    value={model.accuracy || 0} 
                    className="h-2 bg-slate-800"
                  />
                </div>

                {/* F1-Score & Latency */}
                <div className="grid grid-cols-2 gap-3 pt-2">
                  <div>
                    <p className="text-xs text-slate-400">F1-Score</p>
                    <p className="text-sm font-semibold text-cyan-400">{model.f1_score?.toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Latency</p>
                    <p className="text-sm font-semibold text-orange-400">{model.latency_ms}ms</p>
                  </div>
                </div>

                {/* Additional Metrics */}
                <div className="grid grid-cols-2 gap-3 pt-1 border-t border-slate-800">
                  <div className="pt-2">
                    <p className="text-xs text-slate-400">Precision</p>
                    <p className="text-sm text-slate-300">{model.precision?.toFixed(1)}%</p>
                  </div>
                  <div className="pt-2">
                    <p className="text-xs text-slate-400">Recall</p>
                    <p className="text-sm text-slate-300">{model.recall?.toFixed(1)}%</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Model Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bar Chart */}
        <Card className="bg-slate-900/50 border-slate-800">
          <CardHeader>
            <CardTitle className="text-slate-100" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Model Accuracy Comparison</CardTitle>
            <CardDescription className="text-slate-400">Performance metrics across models</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelScores}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94a3b8" angle={-45} textAnchor="end" height={80} />
                <YAxis stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                  labelStyle={{ color: '#f1f5f9' }}
                />
                <Bar dataKey="score" fill="#06b6d4" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Radar Chart */}
        <Card className="bg-slate-900/50 border-slate-800">
          <CardHeader>
            <CardTitle className="text-slate-100" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Multi-Metric Analysis</CardTitle>
            <CardDescription className="text-slate-400">Accuracy, Precision, Recall comparison</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="model" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis stroke="#94a3b8" />
                <Radar name="Accuracy" dataKey="Accuracy" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.3} />
                <Radar name="Precision" dataKey="Precision" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
                <Radar name="Recall" dataKey="Recall" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.3} />
                <Legend />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                  labelStyle={{ color: '#f1f5f9' }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Ensemble Decision Visualizer */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader>
          <CardTitle className="text-slate-100" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Ensemble Decision Process</CardTitle>
          <CardDescription className="text-slate-400">Model contribution weights and performance</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {models.map((model) => (
              <div key={model.name} className="space-y-2" data-testid={`ensemble-model-${model.name}`}>
                <div className="flex justify-between items-center">
                  <div className="flex items-center space-x-3">
                    <span className="text-sm font-medium text-slate-200">{model.name}</span>
                    <span className="text-xs text-slate-400">F1: {model.f1_score?.toFixed(1)}%</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className="text-xs text-slate-500">{model.latency_ms}ms</span>
                    <span className="text-sm font-semibold text-cyan-400">{model.accuracy?.toFixed(1)}%</span>
                  </div>
                </div>
                <div className="relative w-full h-3 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="absolute h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-500"
                    style={{ width: `${model.accuracy}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>

          {/* Average Metrics */}
          <div className="mt-6 pt-4 border-t border-slate-800">
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center">
                <p className="text-xs text-slate-400 mb-1">Avg Accuracy</p>
                <p className="text-lg font-semibold text-emerald-400">
                  {(models.reduce((sum, m) => sum + (m.accuracy || 0), 0) / models.length).toFixed(1)}%
                </p>
              </div>
              <div className="text-center">
                <p className="text-xs text-slate-400 mb-1">Avg F1-Score</p>
                <p className="text-lg font-semibold text-cyan-400">
                  {(models.reduce((sum, m) => sum + (m.f1_score || 0), 0) / models.length).toFixed(1)}%
                </p>
              </div>
              <div className="text-center">
                <p className="text-xs text-slate-400 mb-1">Avg Latency</p>
                <p className="text-lg font-semibold text-orange-400">
                  {Math.round(models.reduce((sum, m) => sum + (m.latency_ms || 0), 0) / models.length)}ms
                </p>
              </div>
              <div className="text-center">
                <p className="text-xs text-slate-400 mb-1">Models Active</p>
                <p className="text-lg font-semibold text-emerald-400">
                  {models.length}/6
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelComparison;