import React from 'react';
import { LineChart, Line, AreaChart, Area, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';

const NetworkMetrics = ({ stats }) => {
  const attackDistribution = stats.attack_distribution?.map(item => ({
    name: item._id,
    value: item.count
  })) || [];

  const COLORS = ['#ef4444', '#f97316', '#f59e0b', '#eab308', '#84cc16', '#22c55e', '#10b981', '#14b8a6'];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6" data-testid="network-metrics">
      {/* Total Alerts */}
      <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-slate-700 transition-all duration-300 hover:scale-105">
        <CardHeader className="pb-3">
          <CardDescription className="text-slate-400">Total Alerts</CardDescription>
          <CardTitle className="text-4xl text-slate-100 transition-all duration-300" style={{ fontFamily: 'Space Grotesk, sans-serif' }} data-testid="total-alerts">
            {stats.total_alerts?.toLocaleString() || 0}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-cyan-500 rounded-full animate-pulse"></div>
            <p className="text-xs text-slate-400">Monitoring active</p>
          </div>
        </CardContent>
      </Card>

      {/* Attack Count */}
      <Card className="bg-gradient-to-br from-red-950/30 to-slate-900 border-red-900/50 transition-all duration-300 hover:scale-105">
        <CardHeader className="pb-3">
          <CardDescription className="text-slate-400">Threats Detected</CardDescription>
          <CardTitle className="text-4xl text-red-400 transition-all duration-300" style={{ fontFamily: 'Space Grotesk, sans-serif' }} data-testid="total-attacks">
            {stats.total_attacks?.toLocaleString() || 0}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-red-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <p className="text-xs text-slate-400">High priority</p>
          </div>
        </CardContent>
      </Card>

      {/* Normal Traffic */}
      <Card className="bg-gradient-to-br from-emerald-950/30 to-slate-900 border-emerald-900/50">
        <CardHeader className="pb-3">
          <CardDescription className="text-slate-400">Normal Traffic</CardDescription>
          <CardTitle className="text-4xl text-emerald-400" style={{ fontFamily: 'Space Grotesk, sans-serif' }} data-testid="total-normal">
            {stats.total_normal?.toLocaleString() || 0}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <p className="text-xs text-slate-400">Legitimate flows</p>
          </div>
        </CardContent>
      </Card>

      {/* Attack Rate */}
      <Card className="bg-gradient-to-br from-orange-950/30 to-slate-900 border-orange-900/50">
        <CardHeader className="pb-3">
          <CardDescription className="text-slate-400">Attack Rate</CardDescription>
          <CardTitle className="text-4xl text-orange-400" style={{ fontFamily: 'Space Grotesk, sans-serif' }} data-testid="attack-rate">
            {stats.attack_rate?.toFixed(1) || 0}%
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="w-full bg-slate-800 rounded-full h-2 mt-2">
            <div
              className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${stats.attack_rate || 0}%` }}
            ></div>
          </div>
        </CardContent>
      </Card>

      {/* Attack Distribution Chart */}
      {attackDistribution.length > 0 && (
        <Card className="col-span-full bg-slate-900/50 border-slate-800">
          <CardHeader>
            <CardTitle className="text-slate-100" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Attack Type Distribution</CardTitle>
            <CardDescription className="text-slate-400">Breakdown of detected threats</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={attackDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {attackDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                  labelStyle={{ color: '#f1f5f9' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default NetworkMetrics;