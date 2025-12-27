import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogDescription } from './ui/dialog';
import { Slider } from './ui/slider';
import { toast } from 'sonner';
import { AlertTriangle, Shield, Play, Pause, RotateCcw, Settings } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const MitigationControl = () => {
  const [config, setConfig] = useState(null);
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);

  useEffect(() => {
    fetchConfig();
    fetchStats();
    fetchHistory();

    // Refresh stats periodically
    const interval = setInterval(() => {
      fetchStats();
      fetchHistory();
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const fetchConfig = async () => {
    try {
      const response = await axios.get(`${API}/mitigation/status`);
      setConfig(response.data);
    } catch (error) {
      console.error('Error fetching config:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API}/mitigation/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchHistory = async () => {
    try {
      const response = await axios.get(`${API}/mitigation/history?limit=20`);
      setHistory(response.data);
    } catch (error) {
      console.error('Error fetching history:', error);
    }
  };

  const updateConfig = async (updates) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/mitigation/config`, updates);
      setConfig(response.data.current_config);
      toast.success(response.data.message);
      fetchConfig();
    } catch (error) {
      toast.error('Failed to update configuration');
      console.error('Error updating config:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleDryRun = async (checked) => {
    await updateConfig({ dry_run_mode: checked });
  };

  const toggleAutoMitigation = async (checked) => {
    if (checked && !config?.dry_run_mode) {
      // Confirm enabling auto-mitigation in live mode
      if (!window.confirm('Enable auto-mitigation in LIVE mode? This will execute real actions.')) {
        return;
      }
    }
    await updateConfig({ auto_mitigation_enabled: checked });
  };

  const rollbackMitigation = async (mitigationId) => {
    try {
      const response = await axios.post(`${API}/mitigation/rollback/${mitigationId}`);
      if (response.data.success) {
        toast.success(response.data.message);
        fetchHistory();
        fetchStats();
      } else {
        toast.error(response.data.message);
      }
    } catch (error) {
      toast.error('Failed to rollback mitigation');
      console.error('Error rolling back:', error);
    }
  };

  const getStatusBadge = (status) => {
    const statusConfig = {
      executed: { color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50', label: 'Executed' },
      dry_run: { color: 'bg-blue-500/20 text-blue-400 border-blue-500/50', label: 'Dry Run' },
      rolled_back: { color: 'bg-orange-500/20 text-orange-400 border-orange-500/50', label: 'Rolled Back' },
      pending: { color: 'bg-slate-500/20 text-slate-400 border-slate-500/50', label: 'Pending' },
      failed: { color: 'bg-red-500/20 text-red-400 border-red-500/50', label: 'Failed' }
    };
    
    const config = statusConfig[status] || statusConfig.pending;
    return <Badge className={`${config.color} border`}>{config.label}</Badge>;
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  if (!config || !stats) {
    return <div className="text-slate-400">Loading mitigation controls...</div>;
  }

  return (
    <div className="space-y-6" data-testid="mitigation-control">
      {/* Control Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Status Card */}
        <Card className="bg-slate-900/50 border-slate-800">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg text-slate-100 flex items-center">
                <Shield className="w-5 h-5 mr-2 text-cyan-400" />
                Engine Status
              </CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Dry Run Mode */}
            <div className="flex items-center justify-between">
              <Label htmlFor="dry-run" className="text-sm text-slate-300">Dry Run Mode</Label>
              <Switch
                id="dry-run"
                checked={config.dry_run_mode}
                onCheckedChange={toggleDryRun}
                disabled={loading}
                data-testid="dry-run-toggle"
              />
            </div>

            {/* Auto Mitigation */}
            <div className="flex items-center justify-between">
              <Label htmlFor="auto-mitigation" className="text-sm text-slate-300">Auto-Mitigation</Label>
              <Switch
                id="auto-mitigation"
                checked={config.auto_mitigation_enabled}
                onCheckedChange={toggleAutoMitigation}
                disabled={loading}
                data-testid="auto-mitigation-toggle"
              />
            </div>

            {/* Status Indicator */}
            <div className="pt-3 border-t border-slate-800">
              <div className="flex items-center space-x-2">
                {config.auto_mitigation_enabled ? (
                  <>
                    <Play className="w-4 h-4 text-emerald-400" />
                    <span className="text-sm text-emerald-400 font-semibold">Active</span>
                  </>
                ) : (
                  <>
                    <Pause className="w-4 h-4 text-slate-400" />
                    <span className="text-sm text-slate-400">Inactive</span>
                  </>
                )}
              </div>
              <p className="text-xs text-slate-500 mt-1">
                {config.dry_run_mode ? 'Simulating actions only' : 'Executing real actions'}
              </p>
            </div>

            {/* Advanced Settings */}
            <Dialog open={configDialogOpen} onOpenChange={setConfigDialogOpen}>
              <DialogTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full mt-4 bg-slate-800/50 border-slate-700 text-slate-300 hover:bg-slate-700"
                  data-testid="advanced-settings-btn"
                >
                  <Settings className="w-4 h-4 mr-2" />
                  Advanced Settings
                </Button>
              </DialogTrigger>
              <DialogContent className="bg-slate-900 border-slate-800 text-slate-100 max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Mitigation Configuration</DialogTitle>
                  <DialogDescription className="text-slate-400">
                    Adjust thresholds and consensus requirements
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-6 py-4">
                  {/* Min Confidence */}
                  <div className="space-y-2">
                    <Label className="text-sm">Minimum Confidence: {config.min_confidence}%</Label>
                    <Slider
                      value={[config.min_confidence]}
                      onValueChange={(value) => updateConfig({ min_confidence: value[0] })}
                      min={50}
                      max={100}
                      step={5}
                      className="w-full"
                    />
                  </div>

                  {/* Model Consensus */}
                  <div className="space-y-2">
                    <Label className="text-sm">Model Consensus: {(config.min_model_consensus * 100).toFixed(0)}%</Label>
                    <Slider
                      value={[config.min_model_consensus * 100]}
                      onValueChange={(value) => updateConfig({ min_model_consensus: value[0] / 100 })}
                      min={40}
                      max={100}
                      step={5}
                      className="w-full"
                    />
                  </div>

                  {/* Severity Thresholds */}
                  <div className="space-y-3">
                    <Label className="text-sm font-semibold">Severity Thresholds</Label>
                    {Object.entries(config.severity_thresholds).map(([level, value]) => (
                      <div key={level} className="flex items-center justify-between">
                        <span className="text-sm text-slate-300 capitalize">{level}</span>
                        <span className="text-sm text-cyan-400 font-semibold">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </CardContent>
        </Card>

        {/* Statistics Cards */}
        <Card className="bg-gradient-to-br from-cyan-950/30 to-slate-900 border-cyan-900/50">
          <CardHeader className="pb-3">
            <CardDescription className="text-slate-400">Total Mitigations</CardDescription>
            <CardTitle className="text-4xl text-cyan-400" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
              {stats.total_mitigations}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Executed</span>
                <span className="text-emerald-400 font-semibold">{stats.executed}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Dry Run</span>
                <span className="text-blue-400 font-semibold">{stats.dry_run}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Rolled Back</span>
                <span className="text-orange-400 font-semibold">{stats.rolled_back}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-orange-950/30 to-slate-900 border-orange-900/50">
          <CardHeader className="pb-3">
            <CardDescription className="text-slate-400">Active Blocks</CardDescription>
            <CardTitle className="text-4xl text-orange-400" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
              {stats.blocked_ips + stats.rate_limited_ips}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Blocked IPs</span>
                <span className="text-red-400 font-semibold">{stats.blocked_ips}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Rate Limited</span>
                <span className="text-yellow-400 font-semibold">{stats.rate_limited_ips}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Mitigation History */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader>
          <CardTitle className="text-slate-100" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
            Mitigation History
          </CardTitle>
          <CardDescription className="text-slate-400">
            Recent mitigation actions and their status
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="border-slate-800">
                  <TableHead className="text-slate-300">Timestamp</TableHead>
                  <TableHead className="text-slate-300">Threat ID</TableHead>
                  <TableHead className="text-slate-300">Actions</TableHead>
                  <TableHead className="text-slate-300">Status</TableHead>
                  <TableHead className="text-slate-300">Rollback</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {history.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center py-8 text-slate-400">
                      No mitigation actions yet
                    </TableCell>
                  </TableRow>
                ) : (
                  history.map((item) => (
                    <TableRow key={item.mitigation_id} className="border-slate-800 hover:bg-slate-800/30">
                      <TableCell className="text-slate-300 text-sm">
                        {formatTimestamp(item.timestamp)}
                      </TableCell>
                      <TableCell className="text-slate-300 font-mono text-xs">
                        {item.threat_id.substring(0, 8)}...
                      </TableCell>
                      <TableCell className="text-slate-300 text-sm">
                        {item.actions_taken?.length || 0} actions
                      </TableCell>
                      <TableCell>
                        {getStatusBadge(item.status)}
                      </TableCell>
                      <TableCell>
                        {item.rollback_available && item.status !== 'rolled_back' && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => rollbackMitigation(item.mitigation_id)}
                            className="text-orange-400 hover:text-orange-300 hover:bg-orange-500/10"
                            data-testid={`rollback-btn-${item.mitigation_id}`}
                          >
                            <RotateCcw className="w-4 h-4" />
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Warning Banner */}
      {config.auto_mitigation_enabled && !config.dry_run_mode && (
        <div className="bg-orange-500/10 border border-orange-500/50 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-5 h-5 text-orange-400 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-orange-400">Auto-Mitigation Active (LIVE MODE)</p>
              <p className="text-xs text-slate-400 mt-1">
                The system is automatically executing mitigation actions. Enable Dry Run mode to test safely.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MitigationControl;
