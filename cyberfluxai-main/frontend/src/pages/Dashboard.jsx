import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { toast } from 'sonner';
import { Shield, Activity, BarChart3, Zap, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';
import ThreatTable from '../components/ThreatTable';
import NetworkMetrics from '../components/NetworkMetrics';
import ModelComparison from '../components/ModelComparison';
import SystemStatus from '../components/SystemStatus';
import LLMCopilot from '../components/LLMCopilot';
import MitigationControl from '../components/MitigationControl';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const [alerts, setAlerts] = useState([]);
  const [stats, setStats] = useState({ 
    total_alerts: 0, 
    total_attacks: 0, 
    total_normal: 0, 
    attack_rate: 0,
    attack_distribution: []
  });
  const [models, setModels] = useState([]);
  const [status, setStatus] = useState(null);
  const [mlMode, setMlMode] = useState(true);
  const [ws, setWs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [simulatingTraffic, setSimulatingTraffic] = useState(false);
  
  // Use refs to prevent excessive API calls
  const fetchingRef = useRef(false);
  const lastFetchTime = useRef({
    status: 0,
    stats: 0,
    alerts: 0,
    models: 0
  });

  // Memoized fetch functions with throttling
  const fetchAlerts = useCallback(async () => {
    const now = Date.now();
    if (fetchingRef.current || now - lastFetchTime.current.alerts < 2000) return;
    
    fetchingRef.current = true;
    try {
      const response = await axios.get(`${API}/alerts?limit=50`);
      setAlerts(response.data);
      lastFetchTime.current.alerts = now;
    } catch (error) {
      console.error('Error fetching alerts:', error);
    } finally {
      fetchingRef.current = false;
    }
  }, []);

  const fetchStats = useCallback(async () => {
    const now = Date.now();
    if (now - lastFetchTime.current.stats < 3000) return;
    
    try {
      const response = await axios.get(`${API}/alerts/stats`);
      setStats(response.data);
      lastFetchTime.current.stats = now;
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  }, []);

  const fetchModels = useCallback(async () => {
    const now = Date.now();
    if (now - lastFetchTime.current.models < 10000) return; // Models don't change often
    
    try {
      const response = await axios.get(`${API}/models`);
      setModels(response.data);
      lastFetchTime.current.models = now;
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  }, []);

  const fetchStatus = useCallback(async () => {
    const now = Date.now();
    if (now - lastFetchTime.current.status < 4000) return; // Throttle to 4 seconds
    
    try {
      const response = await axios.get(`${API}/status`);
      setStatus(response.data);
      setMlMode(response.data.ml_mode);
      lastFetchTime.current.status = now;
    } catch (error) {
      console.error('Error fetching status:', error);
    }
  }, []);

  // Initialize WebSocket with better error handling
  useEffect(() => {
    const wsUrl = BACKEND_URL.replace(/^http/, 'ws') + '/api/ws';
    console.log('Connecting to WebSocket:', wsUrl);
    
    let websocket;
    let pingInterval;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;

    const connect = () => {
      try {
        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
          console.log('✓ WebSocket connected');
          toast.success('Real-time monitoring active', { duration: 2000 });
          reconnectAttempts = 0;
          
          fetchStatus();
          
          pingInterval = setInterval(() => {
            if (websocket.readyState === WebSocket.OPEN) {
              websocket.send('ping');
            }
          }, 30000);
        };

        websocket.onclose = (event) => {
          if (pingInterval) clearInterval(pingInterval);
          
          if (event.code !== 1000 && reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            console.log(`Reconnecting... (attempt ${reconnectAttempts})`);
            setTimeout(connect, 3000);
          }
        };

        websocket.onerror = (error) => {
          console.error('WebSocket error:', error);
        };

        websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'new_alert') {
              setAlerts((prev) => [data.data, ...prev].slice(0, 100));
              fetchStats();
            } else if (data.type === 'pong') {
              fetchStatus();
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        setWs(websocket);
      } catch (error) {
        console.error('Failed to create WebSocket:', error);
      }
    };

    connect();

    return () => {
      if (pingInterval) clearInterval(pingInterval);
      if (websocket) websocket.close(1000);
    };
  }, [fetchStats, fetchStatus]);

  // Initial data fetch
  useEffect(() => {
    const loadInitialData = async () => {
      setLoading(true);
      try {
        await Promise.all([
          fetchAlerts(),
          fetchStats(),
          fetchModels(),
          fetchStatus()
        ]);
      } finally {
        setLoading(false);
      }
    };

    loadInitialData();

    // Periodic refresh with longer intervals
    const interval = setInterval(() => {
      fetchStats();
      fetchStatus();
    }, 8000); // Increased from 5s to 8s

    return () => clearInterval(interval);
  }, [fetchAlerts, fetchStats, fetchModels, fetchStatus]);

  const handleSimulateTraffic = async () => {
    if (simulatingTraffic) return;
    
    setSimulatingTraffic(true);
    try {
      const response = await axios.post(`${API}/simulate-traffic?count=10`);
      toast.success(`Generated ${response.data.alerts_generated} alerts`, { duration: 2000 });
      
      setTimeout(() => {
        fetchAlerts();
        fetchStats();
      }, 500);
    } catch (error) {
      console.error('Error simulating traffic:', error);
      toast.error('Failed to simulate traffic');
    } finally {
      setSimulatingTraffic(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-400">Loading Cyberflux IDS...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Shield className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                Cyberflux IDS
              </h1>
              <p className="text-slate-400 text-sm">AI-Powered Threat Detection & Analysis</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {status && <SystemStatus status={status} />}
            
            <Button
              onClick={handleSimulateTraffic}
              disabled={simulatingTraffic}
              className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white"
            >
              {simulatingTraffic ? (
                <><div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>Simulating...</>
              ) : (
                <><Zap className="w-4 h-4 mr-2" />Simulate Traffic</>
              )}
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="threats" className="space-y-6">
        <TabsList className="bg-slate-900/50 border border-slate-800">
          <TabsTrigger value="threats" className="data-[state=active]:bg-cyan-600 data-[state=active]:text-white">
            <AlertTriangle className="w-4 h-4 mr-2" />
            Threats
          </TabsTrigger>
          <TabsTrigger value="analytics" className="data-[state=active]:bg-cyan-600 data-[state=active]:text-white">
            <BarChart3 className="w-4 h-4 mr-2" />
            Model Analysis
          </TabsTrigger>
          <TabsTrigger value="copilot" className="data-[state=active]:bg-cyan-600 data-[state=active]:text-white">
            <Activity className="w-4 h-4 mr-2" />
            Security CoPilot
          </TabsTrigger>
          <TabsTrigger value="mitigation" className="data-[state=active]:bg-cyan-600 data-[state=active]:text-white">
            <Shield className="w-4 h-4 mr-2" />
            Auto-Mitigation
          </TabsTrigger>
        </TabsList>

        {/* Threats Tab */}
        <TabsContent value="threats" className="mt-6" data-testid="threats-content">
          <NetworkMetrics stats={stats} />

          <ThreatTable alerts={alerts} />
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="mt-6">
          <ModelComparison models={models} />
        </TabsContent>

        {/* CoPilot Tab */}
        <TabsContent value="copilot" className="mt-6">
          <LLMCopilot alerts={alerts} />
        </TabsContent>

        {/* Mitigation Tab */}
        <TabsContent value="mitigation" className="mt-6">
          <MitigationControl />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Dashboard;