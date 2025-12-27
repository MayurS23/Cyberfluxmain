import React, { useState } from 'react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ThreatTable = ({ alerts }) => {
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);

  const getSeverityColor = (severity) => {
    if (severity >= 80) return 'bg-red-500/20 text-red-400 border-red-500/50';
    if (severity >= 60) return 'bg-orange-500/20 text-orange-400 border-orange-500/50';
    if (severity >= 40) return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
    return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50';
  };

  const getAttackTypeColor = (attackType) => {
    if (attackType === 'Normal') return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50';
    return 'bg-red-500/20 text-red-400 border-red-500/50';
  };

  const fetchExplanation = async (alertId) => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/explain/${alertId}`);
      setExplanation(response.data);
    } catch (error) {
      console.error('Error fetching explanation:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAlertClick = (alert) => {
    setSelectedAlert(alert);
    fetchExplanation(alert.id);
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 backdrop-blur-sm overflow-hidden" data-testid="threat-table">
      <div className="p-6 border-b border-slate-800">
        <h2 className="text-xl font-semibold text-slate-100" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Real-Time Threat Monitor</h2>
        <p className="text-sm text-slate-400 mt-1">Live network traffic analysis</p>
      </div>

      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow className="border-slate-800 hover:bg-slate-800/50">
              <TableHead className="text-slate-300">Timestamp</TableHead>
              <TableHead className="text-slate-300">Source IP</TableHead>
              <TableHead className="text-slate-300">Destination IP</TableHead>
              <TableHead className="text-slate-300">Protocol</TableHead>
              <TableHead className="text-slate-300">Attack Type</TableHead>
              <TableHead className="text-slate-300">Severity</TableHead>
              <TableHead className="text-slate-300">Confidence</TableHead>
              <TableHead className="text-slate-300">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {alerts.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="text-center py-8 text-slate-400">
                  No threats detected yet. Monitoring in progress...
                </TableCell>
              </TableRow>
            ) : (
              alerts.map((alert) => (
                <TableRow key={alert.id} className="border-slate-800 hover:bg-slate-800/30" data-testid={`alert-row-${alert.id}`}>
                  <TableCell className="text-slate-300 text-sm">{formatTimestamp(alert.timestamp)}</TableCell>
                  <TableCell className="text-slate-300 font-mono text-sm">{alert.src_ip}</TableCell>
                  <TableCell className="text-slate-300 font-mono text-sm">{alert.dst_ip}</TableCell>
                  <TableCell className="text-slate-300 text-sm">{alert.protocol}</TableCell>
                  <TableCell>
                    <Badge className={`${getAttackTypeColor(alert.attack_type)} border`} data-testid="attack-type-badge">
                      {alert.attack_type}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge className={`${getSeverityColor(alert.severity)} border`} data-testid="severity-badge">
                      {alert.severity}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-slate-300 text-sm">{alert.confidence}%</TableCell>
                  <TableCell>
                    <Dialog>
                      <DialogTrigger asChild>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleAlertClick(alert)}
                          className="bg-slate-800/50 border-slate-700 text-slate-300 hover:bg-slate-700 hover:text-white"
                          data-testid={`view-details-btn-${alert.id}`}
                        >
                          Details
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="bg-slate-900 border-slate-800 text-slate-100 max-w-2xl" data-testid="alert-detail-dialog">
                        <DialogHeader>
                          <DialogTitle className="text-xl" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Threat Details</DialogTitle>
                        </DialogHeader>
                        {selectedAlert && (
                          <div className="space-y-4 mt-4">
                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <p className="text-sm text-slate-400">Attack Type</p>
                                <p className="text-lg font-semibold text-slate-100">{selectedAlert.attack_type}</p>
                              </div>
                              <div>
                                <p className="text-sm text-slate-400">Severity</p>
                                <p className="text-lg font-semibold text-slate-100">{selectedAlert.severity}/100</p>
                              </div>
                              <div>
                                <p className="text-sm text-slate-400">Source IP</p>
                                <p className="text-slate-100 font-mono">{selectedAlert.src_ip}</p>
                              </div>
                              <div>
                                <p className="text-sm text-slate-400">Destination IP</p>
                                <p className="text-slate-100 font-mono">{selectedAlert.dst_ip}</p>
                              </div>
                            </div>

                            {explanation && (
                              <div className="mt-6 space-y-4">
                                <h3 className="text-lg font-semibold text-slate-100">Mitigation Recommendations</h3>
                                <ul className="list-disc list-inside space-y-2 text-slate-300">
                                  {explanation.recommendations?.map((rec, idx) => (
                                    <li key={idx} className="text-sm">{rec}</li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {loading && <p className="text-slate-400">Loading explanation...</p>}
                          </div>
                        )}
                      </DialogContent>
                    </Dialog>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
};

export default ThreatTable;