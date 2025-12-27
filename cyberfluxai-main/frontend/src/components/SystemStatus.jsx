import React from 'react';
import { Badge } from './ui/badge';

const SystemStatus = ({ status }) => {
  if (!status) {
    return (
      <div className="flex items-center space-x-4" data-testid="system-status">
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-slate-500 rounded-full"></div>
          <span className="text-sm text-slate-500">Loading...</span>
        </div>
      </div>
    );
  }

  const isDbConnected = status.database_status === 'connected';
  
  return (
    <div className="flex items-center space-x-4" data-testid="system-status">
      {/* Database Status */}
      <div className="flex items-center space-x-2">
        <div 
          className={`w-2 h-2 rounded-full ${
            isDbConnected 
              ? 'bg-emerald-500 animate-pulse' 
              : 'bg-red-500'
          }`}
          title={isDbConnected ? 'Database Connected' : 'Database Disconnected'}
        ></div>
        <span className={`text-sm ${isDbConnected ? 'text-slate-300' : 'text-red-400'}`}>
          Database
        </span>
      </div>

      {/* Models Loaded */}
      <div className="flex items-center space-x-2">
        <Badge 
          className={`border ${
            status.models_loaded > 0 
              ? 'bg-emerald-900/30 border-emerald-700/50 text-emerald-400' 
              : 'bg-slate-800 border-slate-700 text-slate-400'
          }`}
          data-testid="models-loaded-badge"
          title={`${status.models_loaded} ML model${status.models_loaded !== 1 ? 's' : ''} loaded`}
        >
          {status.models_loaded} Model{status.models_loaded !== 1 ? 's' : ''}
        </Badge>
      </div>

      {/* Active Connections */}
      <div className="flex items-center space-x-2">
        <Badge 
          className={`border ${
            status.active_connections > 0 
              ? 'bg-cyan-900/30 border-cyan-700/50 text-cyan-400' 
              : 'bg-slate-800 border-slate-700 text-slate-400'
          }`}
          data-testid="connections-badge"
          title={`${status.active_connections} active WebSocket connection${status.active_connections !== 1 ? 's' : ''}`}
        >
          {status.active_connections} Connection{status.active_connections !== 1 ? 's' : ''}
        </Badge>
      </div>
    </div>
  );
};

export default SystemStatus;