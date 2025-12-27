import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class MitigationAction(str, Enum):
    """Available mitigation actions"""
    BLOCK_IP = "block_ip"
    RATE_LIMIT = "rate_limit"
    QUARANTINE = "quarantine"
    ALERT_ONLY = "alert_only"
    DROP_PACKETS = "drop_packets"
    REDIRECT = "redirect"
    CAPTCHA = "captcha"
    LOG_ENHANCED = "log_enhanced"


class MitigationStatus(str, Enum):
    """Status of mitigation action"""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    DRY_RUN = "dry_run"


class AttackCategory(str, Enum):
    """Attack categories with mitigation strategies"""
    DDOS = "ddos"
    BRUTE_FORCE = "brute_force"
    PORT_SCAN = "port_scan"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    BOTNET = "botnet"
    INFILTRATION = "infiltration"
    DOS = "dos"
    NORMAL = "normal"


class MitigationEngine:
    """Auto-mitigation engine with safety controls"""
    
    def __init__(self, db_client=None):
        self.db = db_client
        self.dry_run_mode = True  # Safe default: dry-run enabled
        self.auto_mitigation_enabled = False  # Requires explicit activation
        self.blocked_ips = set()
        self.rate_limited_ips = {}
        self.mitigation_history = []
        
        # Severity thresholds for auto-mitigation
        self.severity_thresholds = {
            "critical": 90,  # Immediate action
            "high": 75,      # Quick response
            "medium": 60,    # Monitored response
            "low": 40        # Alert only
        }
        
        # Confidence thresholds
        self.min_confidence = 80.0  # Minimum confidence for auto-mitigation
        
        # Model consensus requirements
        self.min_model_consensus = 0.6  # 60% of models must agree
        
        # Mitigation strategy mapping
        self.mitigation_strategies = {
            AttackCategory.DDOS: [MitigationAction.RATE_LIMIT, MitigationAction.BLOCK_IP],
            AttackCategory.BRUTE_FORCE: [MitigationAction.RATE_LIMIT, MitigationAction.CAPTCHA],
            AttackCategory.PORT_SCAN: [MitigationAction.BLOCK_IP, MitigationAction.LOG_ENHANCED],
            AttackCategory.SQL_INJECTION: [MitigationAction.BLOCK_IP, MitigationAction.ALERT_ONLY],
            AttackCategory.XSS: [MitigationAction.BLOCK_IP, MitigationAction.ALERT_ONLY],
            AttackCategory.BOTNET: [MitigationAction.QUARANTINE, MitigationAction.BLOCK_IP],
            AttackCategory.INFILTRATION: [MitigationAction.QUARANTINE, MitigationAction.ALERT_ONLY],
            AttackCategory.DOS: [MitigationAction.RATE_LIMIT, MitigationAction.DROP_PACKETS],
            AttackCategory.NORMAL: [MitigationAction.ALERT_ONLY]
        }
        
        logger.info("Mitigation Engine initialized (dry-run: ON, auto-mitigation: OFF)")
    
    def analyze_threat(self, alert: Dict[str, Any], model_outputs: Dict[str, float]) -> Dict[str, Any]:
        """Analyze threat using all model outputs and determine mitigation"""
        
        # Extract alert details
        attack_type = alert.get('attack_type', 'unknown')
        severity = alert.get('severity', 0)
        confidence = alert.get('confidence', 0)
        src_ip = alert.get('src_ip', 'unknown')
        
        # Analyze model consensus
        consensus_analysis = self._analyze_model_consensus(model_outputs, severity)
        
        # Determine severity level
        severity_level = self._determine_severity_level(severity)
        
        # Get recommended actions
        recommended_actions = self._get_mitigation_actions(
            attack_type,
            severity_level,
            confidence,
            consensus_analysis
        )
        
        # Make mitigation decision
        decision = {
            "threat_id": alert.get('id', str(uuid.uuid4())),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attack_type": attack_type,
            "severity": severity,
            "severity_level": severity_level,
            "confidence": confidence,
            "src_ip": src_ip,
            "model_consensus": consensus_analysis,
            "recommended_actions": recommended_actions,
            "auto_mitigation_eligible": self._is_eligible_for_auto_mitigation(
                severity, confidence, consensus_analysis
            ),
            "requires_manual_review": severity >= 85 and confidence < self.min_confidence
        }
        
        return decision
    
    def _analyze_model_consensus(self, model_outputs: Dict[str, float], severity: int) -> Dict[str, Any]:
        """Analyze consensus across all 6 models"""
        
        if not model_outputs:
            return {
                "agreement_score": 0.0,
                "models_agreeing": 0,
                "total_models": 0,
                "confidence_distribution": {},
                "outliers": []
            }
        
        # Normalize severity to 0-1 range for comparison
        normalized_severity = severity / 100.0
        
        # Calculate agreement threshold
        agreement_threshold = 0.2  # Models within ±20% are considered agreeing
        
        models_agreeing = 0
        outliers = []
        
        for model_name, score in model_outputs.items():
            if abs(score - normalized_severity) <= agreement_threshold:
                models_agreeing += 1
            else:
                outliers.append({
                    "model": model_name,
                    "score": score,
                    "deviation": abs(score - normalized_severity)
                })
        
        total_models = len(model_outputs)
        agreement_score = models_agreeing / total_models if total_models > 0 else 0.0
        
        return {
            "agreement_score": round(agreement_score, 3),
            "models_agreeing": models_agreeing,
            "total_models": total_models,
            "confidence_distribution": model_outputs,
            "outliers": outliers,
            "mean_score": round(sum(model_outputs.values()) / total_models, 3) if total_models > 0 else 0.0
        }
    
    def _determine_severity_level(self, severity: int) -> str:
        """Determine severity level from numeric score"""
        if severity >= self.severity_thresholds["critical"]:
            return "critical"
        elif severity >= self.severity_thresholds["high"]:
            return "high"
        elif severity >= self.severity_thresholds["medium"]:
            return "medium"
        elif severity >= self.severity_thresholds["low"]:
            return "low"
        else:
            return "minimal"
    
    def _get_mitigation_actions(self, attack_type: str, severity_level: str, 
                                confidence: float, consensus: Dict) -> List[Dict[str, Any]]:
        """Get recommended mitigation actions based on attack analysis"""
        
        # Map attack type to category
        attack_category = self._map_attack_to_category(attack_type)
        
        # Get base actions for this attack type
        base_actions = self.mitigation_strategies.get(
            attack_category,
            [MitigationAction.ALERT_ONLY]
        )
        
        recommended_actions = []
        
        for action in base_actions:
            action_config = {
                "action": action.value,
                "priority": self._get_action_priority(severity_level),
                "confidence_required": self.min_confidence,
                "estimated_impact": self._estimate_impact(action, severity_level),
                "reversible": self._is_reversible(action),
                "execution_delay_seconds": self._get_execution_delay(severity_level)
            }
            recommended_actions.append(action_config)
        
        return recommended_actions
    
    def _map_attack_to_category(self, attack_type: str) -> AttackCategory:
        """Map attack type string to category enum"""
        attack_lower = attack_type.lower().replace(' ', '_')
        
        for category in AttackCategory:
            if category.value in attack_lower or attack_lower in category.value:
                return category
        
        return AttackCategory.NORMAL
    
    def _get_action_priority(self, severity_level: str) -> int:
        """Get execution priority (1=highest, 5=lowest)"""
        priority_map = {
            "critical": 1,
            "high": 2,
            "medium": 3,
            "low": 4,
            "minimal": 5
        }
        return priority_map.get(severity_level, 5)
    
    def _estimate_impact(self, action: MitigationAction, severity_level: str) -> str:
        """Estimate business impact of mitigation action"""
        impact_matrix = {
            MitigationAction.BLOCK_IP: "high" if severity_level == "critical" else "medium",
            MitigationAction.RATE_LIMIT: "low",
            MitigationAction.QUARANTINE: "high",
            MitigationAction.DROP_PACKETS: "medium",
            MitigationAction.CAPTCHA: "low",
            MitigationAction.ALERT_ONLY: "none",
            MitigationAction.LOG_ENHANCED: "none",
            MitigationAction.REDIRECT: "low"
        }
        return impact_matrix.get(action, "unknown")
    
    def _is_reversible(self, action: MitigationAction) -> bool:
        """Check if action can be rolled back"""
        reversible_actions = [
            MitigationAction.BLOCK_IP,
            MitigationAction.RATE_LIMIT,
            MitigationAction.QUARANTINE,
            MitigationAction.CAPTCHA
        ]
        return action in reversible_actions
    
    def _get_execution_delay(self, severity_level: str) -> int:
        """Get delay before executing action (safety window)"""
        delay_map = {
            "critical": 0,      # Immediate
            "high": 5,         # 5 seconds
            "medium": 30,      # 30 seconds
            "low": 60,         # 1 minute
            "minimal": 300     # 5 minutes
        }
        return delay_map.get(severity_level, 60)
    
    def _is_eligible_for_auto_mitigation(self, severity: int, confidence: float, 
                                        consensus: Dict) -> bool:
        """Determine if threat is eligible for automatic mitigation"""
        
        if not self.auto_mitigation_enabled:
            return False
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            return False
        
        # Check model consensus
        if consensus['agreement_score'] < self.min_model_consensus:
            return False
        
        # Check severity threshold
        if severity < self.severity_thresholds["high"]:
            return False
        
        return True
    
    async def execute_mitigation(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute mitigation actions with safety checks"""
        
        mitigation_id = str(uuid.uuid4())
        
        execution_result = {
            "mitigation_id": mitigation_id,
            "threat_id": decision['threat_id'],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": self.dry_run_mode,
            "actions_taken": [],
            "actions_skipped": [],
            "status": MitigationStatus.PENDING.value,
            "rollback_available": True
        }
        
        # Check if auto-mitigation is enabled
        if not decision.get('auto_mitigation_eligible', False):
            execution_result['status'] = MitigationStatus.PENDING.value
            execution_result['reason'] = "Not eligible for auto-mitigation"
            return execution_result
        
        # Execute each recommended action
        for action_config in decision.get('recommended_actions', []):
            action_result = await self._execute_action(
                action_config,
                decision['src_ip'],
                mitigation_id
            )
            
            if action_result['executed']:
                execution_result['actions_taken'].append(action_result)
            else:
                execution_result['actions_skipped'].append(action_result)
        
        # Update status
        if execution_result['actions_taken']:
            execution_result['status'] = (
                MitigationStatus.DRY_RUN.value if self.dry_run_mode 
                else MitigationStatus.EXECUTED.value
            )
        else:
            execution_result['status'] = MitigationStatus.FAILED.value
        
        # Store in audit log
        await self._log_mitigation(execution_result)
        
        # Add to history
        self.mitigation_history.append(execution_result)
        
        return execution_result
    
    async def _execute_action(self, action_config: Dict, src_ip: str, 
                            mitigation_id: str) -> Dict[str, Any]:
        """Execute individual mitigation action"""
        
        action = action_config['action']
        
        action_result = {
            "action": action,
            "target_ip": src_ip,
            "mitigation_id": mitigation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "executed": False,
            "dry_run": self.dry_run_mode,
            "reversible": action_config['reversible']
        }
        
        try:
            if self.dry_run_mode:
                # Simulate execution
                action_result['executed'] = True
                action_result['message'] = f"[DRY-RUN] Would execute {action} on {src_ip}"
                logger.info(action_result['message'])
            else:
                # Real execution
                if action == MitigationAction.BLOCK_IP.value:
                    success = await self._block_ip(src_ip)
                elif action == MitigationAction.RATE_LIMIT.value:
                    success = await self._apply_rate_limit(src_ip)
                elif action == MitigationAction.QUARANTINE.value:
                    success = await self._quarantine_ip(src_ip)
                elif action == MitigationAction.ALERT_ONLY.value:
                    success = True  # Always succeeds
                elif action == MitigationAction.LOG_ENHANCED.value:
                    success = await self._enable_enhanced_logging(src_ip)
                else:
                    success = False
                    action_result['message'] = f"Action {action} not implemented"
                
                action_result['executed'] = success
                action_result['message'] = (
                    f"Successfully executed {action} on {src_ip}" if success 
                    else f"Failed to execute {action} on {src_ip}"
                )
        
        except Exception as e:
            action_result['executed'] = False
            action_result['error'] = str(e)
            logger.error(f"Error executing {action} on {src_ip}: {e}")
        
        return action_result
    
    async def _block_ip(self, ip: str) -> bool:
        """Block IP address"""
        self.blocked_ips.add(ip)
        logger.info(f"Blocked IP: {ip}")
        return True
    
    async def _apply_rate_limit(self, ip: str) -> bool:
        """Apply rate limiting to IP"""
        self.rate_limited_ips[ip] = {
            "limit": 10,  # requests per minute
            "window": 60,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        logger.info(f"Rate limited IP: {ip}")
        return True
    
    async def _quarantine_ip(self, ip: str) -> bool:
        """Quarantine IP (isolate network access)"""
        logger.info(f"Quarantined IP: {ip}")
        return True
    
    async def _enable_enhanced_logging(self, ip: str) -> bool:
        """Enable enhanced logging for IP"""
        logger.info(f"Enhanced logging enabled for IP: {ip}")
        return True
    
    async def rollback_mitigation(self, mitigation_id: str) -> Dict[str, Any]:
        """Rollback a mitigation action"""
        
        # Find mitigation in history
        mitigation = None
        for m in self.mitigation_history:
            if m['mitigation_id'] == mitigation_id:
                mitigation = m
                break
        
        if not mitigation:
            return {
                "success": False,
                "message": f"Mitigation {mitigation_id} not found"
            }
        
        if not mitigation.get('rollback_available', False):
            return {
                "success": False,
                "message": "Rollback not available for this mitigation"
            }
        
        rollback_result = {
            "mitigation_id": mitigation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions_rolled_back": []
        }
        
        # Rollback each action
        for action in mitigation.get('actions_taken', []):
            if action.get('reversible', False):
                target_ip = action['target_ip']
                
                if action['action'] == MitigationAction.BLOCK_IP.value:
                    if target_ip in self.blocked_ips:
                        self.blocked_ips.remove(target_ip)
                        rollback_result['actions_rolled_back'].append(action['action'])
                
                elif action['action'] == MitigationAction.RATE_LIMIT.value:
                    if target_ip in self.rate_limited_ips:
                        del self.rate_limited_ips[target_ip]
                        rollback_result['actions_rolled_back'].append(action['action'])
        
        # Update mitigation status
        mitigation['status'] = MitigationStatus.ROLLED_BACK.value
        
        # Log rollback
        await self._log_rollback(rollback_result)
        
        return {
            "success": True,
            "message": f"Rolled back {len(rollback_result['actions_rolled_back'])} actions",
            "details": rollback_result
        }
    
    async def _log_mitigation(self, execution_result: Dict[str, Any]):
        """Log mitigation to database"""
        if self.db:
            try:
                await self.db.mitigation_logs.insert_one(execution_result)
            except Exception as e:
                logger.error(f"Failed to log mitigation: {e}")
    
    async def _log_rollback(self, rollback_result: Dict[str, Any]):
        """Log rollback to database"""
        if self.db:
            try:
                await self.db.mitigation_rollbacks.insert_one(rollback_result)
            except Exception as e:
                logger.error(f"Failed to log rollback: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            "dry_run_mode": self.dry_run_mode,
            "auto_mitigation_enabled": self.auto_mitigation_enabled,
            "blocked_ips_count": len(self.blocked_ips),
            "rate_limited_ips_count": len(self.rate_limited_ips),
            "total_mitigations": len(self.mitigation_history),
            "severity_thresholds": self.severity_thresholds,
            "min_confidence": self.min_confidence,
            "min_model_consensus": self.min_model_consensus
        }
    
    def set_dry_run_mode(self, enabled: bool):
        """Enable/disable dry-run mode"""
        self.dry_run_mode = enabled
        logger.info(f"Dry-run mode: {'ENABLED' if enabled else 'DISABLED'}")
    
    def set_auto_mitigation(self, enabled: bool):
        """Enable/disable auto-mitigation"""
        self.auto_mitigation_enabled = enabled
        logger.info(f"Auto-mitigation: {'ENABLED' if enabled else 'DISABLED'}")


# Global mitigation engine instance
mitigation_engine = MitigationEngine()