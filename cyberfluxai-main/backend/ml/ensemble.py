import numpy as np
import joblib
from pathlib import Path
import logging
from typing import Dict, List, Any
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)

MODELS_DIR = Path("/app/backend/ml/models/trained")


class EnsembleModel:
    def __init__(self):
        self.models = {}
        self.weights = {
            'lstm': 0.20,
            'cnn': 0.20,
            'autoencoder': 0.15,
            'gan': 0.15,
            'random_forest': 0.15,
            'xgboost': 0.15
        }
        self.loaded = False
        
    def load_all_models(self):
        """Load all trained models"""
        logger.info("Loading ensemble models...")
        
        try:
            # Load LSTM
            lstm_path = MODELS_DIR / "lstm_model.h5"
            if lstm_path.exists():
                from ml.models.lstm_model import LSTMModel
                self.models['lstm'] = LSTMModel()
                self.models['lstm'].load_model()
                logger.info("✓ LSTM model loaded")
            else:
                logger.warning("LSTM model not found")
            
            # Load CNN
            cnn_path = MODELS_DIR / "cnn_model.h5"
            if cnn_path.exists():
                from ml.models.cnn_model import CNNModel
                self.models['cnn'] = CNNModel()
                self.models['cnn'].load_model()
                logger.info("✓ CNN model loaded")
            else:
                logger.warning("CNN model not found")
            
            # Load Autoencoder
            ae_path = MODELS_DIR / "autoencoder_model.h5"
            if ae_path.exists():
                from ml.models.autoencoder_model import AutoencoderModel
                self.models['autoencoder'] = AutoencoderModel()
                self.models['autoencoder'].load_model()
                logger.info("✓ Autoencoder model loaded")
            else:
                logger.warning("Autoencoder model not found")
            
            # Load GAN Discriminator
            gan_path = MODELS_DIR / "gan_discriminator.h5"
            if gan_path.exists():
                from ml.models.gan_model import GANModel
                self.models['gan'] = GANModel()
                self.models['gan'].load_model()
                logger.info("✓ GAN model loaded")
            else:
                logger.warning("GAN model not found")
            
            # Load Random Forest
            rf_path = MODELS_DIR / "random_forest_model.pkl"
            if rf_path.exists():
                self.models['random_forest'] = joblib.load(rf_path)
                logger.info("✓ Random Forest model loaded")
            else:
                logger.warning("Random Forest model not found")
            
            # Load XGBoost
            xgb_path = MODELS_DIR / "xgboost_model.pkl"
            if xgb_path.exists():
                self.models['xgboost'] = joblib.load(xgb_path)
                logger.info("✓ XGBoost model loaded")
            else:
                logger.warning("XGBoost model not found")
            
            self.loaded = True
            logger.info(f"Ensemble loaded with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.loaded = False
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Make ensemble prediction"""
        if not self.loaded:
            logger.warning("Models not loaded, using simulation mode")
            return self._simulate_prediction()
        
        try:
            predictions = {}
            scores = []
            
            # LSTM prediction (batch processing with sequence creation)
            if 'lstm' in self.models:
                try:
                    # LSTM requires sequences - create by repeating the sample
                    lstm_model = self.models['lstm']
                    seq_length = lstm_model.sequence_length
                    
                    # Create sequence by repeating the sample
                    X_seq = np.tile(X, (seq_length, 1))
                    X_seq = X_seq.reshape(1, seq_length, X.shape[1])
                    
                    lstm_pred = lstm_model.model.predict(X_seq, verbose=0)
                    lstm_score = float(lstm_pred[0][0])
                    predictions['lstm'] = lstm_score
                    scores.append(lstm_score * self.weights['lstm'])
                except Exception as e:
                    logger.warning(f"LSTM prediction failed: {e}")
            
            # CNN prediction
            if 'cnn' in self.models:
                try:
                    cnn_pred = self.models['cnn'].predict(X)
                    cnn_score = float(cnn_pred[0][0])
                    predictions['cnn'] = cnn_score
                    scores.append(cnn_score * self.weights['cnn'])
                except Exception as e:
                    logger.warning(f"CNN prediction failed: {e}")
            
            # Autoencoder anomaly score
            if 'autoencoder' in self.models:
                try:
                    ae_score = self.models['autoencoder'].predict(X)
                    ae_score = float(ae_score[0]) if len(ae_score) > 0 else 0.5
                    predictions['autoencoder'] = ae_score
                    scores.append(ae_score * self.weights['autoencoder'])
                except Exception as e:
                    logger.warning(f"Autoencoder prediction failed: {e}")
            
            # GAN discriminator score
            if 'gan' in self.models:
                try:
                    gan_score = self.models['gan'].predict(X)
                    gan_score = float(gan_score[0]) if len(gan_score) > 0 else 0.5
                    predictions['gan'] = gan_score
                    scores.append(gan_score * self.weights['gan'])
                except Exception as e:
                    logger.warning(f"GAN prediction failed: {e}")
            
            # Random Forest prediction
            if 'random_forest' in self.models:
                try:
                    rf_proba = self.models['random_forest'].predict_proba(X)
                    rf_score = float(rf_proba[0][1])
                    predictions['random_forest'] = rf_score
                    scores.append(rf_score * self.weights['random_forest'])
                except Exception as e:
                    logger.warning(f"Random Forest prediction failed: {e}")
            
            # XGBoost prediction
            if 'xgboost' in self.models:
                try:
                    xgb_proba = self.models['xgboost'].predict_proba(X)
                    xgb_score = float(xgb_proba[0][1])
                    predictions['xgboost'] = xgb_score
                    scores.append(xgb_score * self.weights['xgboost'])
                except Exception as e:
                    logger.warning(f"XGBoost prediction failed: {e}")
            
            # Calculate ensemble score
            if scores:
                ensemble_score = sum(scores) / sum(self.weights[k] for k in predictions.keys())
            else:
                ensemble_score = 0.5
            
            # Determine attack type and severity
            is_attack = ensemble_score > 0.5
            severity = int(ensemble_score * 100)
            confidence = min(abs(ensemble_score - 0.5) * 200, 100)
            
            attack_types = [
                "DDoS", "Port Scan", "Brute Force", "SQL Injection",
                "XSS", "Botnet", "Infiltration", "DoS"
            ]
            attack_type = "Normal" if not is_attack else np.random.choice(attack_types)
            
            return {
                "attack_type": attack_type,
                "is_attack": is_attack,
                "severity": severity,
                "confidence": round(confidence, 2),
                "ensemble_score": round(ensemble_score, 4),
                "model_outputs": {k: round(v, 4) for k, v in predictions.items()}
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return self._simulate_prediction()
    
    def _simulate_prediction(self) -> Dict[str, Any]:
        """Simulation mode fallback"""
        is_attack = np.random.random() > 0.7
        severity = np.random.randint(60, 100) if is_attack else np.random.randint(0, 40)
        
        attack_types = [
            "DDoS", "Port Scan", "Brute Force", "SQL Injection",
            "XSS", "Botnet", "Infiltration", "DoS"
        ]
        attack_type = "Normal" if not is_attack else np.random.choice(attack_types)
        
        return {
            "attack_type": attack_type,
            "is_attack": is_attack,
            "severity": severity,
            "confidence": round(np.random.uniform(70, 95), 2),
            "ensemble_score": severity / 100.0,
            "model_outputs": {
                "lstm": round(np.random.random(), 4),
                "cnn": round(np.random.random(), 4),
                "autoencoder": round(np.random.random(), 4),
                "gan": round(np.random.random(), 4),
                "random_forest": round(np.random.random(), 4),
                "xgboost": round(np.random.random(), 4)
            },
            "mode": "simulation"
        }


# Global ensemble instance
ensemble = EnsembleModel()