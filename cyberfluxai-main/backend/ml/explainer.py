import numpy as np
import shap
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

MODELS_DIR = Path("/app/backend/ml/models/trained")
PROCESSED_DIR = Path("/app/backend/ml/datasets/processed")


class ModelExplainer:
    def __init__(self):
        self.explainers = {}
        self.feature_names = []
        self._load_feature_names()
        
    def _load_feature_names(self):
        """Load feature names"""
        try:
            preprocessor_path = PROCESSED_DIR / "preprocessor.pkl"
            if preprocessor_path.exists():
                data = joblib.load(preprocessor_path)
                self.feature_names = data.get('feature_columns', [])
                logger.info(f"Loaded {len(self.feature_names)} feature names")
        except Exception as e:
            logger.warning(f"Could not load feature names: {e}")
            self.feature_names = [f"feature_{i}" for i in range(41)]
    
    def explain_xgboost(self, model, X: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP explanation for XGBoost"""
        try:
            if 'xgboost' not in self.explainers:
                self.explainers['xgboost'] = shap.TreeExplainer(model)
            
            shap_values = self.explainers['xgboost'].shap_values(X)
            
            # Get top features
            if len(shap_values.shape) > 1:
                feature_importance = np.abs(shap_values[0])
            else:
                feature_importance = np.abs(shap_values)
            
            top_indices = np.argsort(feature_importance)[-5:][::-1]
            
            top_features = []
            for idx in top_indices:
                feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                top_features.append({
                    "feature": feature_name,
                    "importance": float(feature_importance[idx]),
                    "value": float(X[0][idx])
                })
            
            return {
                "method": "SHAP",
                "model": "XGBoost",
                "top_features": top_features
            }
            
        except Exception as e:
            logger.error(f"XGBoost explanation error: {e}")
            return {"error": str(e)}
    
    def explain_random_forest(self, model, X: np.ndarray) -> Dict[str, Any]:
        """Generate explanation for Random Forest"""
        try:
            # Use feature importances
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-5:][::-1]
            
            top_features = []
            for idx in top_indices:
                feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                top_features.append({
                    "feature": feature_name,
                    "importance": float(importances[idx]),
                    "value": float(X[0][idx])
                })
            
            return {
                "method": "Feature Importance",
                "model": "Random Forest",
                "top_features": top_features
            }
            
        except Exception as e:
            logger.error(f"Random Forest explanation error: {e}")
            return {"error": str(e)}
    
    def explain_autoencoder(self, model, X: np.ndarray) -> Dict[str, Any]:
        """Generate explanation for Autoencoder"""
        try:
            reconstruction = model.model.predict(X, verbose=0)
            reconstruction_error = np.abs(X[0] - reconstruction[0])
            
            # Top features with highest reconstruction error
            top_indices = np.argsort(reconstruction_error)[-5:][::-1]
            
            top_features = []
            for idx in top_indices:
                feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                top_features.append({
                    "feature": feature_name,
                    "reconstruction_error": float(reconstruction_error[idx]),
                    "original_value": float(X[0][idx]),
                    "reconstructed_value": float(reconstruction[0][idx])
                })
            
            return {
                "method": "Reconstruction Error",
                "model": "Autoencoder",
                "top_features": top_features,
                "total_error": float(np.mean(reconstruction_error))
            }
            
        except Exception as e:
            logger.error(f"Autoencoder explanation error: {e}")
            return {"error": str(e)}
    
    def explain_prediction(self, models_dict: Dict, X: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive explanation"""
        explanations = {}
        
        if 'xgboost' in models_dict:
            explanations['xgboost'] = self.explain_xgboost(models_dict['xgboost'], X)
        
        if 'random_forest' in models_dict:
            explanations['random_forest'] = self.explain_random_forest(models_dict['random_forest'], X)
        
        if 'autoencoder' in models_dict:
            explanations['autoencoder'] = self.explain_autoencoder(models_dict['autoencoder'], X)
        
        return explanations


# Global explainer instance
explainer = ModelExplainer()