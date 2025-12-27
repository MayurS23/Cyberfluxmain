import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = Path("/app/backend/ml/models/trained")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("/app/backend/ml/datasets/processed")


class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        self.feature_importances = None
        
    def train(self, X_train, y_train):
        """Train Random Forest model"""
        logger.info(f"Training Random Forest on {len(X_train)} samples...")
        
        self.model.fit(X_train, y_train)
        self.feature_importances = self.model.feature_importances_
        
        logger.info("Random Forest training completed")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_model(self, filename="random_forest_model.pkl"):
        """Save trained model"""
        model_path = MODELS_DIR / filename
        joblib.dump(self.model, model_path)
        
        # Save feature importances
        if self.feature_importances is not None:
            importance_path = MODELS_DIR / "rf_feature_importances.pkl"
            joblib.dump(self.feature_importances, importance_path)
        
        logger.info(f"Random Forest model saved to {model_path}")
    
    def load_model(self, filename="random_forest_model.pkl"):
        """Load trained model"""
        model_path = MODELS_DIR / filename
        self.model = joblib.load(model_path)
        
        # Load feature importances
        importance_path = MODELS_DIR / "rf_feature_importances.pkl"
        if importance_path.exists():
            self.feature_importances = joblib.load(importance_path)
        
        logger.info(f"Random Forest model loaded from {model_path}")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)


def train_random_forest_model():
    """Main training function"""
    logging.basicConfig(level=logging.INFO)
    
    # Load preprocessed data
    data_path = PROCESSED_DIR / "unified_dataset.pkl"
    if not data_path.exists():
        logger.error(f"Preprocessed data not found at {data_path}")
        return None
    
    data = joblib.load(data_path)
    X = data['X'].values
    y = data['y'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    rf = RandomForestModel(n_estimators=100)
    rf.train(X_train, y_train)
    
    # Evaluate
    metrics = rf.evaluate(X_test, y_test)
    
    # Save model
    rf.save_model()
    
    return rf


if __name__ == "__main__":
    train_random_forest_model()