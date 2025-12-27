import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = Path("/app/backend/ml/models/trained")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("/app/backend/ml/datasets/processed")


class AutoencoderModel:
    def __init__(self, n_features=41):
        self.n_features = n_features
        self.model = None
        self.history = None
        self.threshold = None
        
    def build_model(self):
        """Build Autoencoder model architecture"""
        # Encoder
        encoder_input = layers.Input(shape=(self.n_features,))
        encoded = layers.Dense(128, activation='relu')(encoder_input)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.Dense(16, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(32, activation='relu')(encoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dense(self.n_features, activation='sigmoid')(decoded)
        
        # Autoencoder
        autoencoder = keras.Model(encoder_input, decoded)
        autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = autoencoder
        logger.info("Autoencoder model built")
        return autoencoder
    
    def train(self, X_train, epochs=50, batch_size=256, validation_split=0.2):
        """Train Autoencoder on normal traffic only"""
        logger.info(f"Training Autoencoder on {len(X_train)} normal samples...")
        
        if self.model is None:
            self.build_model()
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        )
        
        # Train on normal data to reconstruct
        self.history = self.model.fit(
            X_train, X_train,  # Input = Output for autoencoder
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Calculate reconstruction error threshold
        reconstructions = self.model.predict(X_train, verbose=0)
        mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95th percentile
        
        logger.info(f"Autoencoder training completed. Threshold: {self.threshold:.6f}")
        return self.history
    
    def save_model(self, filename="autoencoder_model.h5"):
        """Save trained model"""
        model_path = MODELS_DIR / filename
        self.model.save(model_path)
        
        # Save threshold
        threshold_path = MODELS_DIR / "autoencoder_threshold.pkl"
        joblib.dump({'threshold': self.threshold}, threshold_path)
        
        logger.info(f"Autoencoder model saved to {model_path}")
        
        # Save training history
        history_path = MODELS_DIR / "autoencoder_history.pkl"
        joblib.dump(self.history.history, history_path)
    
    def load_model(self, filename="autoencoder_model.h5"):
        """Load trained model"""
        model_path = MODELS_DIR / filename
        self.model = keras.models.load_model(model_path)
        
        # Load threshold
        threshold_path = MODELS_DIR / "autoencoder_threshold.pkl"
        if threshold_path.exists():
            data = joblib.load(threshold_path)
            self.threshold = data['threshold']
        
        logger.info(f"Autoencoder model loaded from {model_path}")
    
    def predict(self, X):
        """Calculate anomaly scores based on reconstruction error"""
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        # Normalize scores to 0-1 range
        if self.threshold:
            anomaly_scores = np.clip(mse / (self.threshold * 2), 0, 1)
        else:
            anomaly_scores = mse / (np.max(mse) + 1e-10)
        
        return anomaly_scores


def train_autoencoder_model():
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
    
    # Use only normal traffic for training
    X_normal = X[y == 0]
    logger.info(f"Using {len(X_normal)} normal samples for training")
    
    # Train model
    autoencoder = AutoencoderModel(n_features=X.shape[1])
    autoencoder.train(X_normal, epochs=50)
    
    # Test on both normal and attack traffic
    X_attack = X[y == 1]
    normal_scores = autoencoder.predict(X_normal[:1000])
    attack_scores = autoencoder.predict(X_attack[:1000])
    
    logger.info(f"Normal traffic avg score: {np.mean(normal_scores):.4f}")
    logger.info(f"Attack traffic avg score: {np.mean(attack_scores):.4f}")
    
    # Save model
    autoencoder.save_model()
    
    return autoencoder


if __name__ == "__main__":
    train_autoencoder_model()