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


class LSTMModel:
    def __init__(self, sequence_length=10, n_features=41):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build LSTM model architecture"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.n_features)),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        logger.info("LSTM model built")
        return model
    
    def prepare_sequences(self, X, y):
        """Convert data into sequences for LSTM"""
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X_train, y_train, epochs=20, batch_size=128, validation_split=0.2):
        """Train LSTM model"""
        logger.info("Preparing sequences for LSTM...")
        X_seq, y_seq = self.prepare_sequences(X_train, y_train)
        
        logger.info(f"Training LSTM on {len(X_seq)} sequences...")
        
        if self.model is None:
            self.build_model()
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        logger.info("LSTM training completed")
        return self.history
    
    def save_model(self, filename="lstm_model.h5"):
        """Save trained model"""
        model_path = MODELS_DIR / filename
        self.model.save(model_path)
        logger.info(f"LSTM model saved to {model_path}")
        
        # Save training history
        history_path = MODELS_DIR / "lstm_history.pkl"
        joblib.dump(self.history.history, history_path)
    
    def load_model(self, filename="lstm_model.h5"):
        """Load trained model"""
        model_path = MODELS_DIR / filename
        self.model = keras.models.load_model(model_path)
        logger.info(f"LSTM model loaded from {model_path}")
    
    def predict(self, X):
        """Make predictions"""
        if len(X) < self.sequence_length:
            # Pad if needed
            padding = np.zeros((self.sequence_length - len(X), self.n_features))
            X = np.vstack([padding, X])
        
        X_seq, _ = self.prepare_sequences(X, np.zeros(len(X)))
        if len(X_seq) == 0:
            return np.array([0.5])  # Default prediction
        
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions


def train_lstm_model():
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
    lstm = LSTMModel(sequence_length=10, n_features=X.shape[1])
    lstm.train(X_train, y_train, epochs=20)
    
    # Evaluate
    X_test_seq, y_test_seq = lstm.prepare_sequences(X_test, y_test)
    test_loss, test_acc, test_precision, test_recall = lstm.model.evaluate(
        X_test_seq, y_test_seq, verbose=0
    )
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    
    # Save model
    lstm.save_model()
    
    return lstm


if __name__ == "__main__":
    train_lstm_model()