import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import logging
import math

logger = logging.getLogger(__name__)

MODELS_DIR = Path("/app/backend/ml/models/trained")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("/app/backend/ml/datasets/processed")


class CNNModel:
    def __init__(self, n_features=41):
        self.n_features = n_features
        # Calculate image dimensions
        self.img_height = int(math.sqrt(n_features))
        self.img_width = int(math.ceil(n_features / self.img_height))
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build CNN model architecture"""
        model = keras.Sequential([
            layers.Input(shape=(self.img_height, self.img_width, 1)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
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
        logger.info("CNN model built")
        return model
    
    def prepare_image_data(self, X):
        """Convert feature vectors to 2D images"""
        n_samples = X.shape[0]
        X_img = np.zeros((n_samples, self.img_height, self.img_width))
        
        for i in range(n_samples):
            # Pad features if needed
            features = X[i]
            if len(features) < self.img_height * self.img_width:
                features = np.pad(features, (0, self.img_height * self.img_width - len(features)))
            
            # Reshape to 2D
            X_img[i] = features[:self.img_height * self.img_width].reshape(
                self.img_height, self.img_width
            )
        
        # Add channel dimension
        X_img = X_img.reshape(n_samples, self.img_height, self.img_width, 1)
        return X_img
    
    def train(self, X_train, y_train, epochs=30, batch_size=128, validation_split=0.2):
        """Train CNN model"""
        logger.info("Preparing image data for CNN...")
        X_img = self.prepare_image_data(X_train)
        
        logger.info(f"Training CNN on {len(X_img)} samples...")
        logger.info(f"Image shape: {X_img.shape[1:]}")
        
        if self.model is None:
            self.build_model()
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_img, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        logger.info("CNN training completed")
        return self.history
    
    def save_model(self, filename="cnn_model.h5"):
        """Save trained model"""
        model_path = MODELS_DIR / filename
        self.model.save(model_path)
        logger.info(f"CNN model saved to {model_path}")
        
        # Save training history
        history_path = MODELS_DIR / "cnn_history.pkl"
        joblib.dump(self.history.history, history_path)
    
    def load_model(self, filename="cnn_model.h5"):
        """Load trained model"""
        model_path = MODELS_DIR / filename
        self.model = keras.models.load_model(model_path)
        logger.info(f"CNN model loaded from {model_path}")
    
    def predict(self, X):
        """Make predictions"""
        X_img = self.prepare_image_data(X)
        predictions = self.model.predict(X_img, verbose=0)
        return predictions


def train_cnn_model():
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
    cnn = CNNModel(n_features=X.shape[1])
    cnn.train(X_train, y_train, epochs=30)
    
    # Evaluate
    X_test_img = cnn.prepare_image_data(X_test)
    test_loss, test_acc, test_precision, test_recall = cnn.model.evaluate(
        X_test_img, y_test, verbose=0
    )
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    
    # Save model
    cnn.save_model()
    
    return cnn


if __name__ == "__main__":
    train_cnn_model()