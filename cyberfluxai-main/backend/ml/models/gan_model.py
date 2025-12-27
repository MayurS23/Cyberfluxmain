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


class GANModel:
    def __init__(self, n_features=41, latent_dim=100):
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.history = {'d_loss': [], 'g_loss': [], 'd_accuracy': []}
        
    def build_generator(self):
        """Build Generator model"""
        model = keras.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(self.n_features, activation='tanh')
        ])
        
        self.generator = model
        logger.info("Generator built")
        return model
    
    def build_discriminator(self):
        """Build Discriminator model"""
        model = keras.Sequential([
            layers.Input(shape=(self.n_features,)),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.discriminator = model
        logger.info("Discriminator built")
        return model
    
    def build_gan(self):
        """Build combined GAN model"""
        if self.generator is None:
            self.build_generator()
        if self.discriminator is None:
            self.build_discriminator()
        
        # Freeze discriminator when training generator
        self.discriminator.trainable = False
        
        gan_input = layers.Input(shape=(self.latent_dim,))
        generated_data = self.generator(gan_input)
        gan_output = self.discriminator(generated_data)
        
        gan = keras.Model(gan_input, gan_output)
        gan.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )
        
        self.gan = gan
        logger.info("GAN model built")
        return gan
    
    def train(self, X_train, epochs=100, batch_size=128):
        """Train GAN model"""
        logger.info(f"Training GAN on {len(X_train)} samples...")
        
        if self.gan is None:
            self.build_gan()
        
        # Normalize data to [-1, 1] for tanh activation
        X_train_norm = 2 * (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-10) - 1
        
        batch_count = len(X_train_norm) // batch_size
        
        for epoch in range(epochs):
            d_losses = []
            g_losses = []
            d_accs = []
            
            for batch_idx in range(batch_count):
                # Train Discriminator
                # Real samples
                idx = np.random.randint(0, len(X_train_norm), batch_size)
                real_data = X_train_norm[idx]
                real_labels = np.ones((batch_size, 1)) * 0.9  # Label smoothing
                
                # Fake samples
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_data = self.generator.predict(noise, verbose=0)
                fake_labels = np.zeros((batch_size, 1))
                
                # Train discriminator
                self.discriminator.trainable = True
                d_loss_real, d_acc_real = self.discriminator.train_on_batch(real_data, real_labels)
                d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(fake_data, fake_labels)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                d_acc = 0.5 * (d_acc_real + d_acc_fake)
                
                # Train Generator
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                valid_labels = np.ones((batch_size, 1))
                
                self.discriminator.trainable = False
                g_loss = self.gan.train_on_batch(noise, valid_labels)
                
                d_losses.append(d_loss)
                g_losses.append(g_loss)
                d_accs.append(d_acc)
            
            # Log progress
            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses)
            avg_d_acc = np.mean(d_accs)
            
            self.history['d_loss'].append(avg_d_loss)
            self.history['g_loss'].append(avg_g_loss)
            self.history['d_accuracy'].append(avg_d_acc)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}, "
                    f"D Acc: {avg_d_acc:.4f}"
                )
        
        logger.info("GAN training completed")
        return self.history
    
    def save_model(self):
        """Save trained models"""
        generator_path = MODELS_DIR / "gan_generator.h5"
        discriminator_path = MODELS_DIR / "gan_discriminator.h5"
        
        self.generator.save(generator_path)
        self.discriminator.save(discriminator_path)
        
        logger.info(f"GAN models saved")
        
        # Save training history
        history_path = MODELS_DIR / "gan_history.pkl"
        joblib.dump(self.history, history_path)
    
    def load_model(self):
        """Load trained models"""
        generator_path = MODELS_DIR / "gan_generator.h5"
        discriminator_path = MODELS_DIR / "gan_discriminator.h5"
        
        self.generator = keras.models.load_model(generator_path)
        self.discriminator = keras.models.load_model(discriminator_path)
        
        logger.info("GAN models loaded")
    
    def generate_samples(self, n_samples=100):
        """Generate synthetic attack samples"""
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        generated = self.generator.predict(noise, verbose=0)
        return generated
    
    def predict(self, X):
        """Use discriminator to classify real vs fake (anomaly detection)"""
        predictions = self.discriminator.predict(X, verbose=0)
        # Higher probability = more real (normal)
        # Lower probability = more fake (anomalous)
        anomaly_scores = 1 - predictions.flatten()
        return anomaly_scores


def train_gan_model():
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
    
    # Use attack samples for GAN training
    X_attack = X[y == 1]
    logger.info(f"Using {len(X_attack)} attack samples for GAN training")
    
    # Train model
    gan = GANModel(n_features=X.shape[1])
    gan.train(X_attack, epochs=100, batch_size=128)
    
    # Test discriminator
    normal_samples = X[y == 0][:1000]
    attack_samples = X[y == 1][:1000]
    
    normal_scores = gan.predict(normal_samples)
    attack_scores = gan.predict(attack_samples)
    
    logger.info(f"Normal traffic avg score: {np.mean(normal_scores):.4f}")
    logger.info(f"Attack traffic avg score: {np.mean(attack_scores):.4f}")
    
    # Save model
    gan.save_model()
    
    return gan


if __name__ == "__main__":
    train_gan_model()