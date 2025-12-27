#!/usr/bin/env python3
"""Fast GAN training with reduced epochs"""

import sys
import os
from pathlib import Path
import logging

backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import joblib
import numpy as np
from ml.models.gan_model import GANModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("/app/backend/ml/datasets/processed")

def train_gan_fast():
    """Train GAN with fewer epochs for faster completion"""
    
    logger.info("="*60)
    logger.info("FAST GAN TRAINING (20 epochs)")
    logger.info("="*60)
    
    try:
        # Load preprocessed data
        data_path = PROCESSED_DIR / "unified_dataset.pkl"
        if not data_path.exists():
            logger.error(f"Preprocessed data not found at {data_path}")
            return False
        
        data = joblib.load(data_path)
        X = data['X'].values
        y = data['y'].values
        
        # Use attack samples for GAN training
        X_attack = X[y == 1]
        logger.info(f"Using {len(X_attack)} attack samples for GAN training")
        
        # Train model with reduced epochs
        gan = GANModel(n_features=X.shape[1])
        logger.info("Training GAN (this will take ~3-5 minutes)...")
        gan.train(X_attack, epochs=20, batch_size=128)  # Reduced from 100 to 20
        
        # Test discriminator
        logger.info("Testing GAN discriminator...")
        normal_samples = X[y == 0][:1000]
        attack_samples = X[y == 1][:1000]
        
        normal_scores = gan.predict(normal_samples)
        attack_scores = gan.predict(attack_samples)
        
        logger.info(f"Normal traffic avg score: {np.mean(normal_scores):.4f}")
        logger.info(f"Attack traffic avg score: {np.mean(attack_scores):.4f}")
        
        # Save model
        gan.save_model()
        
        logger.info("="*60)
        logger.info("✓ GAN TRAINING COMPLETED")
        logger.info("="*60)
        logger.info("Models saved:")
        logger.info("  - /app/backend/ml/models/trained/gan_generator.h5")
        logger.info("  - /app/backend/ml/models/trained/gan_discriminator.h5")
        logger.info("\nRestart backend: sudo supervisorctl restart backend")
        
        return True
        
    except Exception as e:
        logger.error(f"GAN training failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = train_gan_fast()
    sys.exit(0 if success else 1)
