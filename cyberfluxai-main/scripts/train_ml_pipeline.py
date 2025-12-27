#!/usr/bin/env python3
"""
Cyberflux ML Training Pipeline
Trains all 6 ML models on NSL-KDD and KDDCup99 datasets
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

# Set environment to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from ml.datasets.downloader import download_all_datasets
from ml.datasets.preprocessor import DatasetPreprocessor
from ml.models.lstm_model import train_lstm_model
from ml.models.cnn_model import train_cnn_model
from ml.models.autoencoder_model import train_autoencoder_model
from ml.models.gan_model import train_gan_model
from ml.models.random_forest_model import train_random_forest_model
from ml.models.xgboost_model import train_xgboost_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/ml_training.log')
    ]
)
logger = logging.getLogger(__name__)


def print_banner(text):
    """Print a styled banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def main():
    """Main training pipeline"""
    start_time = datetime.now()
    
    print_banner("🚀 CYBERFLUX ML TRAINING PIPELINE")
    logger.info("Starting ML training pipeline...")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Download datasets
        print_banner("[1/8] 📥 Downloading Datasets")
        logger.info("Downloading NSL-KDD and KDDCup99 datasets...")
        try:
            download_all_datasets()
            logger.info("✓ Dataset download completed")
        except Exception as e:
            logger.warning(f"Dataset download warning: {e}")
            logger.info("Continuing with existing datasets...")
        
        # Step 2: Preprocess datasets
        print_banner("[2/8] 🔧 Preprocessing Datasets")
        logger.info("Creating unified dataset from NSL-KDD and KDDCup99...")
        preprocessor = DatasetPreprocessor()
        X, y = preprocessor.create_unified_dataset()
        preprocessor.save_preprocessor()
        logger.info(f"✓ Preprocessing completed: {len(X)} samples with {X.shape[1]} features")
        
        # Step 3: Train Random Forest (fastest, good baseline)
        print_banner("[3/8] 🌲 Training Random Forest")
        logger.info("Training Random Forest classifier...")
        train_random_forest_model()
        logger.info("✓ Random Forest training completed")
        
        # Step 4: Train XGBoost
        print_banner("[4/8] 🚀 Training XGBoost")
        logger.info("Training XGBoost classifier...")
        train_xgboost_model()
        logger.info("✓ XGBoost training completed")
        
        # Step 5: Train LSTM
        print_banner("[5/8] 🧠 Training LSTM")
        logger.info("Training LSTM deep learning model...")
        train_lstm_model()
        logger.info("✓ LSTM training completed")
        
        # Step 6: Train CNN
        print_banner("[6/8] 🖼️ Training CNN")
        logger.info("Training CNN deep learning model...")
        train_cnn_model()
        logger.info("✓ CNN training completed")
        
        # Step 7: Train Autoencoder
        print_banner("[7/8] 🔄 Training Autoencoder")
        logger.info("Training Autoencoder anomaly detector...")
        train_autoencoder_model()
        logger.info("✓ Autoencoder training completed")
        
        # Step 8: Train GAN (fast mode)
        print_banner("[8/8] 🎭 Training GAN")
        logger.info("Training GAN discriminator (fast mode - 20 epochs)...")
        # Import and call fast GAN training
        import joblib
        import numpy as np
        from ml.models.gan_model import GANModel
        
        PROCESSED_DIR = Path("/app/backend/ml/datasets/processed")
        data = joblib.load(PROCESSED_DIR / "unified_dataset.pkl")
        X = data['X'].values
        y = data['y'].values
        X_attack = X[y == 1]
        
        gan = GANModel(n_features=X.shape[1])
        gan.train(X_attack, epochs=20, batch_size=128)  # Fast mode
        gan.save_model()
        logger.info("✓ GAN training completed")
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print_banner("✅ TRAINING PIPELINE COMPLETED")
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration}")
        logger.info("\n🎉 All 6 models have been trained and saved!")
        logger.info("📁 Models saved in: /app/backend/ml/models/trained/")
        logger.info("🚀 Restart the backend to load the new models")
        
        print("\n" + "="*70)
        print("  NEXT STEPS:")
        print("  1. Restart backend: sudo supervisorctl restart backend")
        print("  2. Refresh dashboard to see models loaded")
        print("  3. Toggle ML Mode ON to use trained models")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        print_banner("❌ TRAINING FAILED")
        print(f"Error: {e}")
        print("Check /app/ml_training.log for details")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
