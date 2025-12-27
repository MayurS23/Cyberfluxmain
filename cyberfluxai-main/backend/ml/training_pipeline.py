#!/usr/bin/env python3
"""Complete ML training pipeline for Cyberflux IDS"""

import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.datasets.downloader import download_all_datasets
from ml.datasets.preprocessor import DatasetPreprocessor
from ml.models.lstm_model import train_lstm_model
from ml.models.cnn_model import train_cnn_model
from ml.models.autoencoder_model import train_autoencoder_model
from ml.models.gan_model import train_gan_model
from ml.models.random_forest_model import train_random_forest_model
from ml.models.xgboost_model import train_xgboost_model

logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*70)
    logger.info("CYBERFLUX ML TRAINING PIPELINE")
    logger.info("="*70)
    
    # Step 1: Download datasets
    logger.info("\n[1/7] Downloading datasets...")
    try:
        download_all_datasets()
        logger.info("✓ Dataset download completed")
    except Exception as e:
        logger.error(f"Dataset download failed: {e}")
    
    # Step 2: Preprocess datasets
    logger.info("\n[2/7] Preprocessing datasets...")
    try:
        preprocessor = DatasetPreprocessor()
        X, y = preprocessor.create_unified_dataset()
        preprocessor.save_preprocessor()
        logger.info("✓ Preprocessing completed")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return
    
    # Step 3: Train LSTM
    logger.info("\n[3/7] Training LSTM model...")
    try:
        train_lstm_model()
        logger.info("✓ LSTM training completed")
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
    
    # Step 4: Train CNN
    logger.info("\n[4/7] Training CNN model...")
    try:
        train_cnn_model()
        logger.info("✓ CNN training completed")
    except Exception as e:
        logger.error(f"CNN training failed: {e}")
    
    # Step 5: Train Autoencoder
    logger.info("\n[5/7] Training Autoencoder model...")
    try:
        train_autoencoder_model()
        logger.info("✓ Autoencoder training completed")
    except Exception as e:
        logger.error(f"Autoencoder training failed: {e}")
    
    # Step 6: Train GAN
    logger.info("\n[6/7] Training GAN model...")
    try:
        train_gan_model()
        logger.info("✓ GAN training completed")
    except Exception as e:
        logger.error(f"GAN training failed: {e}")
    
    # Step 7: Train Random Forest
    logger.info("\n[7/7] Training Random Forest model...")
    try:
        train_random_forest_model()
        logger.info("✓ Random Forest training completed")
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")
    
    # Step 8: Train XGBoost
    logger.info("\n[8/8] Training XGBoost model...")
    try:
        train_xgboost_model()
        logger.info("✓ XGBoost training completed")
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info("="*70)
    logger.info("\nAll models have been trained and saved.")
    logger.info("You can now start the FastAPI server to run inference.")


if __name__ == "__main__":
    main()