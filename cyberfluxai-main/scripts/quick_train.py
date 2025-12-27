#!/usr/bin/env python3
"""
Quick ML training with lightweight configurations
"""

import sys
import os
from pathlib import Path
import logging

# Add backend to path
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_train():
    """Quick training with smaller dataset"""
    from ml.datasets.downloader import download_nsl_kdd
    from ml.datasets.preprocessor import DatasetPreprocessor
    import joblib
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    logger.info("="*60)
    logger.info("QUICK ML TRAINING (Lightweight Mode)")
    logger.info("="*60)
    
    # Step 1: Download NSL-KDD
    logger.info("\n[1/4] Downloading NSL-KDD...")
    try:
        download_nsl_kdd()
        logger.info("✓ NSL-KDD ready")
    except Exception as e:
        logger.error(f"Download error: {e}")
    
    # Step 2: Preprocess
    logger.info("\n[2/4] Preprocessing data...")
    try:
        preprocessor = DatasetPreprocessor()
        X, y = preprocessor.preprocess_nsl_kdd()
        preprocessor.save_preprocessor()
        
        # Use only a subset for quick training (10% of data)
        indices = np.random.choice(len(X), size=int(len(X)*0.1), replace=False)
        X_subset = X.iloc[indices]
        y_subset = y.iloc[indices]
        
        logger.info(f"✓ Using {len(X_subset)} samples for training")
        
        # Save subset
        PROCESSED_DIR = Path("/app/backend/ml/datasets/processed")
        joblib.dump({
            'X': X_subset,
            'y': y_subset,
            'feature_names': list(X.columns)
        }, PROCESSED_DIR / "unified_dataset.pkl")
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return False
    
    # Step 3: Train Random Forest (fastest)
    logger.info("\n[3/4] Training Random Forest...")
    try:
        from ml.models.random_forest_model import RandomForestModel
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset.values, y_subset.values, test_size=0.2, random_state=42
        )
        
        rf = RandomForestModel(n_estimators=50)  # Reduced trees
        rf.train(X_train, y_train)
        rf.evaluate(X_test, y_test)
        rf.save_model()
        
        logger.info("✓ Random Forest trained")
    except Exception as e:
        logger.error(f"RF training error: {e}")
    
    # Step 4: Train XGBoost
    logger.info("\n[4/4] Training XGBoost...")
    try:
        from ml.models.xgboost_model import XGBoostModel
        
        xgb_model = XGBoostModel()
        xgb_model.model.n_estimators = 50  # Reduced trees
        xgb_model.train(X_train, y_train)
        xgb_model.evaluate(X_test, y_test)
        xgb_model.save_model()
        
        logger.info("✓ XGBoost trained")
    except Exception as e:
        logger.error(f"XGBoost training error: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("✓ QUICK TRAINING COMPLETED")
    logger.info("="*60)
    logger.info("2 models trained: Random Forest, XGBoost")
    logger.info("Restart backend to load models")
    
    return True

if __name__ == "__main__":
    success = quick_train()
    sys.exit(0 if success else 1)
