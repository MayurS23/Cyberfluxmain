#!/bin/bash

# Cyberflux Model Training Script
# This script trains all 6 ML models for the IDS system

set -e  # Exit on error

echo "====================================="
echo "  CYBERFLUX MODEL TRAINING SCRIPT"
echo "====================================="
echo ""

cd /app/backend

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "[Step 1/8] Checking dependencies..."
python -c "import tensorflow, sklearn, xgboost, shap" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ All dependencies installed"
else
    echo "✗ Missing dependencies. Installing..."
    pip install -q -r requirements.txt
fi

echo ""
echo "[Step 2/8] Downloading datasets..."
python -c "from ml.datasets.downloader import download_all_datasets; download_all_datasets()"

echo ""
echo "[Step 3/8] Preprocessing datasets..."
python -c "
from ml.datasets.preprocessor import DatasetPreprocessor
import logging
logging.basicConfig(level=logging.INFO)
preprocessor = DatasetPreprocessor()
X, y = preprocessor.create_unified_dataset()
preprocessor.save_preprocessor()
print(f'Dataset ready: {len(X)} samples')
"

echo ""
echo "[Step 4/8] Training LSTM model..."
python ml/models/lstm_model.py

echo ""
echo "[Step 5/8] Training CNN model..."
python ml/models/cnn_model.py

echo ""
echo "[Step 6/8] Training Autoencoder model..."
python ml/models/autoencoder_model.py

echo ""
echo "[Step 7/8] Training GAN model..."
python ml/models/gan_model.py

echo ""
echo "[Step 8/8] Training Random Forest model..."
python ml/models/random_forest_model.py

echo ""
echo "[Step 9/9] Training XGBoost model..."
python ml/models/xgboost_model.py

echo ""
echo "====================================="
echo "  TRAINING COMPLETE!"
echo "====================================="
echo ""
echo "Trained models saved to: /app/backend/ml/models/trained/"
echo ""
echo "Next steps:"
echo "1. Restart backend: sudo supervisorctl restart backend"
echo "2. Toggle ML Mode in dashboard"
echo "3. Monitor real-time predictions"
echo ""
