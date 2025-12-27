#!/usr/bin/env python3
"""Quick GAN training"""

import sys
import os
from pathlib import Path

backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ml.models.gan_model import train_gan_model
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

print("Training GAN model...")
try:
    train_gan_model()
    print("✓ GAN training completed")
except Exception as e:
    print(f"✗ GAN training failed: {e}")
