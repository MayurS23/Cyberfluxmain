# Cyberflux - AI-Powered Intrusion Detection System

<div align="center">
  <img src="https://img.shields.io/badge/AI-Powered-cyan" />
  <img src="https://img.shields.io/badge/Models-6-blue" />
  <img src="https://img.shields.io/badge/Real--Time-Monitoring-green" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-red" />
  <img src="https://img.shields.io/badge/React-Frontend-blue" />
</div>

## Overview

Cyberflux is a complete end-to-end ML-based Intrusion Detection System that uses multiple deep learning and machine learning models to detect network threats in real-time.

### Key Features

✅ **6 ML Models**: LSTM, CNN, Autoencoder, GAN, Random Forest, XGBoost  
✅ **Multi-Dataset Training**: CICIDS2017, UNSW-NB15, NSL-KDD, KDDCup99, CTU-13  
✅ **Ensemble Decision Engine**: Weighted voting system  
✅ **Real-Time Detection**: WebSocket-based live threat monitoring  
✅ **Explainable AI**: SHAP values and reconstruction error analysis    
✅ **Modern Dashboard**: React-based SOC dashboard with visualizations  
✅ **Dual Mode**: ML Mode + Simulation Fallback  

---

## Architecture

### Backend (FastAPI + Python)
```
/app/backend/
├── server.py              # FastAPI server with WebSocket
├── ml/
│   ├── datasets/          # Dataset download & preprocessing
│   ├── models/            # All 6 model implementations
│   ├── ensemble.py        # Ensemble decision engine
│   ├── explainer.py       # SHAP & XAI module
│   ├── training_pipeline.py
│   └── synthetic_generator.py
└── requirements.txt
```

### Frontend (React + Tailwind)
```
/app/frontend/
├── src/
│   ├── pages/
│   │   └── Dashboard.jsx   # Main dashboard
│   ├── components/
│   │   ├── ThreatTable.jsx
│   │   ├── NetworkMetrics.jsx
│   │   ├── ModelComparison.jsx
│   │  
│   │   └── SystemStatus.jsx
│   └── App.js
└── package.json
```

---

## Quick Start

### 1. Install Dependencies

Backend dependencies are already installed. Frontend is ready.

### 2. Train ML Models (IMPORTANT)

To use ML mode, you need to train all 6 models:

```bash
cd /app/backend
python ml/training_pipeline.py
```

**Training Process:**
1. Downloads datasets (NSL-KDD, KDDCup99)
2. Preprocesses and unifies data
3. Trains 6 models sequentially
4. Saves trained models to `/app/backend/ml/models/trained/`


### 3. Start Services

Services are already running via supervisor:

```bash
# Check status
sudo supervisorctl status

# Restart if needed
sudo supervisorctl restart backend frontend
```

### 4. Access Dashboard

Open your browser to the frontend URL (provided by Emergent).

---

## ML Models

### 1. LSTM (Sequential Anomaly Detection)
- **Architecture**: 2 LSTM layers (128, 64 units) + Dense
- **Purpose**: Detects temporal patterns in network flows
- **Input**: Sequences of 10 time steps

### 2. CNN (Feature Map Classification)
- **Architecture**: Conv2D layers + MaxPooling + Dense
- **Purpose**: Spatial feature extraction from flow data
- **Input**: 2D feature maps

### 3. Autoencoder (Unsupervised Anomaly Detection)
- **Architecture**: Encoder (128→64→32→16) + Decoder (symmetric)
- **Purpose**: Learns normal traffic patterns, flags anomalies
- **Training**: Only on benign traffic

### 4. GAN (Generator + Discriminator)
- **Generator**: Creates synthetic attack samples
- **Discriminator**: Classifies real vs fake (anomaly detection)
- **Purpose**: Data augmentation + semi-supervised learning

### 5. Random Forest
- **Type**: Ensemble classifier
- **Purpose**: Baseline model with feature importance
- **Trees**: 100 estimators

### 6. XGBoost
- **Type**: Gradient boosting classifier
- **Purpose**: High-accuracy attack classification
- **Integration**: SHAP explainability

---

## API Endpoints

### Core Endpoints

#### `GET /api/status`
System status including ML mode, loaded models, database status.

#### `POST /api/predict`
Make predictions on network flow features.
```json
{
  "features": [0.1, 0.2, ...] // 41 features
}
```

#### `GET /api/alerts`
Retrieve recent alerts (real-time threats).

#### `GET /api/alerts/stats`
Get statistics: total alerts, attacks, distribution.

#### `POST /api/copilot`
LLM Security CoPilot - AI-powered threat analysis.
```json
{
  "question": "What are the most critical threats?",
  "context": {...}
}
```

#### `GET /api/explain/{alert_id}`
Get detailed explanation for a specific alert (SHAP, recommendations).

#### `POST /api/mode`
Toggle between ML mode and simulation.
```json
{
  "ml_mode": true
}
```

#### `GET /api/models`
List all available models and their status.

### WebSocket

#### `WS /ws`
Real-time threat updates broadcast to all connected clients.

---

## Datasets

### Supported Datasets

1. **NSL-KDD** (Automated download)
   - Improved version of KDD Cup 99
   - 41 features
   - Binary & multiclass labels

2. **KDDCup99** (Automated download)
   - Classic network intrusion dataset
   - 10% subset (~500K records)

### Feature Engineering

**Unified Feature Schema (41 features):**
- Flow duration, protocol, service
- Source/destination bytes
- Error rates
- Host statistics
- Connection features

---


### Capabilities
- Threat explanation and analysis
- Mitigation strategy recommendations
- Model output interpretation
- Security best practices
- Incident response guidance



---

## Dashboard Features

### 1. Network Metrics
- Total alerts, attacks, normal traffic
- Attack rate percentage
- Real-time statistics
- Attack type distribution (pie chart)

### 2. Threat Table
- Live threat monitoring
- Severity, confidence, attack type
- Source/destination IPs
- Detailed alert inspection
- Mitigation recommendations

### 3. Model Comparison
- Model status (loaded/inactive)
- Performance metrics (accuracy, precision, recall)
- Bar chart comparison
- Radar chart multi-metric analysis
- Ensemble decision visualizer


## Performance Tuning

### GPU Acceleration
```python
# Check GPU availability
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Batch Processing
Increase batch sizes for faster training:
```python
model.train(X_train, y_train, batch_size=256)
```

### Model Pruning
Disable unused models to reduce inference time.

---

## Troubleshooting

### Issue: Models not loading
**Solution**: Train models first using `python ml/training_pipeline.py`

### Issue: WebSocket not connecting
**Solution**: Check if backend is running on correct port

### Issue: Low detection accuracy
**Solution**: 
1. Train on full datasets (not samples)
2. Increase training epochs
3. Tune hyperparameters

### Issue: Out of memory
**Solution**:
1. Reduce batch size
2. Use data generators
3. Train models separately

---

## Tech Stack

**Backend:**
- FastAPI (REST API)
- TensorFlow/Keras (Deep Learning)
- Scikit-learn (ML)
- XGBoost (Gradient Boosting)
- SHAP (Explainability)
- Motor (Async MongoDB)
- WebSockets (Real-time)
- Emergent Integrations (LLM)

**Frontend:**
- React 19
- Tailwind CSS
- Shadcn/UI
- Recharts (Visualizations)
- Socket.io (WebSocket client)
- Axios (HTTP client)

**Database:**
- MongoDB (Alerts, events, logs)

---

## Production Deployment

### Optimization Checklist
- [ ] Train all models with full datasets
- [ ] Enable GPU acceleration
- [ ] Set up model versioning
- [ ] Configure logging and monitoring
- [ ] Implement rate limiting
- [ ] Add authentication
- [ ] Set up backup schedules
- [ ] Load test WebSocket connections

---

## Contributing

This is a complete ML-based IDS system. For enhancements:
1. Add new model architectures
2. Integrate additional datasets
3. Improve explainability visualizations
4. Optimize inference speed

---

## License

Built for academic/research purposes. For production use, ensure compliance with dataset licenses.

---

## Credits

**Datasets:**
- NSL-KDD: University of New Brunswick
- CICIDS2017: Canadian Institute for Cybersecurity
- UNSW-NB15: UNSW Canberra
- CTU-13: Czech Technical University

**Models:**
- TensorFlow, Keras
- Scikit-learn, XGBoost
- SHAP



---

## Contact

For issues or questions about this implementation, refer to the system logs or use the built-in LLM CoPilot for assistance.

---

