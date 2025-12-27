# **CyberFlux – AI-Powered Intrusion Detection and Monitoring System**

<div align="center">
  <img src="https://img.shields.io/badge/ML--Based-Yes-cyan" />
  <img src="https://img.shields.io/badge/Models-6-blue" />
  <img src="https://img.shields.io/badge/Ensemble-Learning-green" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-red" />
  <img src="https://img.shields.io/badge/React-Frontend-blue" />
</div>

---

## Overview

**CyberFlux** is a **hybrid Machine Learning and Deep Learning–based Intrusion Detection and Monitoring System** designed to detect malicious network activities using an **ensemble of multiple models**.

The system analyzes network traffic data, detects known and unknown attacks, and provides **real-time monitoring through a web-based dashboard**. CyberFlux is developed as an **academic and research-oriented prototype**, focusing on intelligent detection, robustness, and system integration.

---

##  Key Features

 **Hybrid ML + DL Approach**

* LSTM
* CNN
* Autoencoder
* Random Forest
* XGBoost
* GAN-based synthetic data generation

 **Ensemble Decision Engine**

* Weighted aggregation of model predictions
* Improved accuracy and reduced false positives

 **Benchmark Dataset Training**

* NSL-KDD (primary dataset)
* KDDCup99 (legacy reference dataset)

 **Real-Time Monitoring**

* FastAPI backend
* WebSocket-based alert streaming

 **Modern Dashboard**

* React-based SOC dashboard
* Live threat monitoring and visual analytics

 **Dual Mode Operation**

* ML inference mode
* Simulation fallback mode for demonstration

---

##  Architecture Overview

CyberFlux follows a modular, layered architecture consisting of backend services, machine learning modules, database storage, and a frontend interface.

---

### 🔹 Backend Architecture (FastAPI + Python)

```
/app/backend/
├── server.py              # FastAPI server with REST & WebSocket APIs
├── ml/
│   ├── datasets/          # Dataset download and preprocessing
│   ├── models/            # ML & DL model definitions
│   ├── ensemble.py        # Ensemble decision engine
│   ├── training_pipeline.py
│   └── synthetic_generator.py
└── requirements.txt
```

**Responsibilities:**

* Traffic preprocessing
* Model inference
* Ensemble decision making
* Alert generation
* Database interaction
* Real-time communication with frontend

---

###  Frontend Architecture (React)

```
/app/frontend/
├── src/
│   ├── pages/
│   │   └── Dashboard.jsx
│   ├── components/
│   │   ├── ThreatTable.jsx
│   │   ├── NetworkMetrics.jsx
│   │   ├── ModelComparison.jsx
│   │   └── SystemStatus.jsx
│   └── App.js
└── package.json
```

**Responsibilities:**

* Real-time alert visualization
* Attack distribution charts
* Model status display
* User interaction

---

##  Quick Start (Academic Execution)

### 1️ Model Training

```bash
cd /app/backend
python ml/training_pipeline.py
```

### **Training Pipeline**

1. Downloads benchmark datasets
2. Preprocesses and normalizes traffic data
3. Trains ML and DL models
4. Saves trained models locally

>  Training was executed in a controlled cloud environment due to hardware constraints.
> Local execution is possible with adequate system resources.

---

##  Machine Learning Models Used

### **1. LSTM**

* Captures temporal patterns in network flows
* Effective for sequential attack behavior

### **2. CNN**

* Learns spatial feature relationships
* Useful for structured traffic features

### **3. Autoencoder**

* Unsupervised anomaly detection
* Flags deviations from learned normal traffic

### **4. GAN (Synthetic Data Generator)**

* Generates synthetic attack-like samples
* Improves robustness and dataset diversity
* Used for **data augmentation only**

### **5. Random Forest**

* Fast and stable baseline classifier
* Handles tabular network features effectively

### **6. XGBoost**

* High-accuracy gradient boosting model
* Strong performance on labeled attack data

---

##  Datasets Used

### **Primary Dataset**

* **NSL-KDD**

  * Standard benchmark IDS dataset
  * 41 network traffic features
  * Binary and multi-class labels

### **Legacy Reference Dataset**

* **KDDCup99**

  * Used for comparative and reference purposes

> No live network traffic was captured in this project.

---

##  Explainability 

CyberFlux includes **basic interpretability mechanisms**, without advanced post-hoc explainers.

✔ Model confidence scores
✔ Autoencoder reconstruction error
✔ Rule-based explanation logic

> Advanced explainability techniques (e.g., SHAP) are considered **future enhancements**.

---

##  Dashboard Capabilities

###  Network Metrics

* Total alerts
* Attack vs normal traffic
* Attack rate

###  Threat Monitoring

* Live alert table
* Attack type, severity, confidence
* Source and destination details

###  Model Insights

* Model availability status
* Performance comparison
* Ensemble decision visualization

---

##  Testing & Validation

* Unit testing of individual modules
* End-to-end system testing
* Validation using benchmark datasets
* Dry-run testing of mitigation logic

---

##  Future Enhancements (Not Implemented)

> The following are **planned improvements**, not part of the academic implementation:

* Live packet capture integration
* Cloud-scale deployment
* Advanced explainability techniques
* Automated enforcement of mitigation actions
* Continuous model retraining

---

##  Tech Stack

### **Backend**

* FastAPI
* Python
* TensorFlow / Keras
* Scikit-learn
* XGBoost
* MongoDB
* WebSockets

### **Frontend**

* React
* Tailwind CSS
* Recharts
* Axios

---

##  Academic Disclaimer

This project is developed **strictly for academic and research purposes**.
All experiments were conducted using public benchmark datasets and simulated traffic.

---

##  License

Academic use only.
Ensure compliance with dataset licenses for any extended usage.

---

##  Credits

**Datasets:**

* NSL-KDD
* KDDCup99

**Libraries:**

* TensorFlow
* Scikit-learn
* XGBoost

---


