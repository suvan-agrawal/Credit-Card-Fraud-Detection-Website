# 💳 Credit Card Fraud Detection System

<div align="center">

[![MLOps CI/CD Pipeline](https://github.com/YOUR_USERNAME/credit-card-fraud-detection/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/credit-card-fraud-detection/actions/workflows/ci-cd.yml)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**ML-powered fraud detection system with MLOps: CI/CD, Docker containerization, real-time monitoring, and explainable AI**

[Features](#-features) • [Architecture](#️-architecture) • [Dataset](#-dataset) • [Model](#-model-development) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [MLOps](#-mlops-features)

</div>

---

## 📖 Overview

Credit card fraud represents a significant threat to financial transactions and consumer trust in digital commerce. This project delivers a comprehensive, production-ready machine learning solution for detecting fraudulent credit card transactions — complete with a real-time web interface, MLOps pipeline, Docker containerization, and explainable AI.

### 🎯 Project Objectives

- Analyze patterns and trends in credit card fraud transactions
- Build and compare multiple machine learning algorithms on highly imbalanced data
- Handle class imbalance using advanced techniques (SMOTE)
- Deploy a production-ready web application for real-time predictions
- Provide actionable insights with risk assessment, explainability, and monitoring

---

## ✨ Features

### 🔍 Core Capabilities

- **Real-time Fraud Detection** — Instant transaction analysis with risk assessment (<200ms inference)
- **Multi-Model Comparison** — Logistic Regression, Decision Tree, Random Forest, and SVM evaluated
- **Advanced Preprocessing** — SMOTE for handling class imbalance without data leakage
- **Risk Categorization** — Low / Medium / High classification with confidence scores
- **Interactive Dashboard** — Professional UI with dynamic visualizations
- **RESTful API** — Easy integration with existing systems

### 💡 Smart Features

- Dynamic dropdowns populated from trained model encoders
- Fraud probability meter with visual indicators
- Time-based pattern recognition (night transactions, weekends)
- Merchant risk profiling
- Geographic fraud hotspot identification
- Responsive design for mobile and desktop

### 🔧 MLOps & DevOps

- **CI/CD Pipeline** — Automated testing, code quality, and security scanning on every push
- **Docker** — Containerized deployment for consistency across environments
- **Model Versioning** — Track model versions with full metadata
- **Monitoring** — Real-time health checks and performance tracking
- **Explainability** — Feature importance for transparent, auditable predictions

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           User Interface (Flask)            │
├─────────────────────────────────────────────┤
│      ML Model (Random Forest)               │
│      - Feature Engineering                  │
│      - Real-time Predictions                │
│      - Explainability Engine                │
├─────────────────────────────────────────────┤
│           Monitoring System                 │
│      - Health Checks                        │
│      - Performance Tracking                 │
│      - Model Versioning                     │
├─────────────────────────────────────────────┤
│           Data Layer (SQLite)               │
│      - Prediction History                   │
│      - Statistics                           │
└─────────────────────────────────────────────┘
```

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Total Transactions** | 100,000 |
| **Genuine Transactions** | 99,000 (99.0%) |
| **Fraudulent Transactions** | 1,000 (1.0%) |
| **Imbalance Ratio** | 99:1 |
| **Geographic Coverage** | 10 major US cities |
| **Amount Range** | $29 – $5,000 |

> 💡 The 1% fraud rate mirrors real-world credit card fraud statistics (0.5–2%), ensuring realistic model training.

### 📋 Features

| Feature | Type | Description |
|---------|------|-------------|
| `TransactionID` | Identifier | Unique transaction identifier |
| `TransactionDate` | Datetime | Transaction timestamp |
| `Amount` | Numeric | Transaction amount ($29–$5,000) |
| `MerchantID` | Categorical | Merchant identifier (1–999) |
| `TransactionType` | Categorical | Purchase or Refund |
| `Location` | Categorical | Transaction location (10 US cities) |
| `IsFraud` | Binary | Target variable (0=Genuine, 1=Fraud) |

---

## 🔧 Feature Engineering

### Temporal Features
```python
Hour              # 0-23 (transaction hour)
IsNightTime       # Boolean (22:00–06:00)
DayOfWeek         # 0-6 (Monday–Sunday)
IsWeekend         # Boolean (Saturday/Sunday)
```

### Transaction & Merchant Features
```python
HighAmount        # Boolean (Amount > $3,000)
IsRefund          # Boolean (Refund transaction flag)
MerchantRisk      # Calculated fraud rate per merchant
```

### Encoding Strategy
- **Label Encoding**: `TransactionType`, `Location`
- **Standardization**: `Amount` scaling via `StandardScaler`

---

## 🤖 Model Development

### Algorithms Compared

| Model | Role |
|-------|------|
| 🟢 Logistic Regression | Baseline — high interpretability, fast |
| 🟡 Decision Tree | Non-linear patterns, feature importance |
| 🔵 **Random Forest** ⭐ | **Best model** — ensemble, robust, handles overfitting |
| 🟣 SVM | High-dimensional data, kernel methods |

### Handling Class Imbalance — SMOTE

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

- ✅ Applied **only to training data** — prevents data leakage
- ✅ Balances classes from 99:1 to 1:1
- ✅ Generates synthetic samples via K-Nearest Neighbors
- ✅ Improved recall from 68% → 92%

---

## 💻 Technologies

| Layer | Stack |
|-------|-------|
| **Machine Learning** | Python 3.8+, pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn |
| **Web** | Flask 2.0+, HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, Chart.js |
| **MLOps** | Docker, Docker Compose, GitHub Actions, SQLite |
| **Deployment** | pickle, REST API, Gunicorn |

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
Docker (optional, for containerized setup)
```

### Option 1 — Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Build and run
docker-compose up

# Access application
open http://localhost:5000
```

### Option 2 — Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Verify model file exists at: model/fraud_detection_complete.pkl

# Run application
python app.py

# Run tests
python -m pytest tests/ -v
```

Access the app at `http://localhost:5000`.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── app.py                              # Flask application
├── database.py                         # Database operations
├── model_version.py                    # Model versioning
├── monitoring.py                       # Health monitoring
├── Dockerfile                          # Docker configuration
├── docker-compose.yml                  # Docker orchestration
├── requirements.txt                    # Python dependencies
├── model/
│   └── fraud_detection_complete.pkl    # Trained ML model
├── templates/
│   ├── index.html                      # Main UI
│   ├── history.html                    # Prediction history
│   ├── metrics.html                    # Model metrics
│   └── monitoring.html                 # System monitoring
├── static/
│   ├── css/style.css
│   └── js/script.js
├── notebooks/
│   ├── EDA.ipynb                       # Exploratory Data Analysis
│   └── model_training.ipynb            # Model development
├── data/
│   └── credit_card_fraud_dataset.csv   # Training dataset
└── tests/
    ├── test_app.py
    └── test_model.py
```

---

## 📱 Usage

### Training the Model

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle

df = pd.read_csv('data/credit_card_fraud_dataset.csv')

# Feature engineering
df['Hour'] = pd.to_datetime(df['TransactionDate']).dt.hour
df['IsNightTime'] = df['Hour'].apply(lambda x: 1 if x >= 22 or x <= 6 else 0)

X = df.drop(['IsFraud', 'TransactionID', 'TransactionDate'], axis=1)
y = df['IsFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

with open('model/fraud_detection_complete.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)
```

### Making Predictions via Code

```python
import pickle

with open('model/fraud_detection_complete.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']

transaction = {
    'Amount': 4500, 'Hour': 23, 'IsNightTime': 1,
    'IsHighAmount': 1, 'IsWeekend': 0,
    'TransactionType_encoded': 0, 'Location_encoded': 3
}

transaction_scaled = scaler.transform([list(transaction.values())])
prediction = model.predict(transaction_scaled)
probability = model.predict_proba(transaction_scaled)[0][1]

print(f"Fraud: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Probability: {probability*100:.2f}%")
```

### Using the Web Interface

1. Enter transaction details — amount, type, merchant, location, date/time
2. Click **"Check Transaction"**
3. View the fraud probability gauge, risk level (Low/Medium/High), and recommended actions

---

## 🎯 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main application page |
| `POST` | `/predict` | Submit transaction for fraud prediction |
| `GET` | `/api/options` | Retrieve dropdown options from model |
| `GET` | `/health` | Health check endpoint |

### Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 4500, "transaction_type": "purchase", "merchant_category": "electronics", "country": "USA", "hour": 23}'
```

```json
{
  "fraud": true,
  "probability": 0.87,
  "risk_level": "High",
  "message": "High risk transaction detected",
  "recommendation": "Manual review recommended"
}
```

---

## 📈 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** ⭐ | **99.89%** | **87.10%** | **88.10%** | **87.60%** | **0.95** |
| SVM | — | 81.5% | 89.4% | 85.3% | 0.91 |
| Decision Tree | 89.56% | 14.60% | 88.10% | 25.05% | 0.88 |
| Logistic Regression | 73.43% | 2.23% | 55.87% | 4.29% | 0.82 |

### Feature Importance

```
1. Amount           (28.5%) — Transaction amount is most predictive
2. MerchantRisk     (22.1%) — Merchant fraud history matters
3. IsNightTime      (18.3%) — Night transactions are higher risk
4. Hour             (12.7%) — Time-of-day patterns
5. Location         (10.4%) — Geographic risk varies
6. IsHighAmount      (5.2%) — Large transactions flagged
7. TransactionType   (2.8%) — Refunds slightly riskier
```

### Key Insights from EDA

- ✅ **No Missing Values** — Clean dataset, 100% completeness
- ✅ **Night Transactions** — 22:00–06:00 window shows **3× higher fraud rate**
- ✅ **Geographic Variation** — Certain cities show **2× higher fraud rates**
- ✅ **Merchant Concentration** — Top 10% of merchants account for **40% of fraud cases**
- ✅ **SMOTE Impact** — Recall improved from **68% → 92%** after resampling

---

## 🔄 CI/CD Pipeline

Every code push automatically triggers:

- ✅ Automated testing (15 unit tests)
- ✅ Code quality checks
- ✅ Security scanning
- ✅ Docker build and validation
- ✅ Model performance validation

---

## 🎯 MLOps Features

1. **Model Versioning** — Track model versions with metadata and performance history
2. **Continuous Integration** — Automated testing on every commit via GitHub Actions
3. **Containerization** — Docker + Docker Compose for consistent, reproducible deployment
4. **Monitoring** — Real-time health checks, performance tracking, and alerting
5. **Explainability** — Feature importance surfaces the reasoning behind every prediction

---

## 🔮 Future Enhancements

- [ ] Real-time data streaming pipeline
- [ ] A/B testing framework for model comparison
- [ ] Hyperparameter tuning (GridSearchCV / Optuna)
- [ ] Deep learning models (LSTM, Autoencoders)
- [ ] SHAP / LIME for per-prediction explainability
- [ ] Kubernetes deployment
- [ ] AWS / Azure cloud integration
- [ ] Email/SMS alerts for high-risk transactions
- [ ] Payment gateway integration (Stripe, PayPal)
- [ ] Mobile application (iOS / Android)
- [ ] API rate limiting and OAuth authentication

---

## 👤 Author

**Suvan Agrawal**

📧 [suvssextras@gmail.com](mailto:suvssextras@gmail.com)
🔗 [linkedin.com/in/suvan-agrawal](https://www.linkedin.com/in/suvan-agrawal/)
🐙 [github.com/suvan-agrawal](https://github.com/suvan-agrawal)

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset inspired by real-world credit card transaction patterns
- Built as a comprehensive end-to-end ML + MLOps project
- Thanks to the open-source community for the amazing tools and libraries

---

<div align="center">

**⭐ Star this repo if you find it helpful!**


</div>