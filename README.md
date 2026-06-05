# 🫀 CardioRisk AI — Heart Disease Risk Predictor

A production-grade ML web app for cardiovascular risk assessment using XGBoost and Logistic Regression, trained on the UCI Heart Disease dataset.

## 🚀 Live Demo
https://cardiorisk-ai-pk04.streamlit.app/

## 📁 Files
```
cardiorisk-ai/
├── app.py                   ← Main Streamlit application
├── requirements.txt         ← Python dependencies
├── README.md                ← This file
└── heart_disease_uci.csv    ← Dataset
```


## 🧠 How It Works

### Data
- Synthetic data mirroring UCI Heart Disease distribution
- 13 clinical features, ~920 patient records
- 46% positive class (heart disease present)

### Models
| Model | Role | AUC-ROC |
|-------|------|---------|
| XGBoost | Primary (65% weight) | ~0.905 |
| Logistic Regression | Secondary (35% weight) | ~0.895 |
| **Ensemble** | **Final prediction** | **~0.882 (5-Fold CV AUC)** |

### Features Used
| Feature | Description |
|---------|-------------|
| age | Patient age in years |
| sex | 0=Female, 1=Male |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dL) |
| fbs | Fasting blood sugar >120 mg/dL |
| restecg | Resting ECG results (0-2) |
| thalch | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0-3) |
| thal | Thalassemia type |

## ⚠️ Disclaimer
This app is for **educational and research purposes only**. It is not a certified medical device and must not be used for clinical diagnosis or treatment decisions.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **ML**: XGBoost, scikit-learn
- **Viz**: Matplotlib
- **Data**: NumPy, Pandas
- **Deployment**: Streamlit Community Cloud 
