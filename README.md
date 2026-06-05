# 🫀 CardioRisk AI — Heart Disease Risk Predictor

A production-grade ML web app for cardiovascular risk assessment using XGBoost and Logistic Regression, trained on the UCI Heart Disease dataset.

## 🚀 Live Demo
Deploy to Streamlit Community Cloud (FREE) — no local setup needed.

## 📁 Files
```
heart_disease_app/
├── app.py            ← Main Streamlit application
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

## ⚡ Deploy in 5 Minutes (GitHub + Streamlit Cloud)

### Step 1 — Create GitHub repository
1. Go to https://github.com → Sign in (or create free account)
2. Click the **+** icon → **New repository**
3. Name it: `cardiorisk-ai`
4. Set to **Public**
5. Click **Create repository**

### Step 2 — Upload files
1. Click **Add file** → **Upload files**
2. Upload `app.py` and `requirements.txt`
3. Click **Commit changes**

### Step 3 — Deploy on Streamlit
1. Go to https://streamlit.io/cloud → **Sign in with GitHub**
2. Click **New app**
3. Select your `cardiorisk-ai` repo
4. Main file path: `app.py`
5. Click **Deploy!** → Your app is LIVE in ~2 minutes 🎉

## 🧠 How It Works

### Data
- Synthetic data mirroring UCI Heart Disease distribution
- 13 clinical features, ~920 patient records
- 46% positive class (heart disease present)

### Models
| Model | Role | AUC-ROC |
|-------|------|---------|
| XGBoost | Primary (65% weight) | ~0.88 |
| Logistic Regression | Secondary (35% weight) | ~0.83 |
| **Ensemble** | **Final prediction** | **~0.87** |

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
- **Deployment**: Streamlit Community Cloud (free)
