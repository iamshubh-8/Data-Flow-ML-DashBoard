# 🌊 DataFlow Pro

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)
![Build](https://img.shields.io/badge/Build-Proprietary-red)

**DataFlow Pro** is an incredibly lightweight, enterprise-grade Machine Learning pipeline and dashboard. It bridges a beautiful, responsive Vanilla JavaScript frontend with a mathematically rigorous Python/FastAPI backend, allowing users to train, evaluate, and deploy Scikit-Learn models in seconds.

---

## 📑 Table of Contents
- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Quick Start Guide](#-quick-start-guide)
- [📖 How It Works](#-how-it-works)
- [©️ License & Ownership](#️-license--ownership)

---

## ✨ Key Features

* **🧠 Smart Task Detection:** Automatically detects whether your dataset requires **Classification** or **Regression** based on the target variable.
* **⚙️ Enterprise Preprocessing:** Built with `sklearn.pipeline`. Automatically handles missing values (Median/Mode imputation), applies `StandardScaler` to numerical data, and utilizes `OneHotEncoder` for categorical strings.
* **🛡️ Cardinality Guard:** Automatically scans for and drops high-cardinality identifiers (like User IDs or Names) to prevent severe model overfitting.
* **🎛️ Hyperparameter Tuning:** Adjust the complexity of your Random Forest or Gradient Boosting engines directly from the UI.
* **📥 Model Export:** Download your fully trained pipeline as a `.joblib` file for immediate production deployment.
* **🔮 Live Inference Engine:** Upload a completely new, unseen CSV file, and the backend will apply your exact preprocessing steps and model to generate a downloadable file of predictions.

---

## 🛠️ Tech Stack

**Frontend (The Face)**
* HTML5 & CSS3 (Custom responsive styling, no heavy frameworks)
* Vanilla JavaScript (DOM manipulation and API bridging)
* Chart.js (Data visualization and residual plotting)

**Backend (The Brain)**
* Python 3
* FastAPI & Uvicorn (High-performance asynchronous API)
* Scikit-Learn (Machine learning and pipelines)
* Pandas & NumPy (Data manipulation)

---

## 🚀 Quick Start Guide

Because DataFlow Pro utilizes a real Python backend to process heavy ML mathematics, it requires two simple steps to run locally.

### 1. Start the Backend Engine
Open your terminal, clone the repository, and install the required dependencies:

```bash
git clone [https://github.com/your-username/Data-Flow-ML-DashBoard.git](https://github.com/your-username/Data-Flow-ML-DashBoard.git)
cd Data-Flow-ML-DashBoard

# Install requirements
pip install fastapi uvicorn pandas scikit-learn python-multipart joblib

# Start the server
python -m uvicorn main:app

(Leave this terminal window open. It serves as the brain of your dashboard.)

2. Open the Dashboard
Simply double-click the index.html file in your file explorer to open it in your preferred web browser.

📖 How It Works
Data Source: Drop your raw .csv file. The frontend parses a preview and sends metadata to the UI.

Analysis (EDA): Select your Target Variable. The app instantly generates feature distributions and a color-graded Pearson correlation heatmap.

Engineering: Select the features you want to include. The backend will prepare to One-Hot encode categories and scale numbers.

Training: Choose your algorithm (Random Forest, Gradient Boosting, or Linear/Logistic), adjust engine complexity, and hit deploy.

Evaluation: View your R² or Accuracy scores, analyze the Predicted vs. Actual scatter plot, and download your .joblib model.

Inference: Drop a new, empty dataset into the engine to receive instant predictions based on your newly trained pipeline.
