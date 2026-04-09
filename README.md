# DataFlow Pro

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)
![Build](https://img.shields.io/badge/Build-Proprietary-red)

**[View the Live Dashboard Here](https://iamshubh-8.github.io/Data-Flow-ML-DashBoard/)**  
*(Note: To execute the machine learning pipeline, the Python backend must be running locally or hosted via a cloud provider.)*

---

DataFlow Pro is a streamlined, enterprise-grade machine learning pipeline and dashboard. Built with a decoupled architecture, it integrates a responsive Vanilla JavaScript client with a high-performance FastAPI and Scikit-Learn backend.

The application enables users to train, evaluate, and deploy machine learning models through an intuitive graphical interface.

---

## Table of Contents
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Quick Start Guide](#quick-start-guide)
- [System Workflow](#system-workflow)
- [License & Ownership](#license--ownership)

---

## Key Features

- **Automated Task Detection:** Dynamically determines whether a dataset requires classification or regression based on the statistical properties of the target variable.  
- **Robust Preprocessing Pipeline:** Uses `sklearn.pipeline` for missing value imputation (median/mode), `StandardScaler` for numerical data, and `OneHotEncoder` for categorical features.  
- **High-Cardinality Filtering:** Automatically removes high-cardinality identifiers (e.g., User IDs, UUIDs, Names) to prevent overfitting and memory bloat.  
- **Dynamic Hyperparameter Tuning:** Adjust model complexity for Random Forest and Gradient Boosting directly from the UI.  
- **Model Artifact Export:** Export the complete trained pipeline as a `.joblib` file for production use.  
- **Real-Time Inference:** Upload new datasets and generate predictions instantly using the trained model pipeline.  

---

## Technology Stack

### Client Application (Frontend)
- HTML5 & CSS3 (Custom responsive layout, no external CSS frameworks)  
- Vanilla JavaScript (DOM manipulation, state management, API calls)  
- Chart.js (Data visualization, scatter plots, residual analysis)  

### Server Application (Backend)
- Python 3  
- FastAPI & Uvicorn (Asynchronous API framework)  
- Scikit-Learn (ML algorithms and pipelines)  
- Pandas & NumPy (Data processing and numerical operations)  

---

## Quick Start Guide

DataFlow Pro requires a Python backend to handle machine learning operations.

### 1. Initialize the Backend

```bash
git clone https://github.com/your-username/Data-Flow-ML-DashBoard.git
cd Data-Flow-ML-DashBoard

# Install dependencies
pip install fastapi uvicorn pandas scikit-learn python-multipart joblib

# Start the server
python -m uvicorn main:app

Keep this terminal running to maintain the backend API connection.

---

### 2. Launch the Client

Open the `index.html` file in your browser.

---

## System Workflow

1. **Data Ingestion:** Upload a `.csv` file and preview dataset structure.  
2. **EDA:** Select target variable and view distributions and correlation matrix.  
3. **Feature Engineering:** Choose features; the system applies encoding and scaling.  
4. **Model Training:** Select algorithm and tune parameters.  
5. **Evaluation:** Analyze metrics (R² / Accuracy) and prediction plots.  
6. **Inference:** Upload new data and generate predictions instantly.  

---

## License & Ownership

© 2026 DataFlow Pro. All Rights Reserved.

This project is proprietary. Unauthorized use, modification, or distribution is not permitted without explicit written permission.
