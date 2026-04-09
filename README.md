# DataFlow Pro

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)
![Build](https://img.shields.io/badge/Build-Proprietary-red)

**[View the Live Dashboard Here](https://iamshubh-8.github.io/Data-Flow-ML-DashBoard/)** *(Note: To execute the machine learning pipeline, the Python backend must be running locally or hosted via a cloud provider).*

DataFlow Pro is a streamlined, enterprise-grade machine learning pipeline and dashboard. Built with a decoupled architecture, it integrates a responsive Vanilla JavaScript client with a high-performance FastAPI and Scikit-Learn backend. The application enables users to train, evaluate, and deploy machine learning models through an intuitive graphical interface.

---

## Table of Contents
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Quick Start Guide](#quick-start-guide)
- [System Workflow](#system-workflow)
- [License & Ownership](#license--ownership)

---

## Key Features

* **Automated Task Detection:** Dynamically determines whether a dataset requires classification or regression based on the statistical properties of the target variable.
* **Robust Preprocessing Pipeline:** Utilizes `sklearn.pipeline` to automatically handle missing value imputation (median/mode), apply `StandardScaler` to numerical matrices, and execute `OneHotEncoder` for categorical features.
* **High-Cardinality Filtering:** Automatically scans for and drops high-cardinality identifiers (e.g., User IDs, UUIDs, or Names) prior to training to prevent model overfitting and memory bloat.
* **Dynamic Hyperparameter Tuning:** Allows users to adjust the estimator complexity of Random Forest or Gradient Boosting algorithms directly from the client interface.
* **Model Artifact Export:** Packages the fully trained preprocessing and modeling pipeline into a downloadable `.joblib` artifact for immediate production deployment.
* **Real-Time Inference:** Supports the ingestion of unlabelled CSV datasets, applying the saved pipeline transformations and model weights to generate a downloadable file of predictions.

---

## Technology Stack

**Client Application (Frontend)**
* HTML5 & CSS3 (Custom responsive grid layout, zero external CSS frameworks)
* Vanilla JavaScript (DOM manipulation, state management, and asynchronous API bridging)
* Chart.js (Data visualization, scatter plots, and residual charting)

**Server Application (Backend)**
* Python 3
* FastAPI & Uvicorn (High-performance asynchronous API routing)
* Scikit-Learn (Machine learning algorithms and pipeline architecture)
* Pandas & NumPy (Data manipulation and matrix operations)

---

## Quick Start Guide

DataFlow Pro utilizes a dedicated Python backend to process the core machine learning mathematics. To run the application locally, you must initialize the backend server.

### 1. Initialize the Backend Engine
Open your terminal, clone the repository, and install the required dependencies:

```bash
git clone [https://github.com/your-username/Data-Flow-ML-DashBoard.git](https://github.com/your-username/Data-Flow-ML-DashBoard.git)
cd Data-Flow-ML-DashBoard

# Install Python requirements
pip install fastapi uvicorn pandas scikit-learn python-multipart joblib

# Start the local server
python -m uvicorn main:app

Gemini said
Here is a highly professional, meticulously structured README.md. I have stripped out the excessive emojis, refined the language to sound like technical documentation written by a senior engineer, and maintained all the necessary links and proprietary licensing.

Copy and paste this directly into your repository:

Markdown
# DataFlow Pro

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)
![Build](https://img.shields.io/badge/Build-Proprietary-red)

**[View the Live Dashboard Here](https://your-username.github.io/Data-Flow-ML-DashBoard)** *(Note: To execute the machine learning pipeline, the Python backend must be running locally or hosted via a cloud provider).*

DataFlow Pro is a streamlined, enterprise-grade machine learning pipeline and dashboard. Built with a decoupled architecture, it integrates a responsive Vanilla JavaScript client with a high-performance FastAPI and Scikit-Learn backend. The application enables users to train, evaluate, and deploy machine learning models through an intuitive graphical interface.

---

## Table of Contents
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Quick Start Guide](#quick-start-guide)
- [System Workflow](#system-workflow)
- [License & Ownership](#license--ownership)

---

## Key Features

* **Automated Task Detection:** Dynamically determines whether a dataset requires classification or regression based on the statistical properties of the target variable.
* **Robust Preprocessing Pipeline:** Utilizes `sklearn.pipeline` to automatically handle missing value imputation (median/mode), apply `StandardScaler` to numerical matrices, and execute `OneHotEncoder` for categorical features.
* **High-Cardinality Filtering:** Automatically scans for and drops high-cardinality identifiers (e.g., User IDs, UUIDs, or Names) prior to training to prevent model overfitting and memory bloat.
* **Dynamic Hyperparameter Tuning:** Allows users to adjust the estimator complexity of Random Forest or Gradient Boosting algorithms directly from the client interface.
* **Model Artifact Export:** Packages the fully trained preprocessing and modeling pipeline into a downloadable `.joblib` artifact for immediate production deployment.
* **Real-Time Inference:** Supports the ingestion of unlabelled CSV datasets, applying the saved pipeline transformations and model weights to generate a downloadable file of predictions.

---

## Technology Stack

**Client Application (Frontend)**
* HTML5 & CSS3 (Custom responsive grid layout, zero external CSS frameworks)
* Vanilla JavaScript (DOM manipulation, state management, and asynchronous API bridging)
* Chart.js (Data visualization, scatter plots, and residual charting)

**Server Application (Backend)**
* Python 3
* FastAPI & Uvicorn (High-performance asynchronous API routing)
* Scikit-Learn (Machine learning algorithms and pipeline architecture)
* Pandas & NumPy (Data manipulation and matrix operations)

---

## Quick Start Guide

DataFlow Pro utilizes a dedicated Python backend to process the core machine learning mathematics. To run the application locally, you must initialize the backend server.

### 1. Initialize the Backend Engine
Open your terminal, clone the repository, and install the required dependencies:

```bash
git clone [https://github.com/your-username/Data-Flow-ML-DashBoard.git](https://github.com/your-username/Data-Flow-ML-DashBoard.git)
cd Data-Flow-ML-DashBoard

# Install Python requirements
pip install fastapi uvicorn pandas scikit-learn python-multipart joblib

# Start the local server
python -m uvicorn main:app
(Leave this terminal session running in the background to maintain the API connection).

2. Launch the Client
Navigate to the project directory in your file explorer and open the index.html file in your preferred modern web browser.

System Workflow
Data Ingestion: Upload a raw .csv file. The client parses a preview and extracts initial column metadata.

Exploratory Data Analysis (EDA): Select the target variable. The system generates feature distributions and a color-graded Pearson correlation matrix.

Feature Engineering: Select features for model inclusion. The backend stages the data for automated one-hot encoding and numerical scaling.

Model Training: Select the algorithmic approach (Random Forest, Gradient Boosting, or Linear/Logistic), configure engine complexity, and initiate the build.

Evaluation: Review cross-validation metrics (R² or Accuracy), analyze the predicted vs. actual scatter plots, and export the .joblib model artifact.

Inference: Submit a new, unlabeled dataset to the engine to receive instantaneous predictions based on the active pipeline.

License & Ownership
© 2026 DataFlow Pro. All Rights Reserved.

This project is a personal, proprietary build. It is not licensed for open-source distribution, unauthorized modification, reproduction, or commercial use without explicit, written permission from the creator.
