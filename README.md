# DataFlow | Client-Side Machine Learning Dashboard

[**Launch Live Dashboard**](https://www.google.com/search?q=https://)

DataFlow is a lightweight, browser-based tool designed to simplify the transition from raw data to a predictive model. I built this because most machine learning workflows require a specific environment setup, Jupyter notebooks, or heavy backend dependencies. This dashboard removes those barriers by handling the entire pipeline—upload, analysis, and training—directly in the browser.

### The Concept

The core idea was to create a sandbox that remains independent of the dataset. Whether you are looking at environmental statistics, housing prices, or user behavior, the logic adapts. It identifies your features, handles the messy parts of data cleaning, and trains a regression model locally on your machine.

### Core Features

  * **Zero-Setup Pipeline:** You do not need Python or a server. It runs on vanilla JavaScript, making it fast and completely portable.
  * **Local Processing:** Data privacy is baked in by default. Since the processing happens on the client side, your CSV data never gets uploaded to a third-party server.
  * **Automatic Preprocessing:** The system handles label encoding for categorical text and normalization for numerical values so the algorithms can process them accurately right away.
  * **Integrated Diagnostics:** Beyond just training a model, it provides a correlation matrix to show how features relate and residual plots to help you understand where the model is actually failing.

### How it Works

1.  **Data Source:** Import any standard CSV.
2.  **Exploratory Analysis:** View automated summaries and feature relationships.
3.  **Feature Engineering:** The tool prepares the data by filling missing values and scaling inputs.
4.  **Model Training:** Select between Random Forest, Gradient Boosting, or Ridge Regression based on your accuracy needs.
5.  **Evaluation:** Analyze performance using R² scores and scatter plots to see how well the predictions align with reality.

### Technical Stack

The dashboard is built using standard web technologies (HTML, CSS, and JS) with Chart.js for rendering the analytics. The machine learning engine is implemented in pure JavaScript to ensure a seamless, installation-free experience.

-----

*Developed for researchers and developers looking for a quick, visual way to prototype regression models.*
