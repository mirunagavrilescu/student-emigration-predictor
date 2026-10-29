# Student Emigration Predictor: Machine Learning Pipeline

An end-to-end binary classification machine learning project built in Python, designed to predict the likelihood of university graduates emigrating based on academic and socio-economic factors.

## Tech Stack & Tools
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
* **Core Concepts:** Synthetic Data Generation, Exploratory Data Analysis (EDA), Data Preprocessing, Binary Classification (Logistic Regression), GUI Development.

## Pipeline Overview

### 1. Custom Dataset Engineering
Synthesized a realistic 1,000-row dataset from scratch using `numpy` and `pandas`. The logic incorporates complex real-world dependencies (e.g., faculty domain correlating with salary expectations). Deliberately introduced a 5% missing data rate to simulate raw, imperfect data.

### 2. Data Preprocessing & EDA
* **Imputation:** Prevented data leakage by applying mean imputation for numerical data and mode imputation for categorical data separately on the training and testing subsets.
* **Transformation:** Normalized numerical features using `StandardScaler` to balance large variance variables (salary vs. grades) and transformed categorical text variables via One-Hot Encoding.
* **EDA:** Conducted comprehensive visual analysis using Correlation Heatmaps and Violin Plots to identify feature relationships, confirm class balance, and verify the absence of extreme outliers.

### 3. Predictive Modeling (Logistic Regression)
Trained a binary classification algorithm that achieved an outstanding **99% accuracy**. The model was rigorously evaluated using a Confusion Matrix (184 True Positives, 113 True Negatives, 3 total errors) and a Classification Report to ensure unbiased pattern recognition for both classes (staying vs. leaving).

### 4. Interactive Graphical Interface
Developed a user-friendly GUI allowing users to input custom student profiles via visual elements. The application runs the model in the background to deliver real-time emigration predictions, probability metrics, and visual performance proof (Confusion Matrix).
