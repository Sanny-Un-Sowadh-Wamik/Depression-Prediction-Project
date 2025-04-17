# Depression Prediction Project

**Kaggle Competition:** [Playground Series Season 4, Episode 11 – Exploring Mental Health Data](https://www.kaggle.com/competitions/playground-series-s4e11)  
**Final Test ROC‑AUC Score:** **0.94125**

---

## 🔍 Overview

This project presents an end-to-end machine learning pipeline to predict depression status using mental health and lifestyle survey data, as part of Kaggle's Playground Series (Season 4, Episode 11). The solution leverages data cleaning, feature engineering, statistical analysis, and advanced model tuning to achieve a strong leaderboard performance.

---

## 🗂️ Project Structure

. ├── data/ │ ├── raw/ # Original downloaded data │ └── processed/ # Cleaned & engineered datasets ├── notebooks/ │ └── Depression_Prediction_Project.ipynb ├── src/ │ ├── data_preparation.py # Cleaning, encoding, imputation, scaling │ ├── feature_engineering.py # Custom features and domain logic │ ├── modeling.py # Model training, CV, hyperparameter tuning │ └── evaluation.py # Metrics and evaluation visualizations ├── requirements.txt # Environment dependencies └── README.md # Project documentation
---

## 🚀 Approach

### 1. **Exploratory Data Analysis (EDA)**
- Visualized feature distributions and relationships.
- Identified data imbalance and correlations.
- Performed statistical tests (chi-square, t-tests) to validate significance.

### 2. **Preprocessing**
- Handled missing values with `SimpleImputer`.
- Encoded categorical variables using `OneHotEncoder`.
- Scaled features using `StandardScaler`.

### 3. **Feature Engineering**
- Created domain-specific features and interaction terms.
- Applied log transformations and binning for skewed features.

### 4. **Modeling**
- Trained a LightGBM classifier using `StratifiedKFold`.
- Hyperparameter tuning with `RandomizedSearchCV`.
- Built ensemble model by averaging top fold predictions.
- Evaluated with ROC-AUC and classification metrics.

---

## 📊 Results

| Metric                  | Score     |
|-------------------------|-----------|
| Cross-Validation ROC-AUC| ~0.938    |
| Public Test ROC-AUC     | 0.940XX   |
| **Private Test ROC-AUC**| **0.94125** ✅ |

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/depression-prediction.git
cd depression-prediction

# Set up a virtual environment
python -m venv venv
source venv/bin/activate        # On Linux/macOS
venv\Scripts\activate           # On Windows

# Install required packages
pip install -r requirements.txt
