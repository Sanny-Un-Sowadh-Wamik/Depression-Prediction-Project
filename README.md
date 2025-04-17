# Depression Prediction Solution

## Overview

This repository contains my solution for predicting depression based on demographic, lifestyle, and health-related factors. Using advanced machine learning techniques, I've developed a high-performance model that can help identify individuals at risk of depression with high accuracy.

## Problem Statement

Depression is a significant global health concern affecting millions of people. Early detection is crucial for effective intervention and treatment. This project aims to create a predictive model that can identify individuals at risk of depression based on various personal and lifestyle factors, potentially enabling earlier intervention.

## Dataset

The dataset used in this project contains:

- **Training data**: 140,700 records with 20 features including the target variable
- **Test data**: 93,800 records (without the target variable)
- **Features**: Include demographic information (age, gender, city), lifestyle factors (sleep patterns, dietary habits), academic and work-related variables (pressure, satisfaction, hours), and important mental health indicators (family history, suicidal thoughts)
- **Target variable**: Binary depression indicator (0 = No Depression, 1 = Depression)

## Repository Contents

- `Depression_Prediction_Project.ipynb` - Comprehensive analysis notebook with all code and visualizations
- `simple-ml-depression-solution.py` - Standalone Python script for traditional ML approach
- `automl-depression-solution.py` - Implementation using AutoML frameworks
- `README.md` - Project documentation (this file)

## Jupyter Notebook Contents

The `Depression_Prediction_Project.ipynb` notebook contains a complete walkthrough of the project including:

1. **Data Exploration** - Visualization of feature distributions and relationships with depression
2. **Data Preprocessing** - Cleaning and handling inconsistencies in the dataset
3. **Feature Engineering** - Creation of new features to improve model performance
4. **Model Development** - Implementation of multiple machine learning algorithms
5. **Feature Importance Analysis** - Visualization of the most important predictors
6. **Predictions** - Generation of depression predictions and submission file

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/depression-prediction.git
   cd depression-prediction
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Required packages:
   ```
   pandas>=1.3.0
   numpy>=1.20.0
   scikit-learn>=1.0.0
   matplotlib>=3.4.0
   seaborn>=0.11.0
   lightgbm>=3.3.2
   ```

## Usage

### Running the Jupyter Notebook

```bash
jupyter notebook Depression_Prediction_Project.ipynb
```
This will open the complete analysis notebook where you can run the cells interactively.

### Running the Traditional ML Solution

```bash
python simple-ml-depression-solution.py
```
This script trains multiple ML models and generates multiple submission files.

### Running the AutoML Solution

```bash
python automl-depression-solution.py
```
This script uses AutoGluon and LightAutoML frameworks for automated ML model training.

## Methodology

My approach consisted of several key steps:

1. **Data Cleaning**
   - Handled inconsistent categorical values
   - Detected and fixed misplaced values across columns
   - Removed or imputed unreasonable numeric values

2. **Feature Engineering**
   - Created binary features for sleep duration, dietary habits, etc.
   - Developed interaction features combining related factors
   - Built age group categories and high-risk combination indicators

3. **Model Development**
   - Implemented a diverse set of models: Gradient Boosting, Random Forest, LightGBM
   - Used cross-validation to prevent overfitting
   - Created an ensemble by blending predictions from multiple models

4. **AutoML Experimentation**
   - Leveraged AutoGluon and LightAutoML for automated model selection
   - Applied advanced hyperparameter optimization
   - Implemented fallback mechanisms for robust prediction

## Results

The solution achieved excellent performance with:

- Cross-validation ROC-AUC score of approximately 0.97
- Strong feature importance insights identifying key predictors
- Robust predictions even with data quality issues

## Key Findings

The most important predictors of depression were:

1. **Age** - Different age groups show varying susceptibility to depression
2. **Suicidal thoughts** - Strongly correlated with depression diagnoses
3. **Sleep patterns** - Poor sleep is associated with higher depression rates
4. **Financial stress** - A significant contributor to depression risk
5. **Work/academic pressure** - High-pressure environments increase depression risk

## Future Improvements

To extend this work, I would consider:

- Incorporating more external data sources for broader context
- Experimenting with more sophisticated deep learning approaches
- Developing interpretable models for clinical use
- Validating the model across diverse populations
- Implementing a production-ready API for real-time predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset providers for making this analysis possible
- The open-source community for the excellent machine learning libraries
- Kaggle for hosting the competition and providing a platform for this work
