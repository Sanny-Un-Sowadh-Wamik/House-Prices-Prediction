# House Prices Prediction

This project tackles the Kaggle [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) competition. The goal is to predict the final sale prices of homes in Ames, Iowa by developing robust machine learning models.

## Overview

- **Objective:**  
  Build a predictive model to estimate house sale prices using data preprocessing, feature engineering, and ensemble machine learning techniques.

- **Dataset Files:**
  - **train.csv:** Contains training data with features and the target variable (`SalePrice`).
  - **test.csv:** Test data used to generate predictions.
  - **data_description.txt:** Detailed description and explanation of each feature in the dataset.
  - **house_prices_prediction.ipynb:** Jupyter Notebook with the complete workflow—from data exploration and preprocessing to modeling and submission.
  - **submission.csv:** The output file with predictions, ready to be submitted to the Kaggle competition.

## Steps to Reproduce

1. **Environment Setup:**
   - Ensure Python 3 is installed.
   - Install the required libraries:
     ```bash
     pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm scipy
     ```

2. **Data Preparation:**
   - Download `train.csv`, `test.csv`, and `data_description.txt` from the Kaggle competition page.
   - Place all files in the project directory.

3. **Running the Notebook:**
   - Open `house_prices_prediction.ipynb` in Jupyter Notebook or JupyterLab.
   - Run all cells sequentially. The Notebook will:
     - Load and explore the data.
     - Handle missing values and perform feature engineering.
     - Train multiple models, using an ensemble approach.
     - Generate predictions and create `submission.csv`.

4. **Submission:**
   - Verify the generated `submission.csv` file.
   - Upload the file to the Kaggle competition page.

## Methodology

- **Data Exploration & Visualization:**  
  Understand data distributions, correlations, and missing values.

- **Data Preprocessing & Feature Engineering:**  
  Handle missing data according to feature context, create new features (e.g., total square footage, house age, quality scores), and transform skewed variables.

- **Modeling:**  
  Use a diverse set of models including Lasso, Ridge, ElasticNet, Random Forest, Gradient Boosting, XGBoost, and LightGBM. Evaluate using 5-fold cross-validation (RMSLE) and ensemble their predictions for robustness.

- **Evaluation:**  
  The model performance is primarily evaluated using the Root Mean Squared Log Error (RMSLE), which is suitable for the range and distribution of sale prices.

## Further Improvements

- Hyperparameter tuning using GridSearchCV or Bayesian optimization.
- Experiment with advanced ensemble techniques (stacking or blending).
- Additional feature selection and engineering to further optimize model performance.

## License

This project is provided for educational and research purposes. You are welcome to modify and share the code as needed.

**Happy Modeling!**
