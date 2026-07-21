
# Flight Price Prediction with Random Forest Regressor

## Overview
This Jupyter notebook demonstrates a machine learning project focused on predicting flight prices using a Random Forest Regressor. The project involves data loading, extensive preprocessing (including one-hot encoding), exploratory data analysis (EDA), model training, evaluation, and hyperparameter tuning using GridSearchCV and RandomizedSearchCV.

## Dataset
# [Data-set LINK in Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
The project utilizes the `Clean_Dataset.csv` file, which contains various features related to flight bookings, such as airline, departure/arrival times, cities, number of stops, duration, days left until departure, and the target variable, price.

## Methodology

1.  **Data Loading and Initial Exploration**: The dataset is loaded into a pandas DataFrame. Initial checks include viewing the head of the DataFrame, identifying unique elements in categorical columns (e.g., `stops`), and checking for missing values and data types using `df.info()`.

2.  **Data Preprocessing**: 
    *   Unnecessary columns like `'Unnamed: 0'` and `'flight'` are dropped.
    *   The `'class'` column is converted into a numerical binary format (1 for 'Economy', 0 for others).
    *   The `'stops'` column is factorized into numerical categories.
    *   One-hot encoding is applied to all remaining categorical features such as `airline`, `source_city`, `destination_city`, `departure_time`, and `arrival_time`.

3.  **Model Training**: 
    *   The preprocessed data is split into training and testing sets using `train_test_split` (80% training, 20% testing).
    *   A Random Forest Regressor is initialized and trained on the training data.

4.  **Model Evaluation**: 
    *   The initial model performance is evaluated using the `score` method (R-squared).
    *   Additional metrics like R-squared (R2), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are calculated and displayed.
    *   A scatter plot visualizes the actual vs. predicted prices to assess model fit.

5.  **Feature Importance**: 
    *   Feature importances from the trained Random Forest model are extracted and sorted.
    *   A bar plot displays the top 5 most important features influencing flight prices.

6.  **Hyperparameter Tuning**: 
    *   **GridSearchCV**: An initial attempt at hyperparameter tuning is made using `GridSearchCV` with a reduced set of parameters to identify optimal combinations. Due to `sklearn` version updates, the `max_features='auto'` parameter might cause warnings or failures; it's recommended to use `'sqrt'`, `'log2'`, or `None` instead.
    *   **RandomizedSearchCV**: A more efficient tuning approach, `RandomizedSearchCV`, is employed with a broader range of hyperparameters to find the best model configuration. This method samples a fixed number of parameter settings from specified distributions.

7.  **Final Model Evaluation**: The best estimator obtained from `RandomizedSearchCV` is re-evaluated using the same metrics (R2, MAE, RMSE) on the test set.

## Results

**Initial Random Forest Model**:
*   **R2 Score**: ~0.985
*   **MSE (Mean Absolute Error)**: ~1073.56
*   **RMSE (Root Mean Squared Error)**: ~7615941.29

**Optimized Random Forest Model (after RandomizedSearchCV)**:
*   **R2 Score**: ~0.978
*   **MSE (Mean Absolute Error)**: ~1841.48
*   **RMSE (Root Mean Squared Error)**: ~11185739.86

*Note: While the R2 score decreased slightly after hyperparameter tuning, this could indicate that the initial default parameters were already performing well or that the sampled parameter space in `RandomizedSearchCV` didn't yield a significantly better combination within the given iterations. The increase in MAE and RMSE suggests a slight degradation in absolute error for this specific run.*

**Key Feature Importances**: The analysis revealed `class`, `duration`, and `days_left` as the most significant features influencing flight prices.

## Technologies Used
*   Python 3
*   pandas
*   scikit-learn
*   matplotlib

## Usage
To run this notebook:
1.  Ensure you have `Clean_Dataset.csv` in the same directory as the notebook.
2.  Install the required Python libraries:
    ```bash
    pip install pandas scikit-learn matplotlib
    ```
3.  Open and run the notebook cells sequentially in a Jupyter environment (e.g., Google Colab, Jupyter Lab, Jupyter Notebook).

## Future Work
*   Explore other regression models (e.g., XGBoost, LightGBM).
*   Perform more extensive feature engineering.
*   Conduct more thorough hyperparameter tuning with a wider search space or more iterations.
*   Analyze the impact of different preprocessing techniques on model performance.
*   Implement cross-validation consistently across all model evaluations.
