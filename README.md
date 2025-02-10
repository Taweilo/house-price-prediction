# House Price Prediction - Multivariate analysis

![Python version](https://img.shields.io/badge/Python%20version-3.10%2B-lightgrey)
![GitHub last commit](https://img.shields.io/github/last-commit/Taweilo/house-price-prediction)
![GitHub repo size](https://img.shields.io/github/repo-size/Taweilo/house-price-prediction)
![Type of ML](https://img.shields.io/badge/Type%20of%20ML-Regression%20-red)

Badge [source](https://shields.io/)

 <img src="https://www.bouzaien.com/post/house-pricing-prediction/featured.png">
This project will follow the Business Analysis (BA) workflow to address house price prediction using linear regression techniques. The business problem is creating a regression model that can predict house prices based on the provided features. Therefore, real estate agents can utilize this model to evaluate the property.

```
├── Image                       
│
├── Code_USA_House_Price_Prediction.ipynb             <- code
├── LICENSE                                           <- MIT license

```

## 1. Business Understanding
The goal of this project is to develop a predictive model for housing prices based on various property attributes, including area, number of bedrooms and bathrooms, etc.

## 2. Data Understanding 

- Data resource: 

The dataset was loaded via Colab. The dataset is from Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview. Basic data analysis was performed to identify the shape of data, get column names, find missing values, and generate descriptive statistics. The pair plot demonstrated the relationship between variables. The distribution of the target variable was shown.

- Summary: 
  - 80 features, 1460 instances (small dataset)
  - Lost of overlapped
  - missing data
  - technical term
    
1. Property Characteristics**  
  - **MSSubClass**: Type of dwelling (e.g., 1-story, 2-story, duplex).  
  - **MSZoning**: Zoning classification (Residential, Commercial, Industrial, etc.).  
  - **LotFrontage**: Street-connected linear feet.  
  - **LotArea**: Lot size in square feet.  
  - **Street**: Type of road access (Paved or Gravel).  
  - **Alley**: Alley access type (Paved, Gravel, or None).  
  - **LotShape**: Shape of the lot (Regular, Irregular).  
  - **LandContour**: Flatness of the land (Flat, Hillside, Depression).  
  - **Utilities**: Available utilities (Public, Electricity only, etc.).  
  - **LotConfig**: Lot location (Corner, Cul-de-sac, etc.).  
  - **LandSlope**: Property slope (Gentle, Moderate, Severe).  
  - **Neighborhood**: Physical location within Ames.  

2. Proximity & Condition**  
  - **Condition1 & Condition2**: Proximity to railroads, streets, or parks.  

3. Building Details**  
  - **BldgType**: Type of dwelling (Single-family, Townhouse, Duplex).  
  - **HouseStyle**: Architectural style (1-story, 2-story, Split-level).  
  - **OverallQual & OverallCond**: Quality and condition of the house (1-10).  
  - **YearBuilt & YearRemodAdd**: Year of construction and last remodel.  

4. Roof & Exterior**  
  - **RoofStyle & RoofMatl**: Roof type and material.  
  - **Exterior1st & Exterior2nd**: Primary and secondary exterior materials.  
  - **MasVnrType**: Masonry veneer type (Brick, Stone, None).

![the distribution of each numeric variable](https://github.com/user-attachments/assets/dcadbc3b-079a-4944-9d45-9d4950f50a13)

 
## 3. Data Preparation 
1. Define the Categorical variables
2. Define X, y
3. Split data (train: 80%, val: 10%, test: 10%)
   -X_train, y_train: (1168, 79) (1168,)
   -X_Val, y_val: (146, 79) (146,)
   -X_test, y_test: (146, 79) (146,)
   
## 4. Modeling 
1. XGB
2. Light GBM
3. CatBoost
4. Ensemble

- Consider the model can deal with: 
  - Many variables: Both XGBoost and LightGBM can efficiently handle high-dimensional data.
  - Missing data: XGBoost can process missing values internally without imputation.
  - Technical and categorical data: LightGBM natively handles categorical features without requiring one-hot encoding.
  - Need for robust performance: Ensemble learning combines both models to achieve higher accuracy and generalization.

## 5. Hyperparameter tuning
- Optuna library to optimize the best hyperparameter for each model. Optuna primarily uses a Bayesian optimization approach, specifically the Tree-structured Parzen Estimator (TPE), to optimize hyperparameters efficiently. Bayesian optimization is an intelligent search strategy that models the objective function to select promising hyperparameter values instead of randomly guessing.
- Check hyperparameter tuning file on the repo
- XGB Hyperparameter  
`best_params = {
    "reg_lambda": 0.12688228439958346,
    "reg_alpha": 0.000889636209268654,
    "colsample_bytree": 0.613956213715118,
    "subsample": 0.7536161508442696,
    "learning_rate": 0.0531454528384571,
    "max_depth": 3,
    "min_child_weight": 0.1476017401560544,
    "n_estimators": 912,
    "tree_method": "hist",  # Using GPU for speedup
    "device": "cuda",  # Use GPU
    "random_state": 42
}`
- LightGBM Hyperparameter
`best_params = {
    "reg_lambda": 2.0246287271463874,
    "reg_alpha": 0.008333484840564514,
    "colsample_bytree": 0.5299443544711497,
    "subsample": 0.5421615536336953,
    "learning_rate": 0.09941176488664437,
    "max_depth": 39,
    "min_child_weight": 15.96738750769411,
    "n_estimators": 855,
    "num_leaves": 229,
    "min_data_in_leaf": 44,
    "feature_fraction": 0.5652203298763192,
    "bagging_fraction": 0.826706041753661,
    "bagging_freq": 2,
    "cat_smooth": 4.350099519287858,
    "random_state": 42,
    "verbose": -1
}`
- CatBoost Hyperparameter
`best_params = {
    'max_depth': 5,
    'learning_rate': 0.1295643155523358,
    'n_estimators': 304,
    'l2_leaf_reg': 0.015683846010523354,
    'bagging_temperature': 0.1863686770153567,
    'random_strength': 4.385386925403732,
    'border_count': 91,
    "cat_features": cat_features,  # Use categorical features
    "verbose": 0,
    "random_seed": 42,
    "task_type": "GPU",  # Use GPU instead of CPU
    "nan_mode": "Min"  # CatBoost automatically handles NaN
}`

## 6. Evaluation
| Model     | MSE (Mean Squared Error) | R² Score |
|-----------|-------------------------|----------|
| XGBoost   | 409,712,700.00           | 0.930   |
| LightGBM  | 383,640,300.00           | 0.935   |
| CatBoost  | 508,492,500.00           | 0.913   |
| Ensemble  | 422,597,000.00           | 0.928  |
- Consider the LightGBM had the best performance, it was deployed.
- LightGBM on test dataset:   MSE = 1.465984e+09  R² Score = 0.844193

## 7. Deployment

## 6. Future improvement
- Collect more data
- Dimension reduction: reduce the features for better 
- Realize the the domain knowledge: 

