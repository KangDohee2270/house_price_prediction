# house_price_prediction
![image](https://github.com/KangDohee2270/house_price_prediction/assets/39416550/5009a1d0-61a6-4437-8a7e-8a0ac4ba6979)
## Summary
- Competition_site: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
- Goal: Predict saleprice with 79 variables on kaggle
- Evaluation metric: Root mean squared error(RMSE)
- Number of samples(train/test): 1460 / 1459
- Best RMSE(rank): 0.12349(476/4220 on 1/26)

## Get Started
- Environment
  - Python 3.10
- Install Packages
  ```
  pip install -r requreiment.txt
  conda install ipykernel ipywidgets #for notebook files
  ``` 

## Usage
### 1. Preprocess dataset
  - My preprocessing steps are all documented in main/eda.ipynb.
  - If you want to use the preprocessed dataset, skip this step
  - Or if you want to use a preprocessed dataset that you already have, save the dataset as follows
 
  ```
  data
  |--train_features.csv # Trainset with all columns except the target column
  |--test_features.csv # Testset
  |--train_target.csv # Target column of trainset. Must be a dataset that has been de-skewed via `numpy.log1p()`
  ```

### 2. Validate model
  - Validate model(of sklearn or xgboost) with cross-validation
  - The overall configuration, including model selection and hyperparameter settings, can be set in main/config.py.
  - After all the configuration settings, run the code below
  ```
  cd main
  python validate_with_cv.py
  ```
## 3. Get Submission file
  - Run the code below with the same config you used for validation
  ```
  python train_and_get_result.py
  ```
  - The submission file path can be replaced with `submission_csv` in config.

## Reference
- https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- https://www.kaggle.com/code/niteshx2/top-50-beginners-stacking-lgb-xgb/notebook
- https://scikit-learn.org/stable/
- https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor
