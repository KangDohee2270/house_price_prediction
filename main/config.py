from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
config = {
  "mode": "grid_search", # grid_search or cv_only (for single model) / use_ensemble (for ensemble model)
  'files': {
    'X_csv': '../data/train_feature.csv',
    'y_csv': '../data/train_target.csv',
    'X_test_csv': '../data/test_feature.csv',
    'output_csv': '../results/five_fold.csv',
    'ml_output_model': '../results/ml_model.pkl',
    'submission_csv': '../results/new_stacking_test.csv',
  },

  "scaler": MinMaxScaler(),
  
  "ensembler": StackingRegressor, # VotingRegressor or StackingRegressor
  "ensembler_params": {"final_estimator": SVR()}, # "weights": [0.1, 0.3, 0.3, 0.3] for Voting
  "ensemble_model_list": [GradientBoostingRegressor, XGBRegressor, SVR], #RandomForestRegressor, 

  'model': SVR, # or RandomForestRegressor  
  
  'rfr_search_params': {
      "max_depth": [3, 5, 7],
  },
  'gbr_search_params': {
      "n_estimators": [1000, 2000], 
      "learning_rate": [0.001, 0.01],
      # "max_depth": [15, 20, 25]
  },
  'xgb_search_params': {
      "n_estimators": [500], 
      "learning_rate": [0.1],
      # "max_depth": [5, 10, 20]
  },
  'svr_search_params': {
      # "C": [3,4,5,6],
      # "epsilon": [0.02, 0.03, 0.04]
      "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01]
  },
  
  'rfr_model_params':dict(random_state=42
                          ),
  'gbr_model_params': dict(random_state=42,
                          n_estimators=500,
                          ),
                          # n_estimators=3000, 
                          # learning_rate=0.05, 
                          # max_depth=4, 
                          # max_features='sqrt', 
                          # min_samples_leaf=15, 
                          # min_samples_split=10, 
                          # loss='huber',

  'ridge_model_params': dict(),
  'lasso_model_params': dict(), #max_iter=int(1e7), alpha=5e-4, random_state=42
  'elasticnet_params': dict(max_iter=int(1e7), alpha=0.0005, l1_ratio=0.9),
  'svr_params': dict(C=3, epsilon=0.03,),
  'xgb_params': dict(random_state=42,
                     n_estimators=500,
                     learning_rate=0.1),
  
  # cv param
  "n_splits": 5

}


MODEL_PARAMS_DICT = {RandomForestRegressor: config["rfr_model_params"], 
                     GradientBoostingRegressor: config["gbr_model_params"], 
                     Ridge: config['ridge_model_params'], 
                     Lasso: config['lasso_model_params'], 
                     ElasticNet: config['elasticnet_params'], 
                     SVR:config['svr_params'],
                     XGBRegressor: config["xgb_params"]}

SEARCH_PARAMS_DICT = {RandomForestRegressor: config["rfr_search_params"], 
                      GradientBoostingRegressor: config["gbr_search_params"],
                      XGBRegressor: config["xgb_search_params"],
                      SVR: config["svr_search_params"] }