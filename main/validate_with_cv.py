
import pandas as pd

from sklearn.metrics import root_mean_squared_error

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
import numpy as np

from copy import deepcopy

from config import config as cfg
from config import MODEL_PARAMS_DICT, SEARCH_PARAMS_DICT


def cross_validate(model, x_train: np.array, y_train: np.array, model_params: tuple, n_splits: int):
  """ Cross Validation for ML model

    Args:
      Model : ML model object from sklearn(ex. sklearn.ensemble.RandomForestRegressor())
      x_train (np.array): features used for training. It must be of type np.array 
      y_train (np.array): target used for training. It must be of type np.array 
      model_params (tuple): model parameters
      n_splits (int): number of folds

    Return:
      mean absolute error for train, validation set
  """
  models = [deepcopy(model) for _ in range(n_splits)]
  metrics = {'trn_rmse': [], 'val_rmse': []} # loss_list to metrics

  kf = KFold(n_splits = n_splits, shuffle = True, random_state = 1111)
  ##하나의 고정된 validation set을 사용한다면 해당 성능이 일반적으로 좋은지 운으로 좋았던건지 판단할 수 있음
  # print(y_train.shape)

  for i, (trn_idx, val_idx) in enumerate(kf.split(x_train)):
    model= models[i]
    X_trn, y_trn = x_train[trn_idx], y_train[trn_idx]
    X_val, y_val = x_train[val_idx], y_train[val_idx]
    model.fit(X_trn, y_trn)
    y_pred_trn = model.predict(X_trn)
    y_pred_val = model.predict(X_val)
    trn_rmse = root_mean_squared_error(y_pred_trn, y_trn)
    val_rmse = root_mean_squared_error(y_pred_val, y_val)
    print(f"Fold {i+1} Done.")
    print(f"Loss: Train: {trn_rmse:.6f}, Val: {val_rmse:.6f}")
    print(y_val[:5], y_pred_val[:5])
    metrics['trn_rmse'].append(trn_rmse)
    metrics['val_rmse'].append(val_rmse)
    
  return pd.DataFrame(metrics)

def find_best_params(model, x_train: np.array, y_train: np.array, search_params: dict, n_splits: int):
  reg_model = GridSearchCV(model, search_params, scoring="neg_root_mean_squared_error", cv=10)
  print(f"Test params: {search_params}")
  reg_model.fit(x_train, y_train)
  print(f"Best model: {reg_model.best_estimator_}")
  best_model = reg_model.best_estimator_
  scores = cross_val_score(best_model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=n_splits)
  return -scores.mean(), scores.std()

if __name__ == "__main__":
  # 0. Load conig

  files = cfg.get('files') # File paths for load & save
  scaler = cfg.get('scaler')
  # 1. Load csv files (with scalining)
  X_train = pd.read_csv(files.get('X_csv'))
  if scaler != None:
    X_train = scaler.fit_transform(X_train)
  else:
    X_train = X_train.to_numpy(dtype=np.float32)

  y_train = pd.read_csv(files.get('y_csv')).to_numpy(dtype=np.float32)
  y_train = y_train.ravel()

  mode = cfg.get("mode")
  # 2. Model/Ensembler setting
  if mode == "use_ensemble":
    Ensembler = cfg.get("ensembler")
    ensembler_params = cfg.get("ensembler_params")
    print(f"Selected Ensembler: {Ensembler.__name__}")
    Model_list = cfg.get("ensemble_model_list")
    model_list = []
    for Model in Model_list:
      model_params = MODEL_PARAMS_DICT[Model]
      model = Model(**model_params)
      model_list.append((f"{Model.__name__}", model))
    print(f"model list to ensemble: {model_list}")
    model = Ensembler(model_list, **ensembler_params )
  
  else:
    Model = cfg.get('model')
    print(f"Selected model: {Model.__name__}")
    model_params = MODEL_PARAMS_DICT[Model]
    model = Model(**model_params)
  
  # 3. Cross-validation
  n_splits = cfg.get("n_splits")
  if mode == "grid_search":
    print("Grid Search Mode...")
    model.fit(X_train, y_train)
    default_scores = cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=n_splits)
    print(f"Default Loss: {-default_scores.mean():.4f}({default_scores.std():.4f})")
    
    search_params = SEARCH_PARAMS_DICT[Model]
    model = Model(**model_params)
    loss_mean, loss_std = find_best_params(model, X_train, y_train, search_params, n_splits)
    print(f"Best Loss: {loss_mean:.4f}({loss_std:.4f})")
  
  else:  
    res = cross_validate(model, X_train, y_train, model_params, n_splits)

    res = pd.concat([res, res.apply(['mean', 'std'])])
    print(res)
    res.to_csv(files.get('output_csv'))
  