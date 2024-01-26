import pandas as pd
import numpy as np
import os

from config import config as cfg
from config import MODEL_PARAMS_DICT
from sklearn.metrics import root_mean_squared_error


if __name__ == "__main__":

  files = cfg.get('files') # File paths for load & save
  scaler = cfg.get('scaler')
  # 1. Load csv files (with scalining)
  X_train = pd.read_csv(files.get('X_csv'))
  X_test = pd.read_csv(files.get('X_test_csv'))
  if scaler != None:
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
  else:
    X_train, X_test = X_train.to_numpy(dtype=np.float32), X_test.to_numpy(dtype=np.float32)

  y_train = pd.read_csv(files.get('y_csv')).to_numpy(dtype=np.float32)
  y_train = y_train.ravel()

  # 2. Model/Ensembler setting
  if cfg.get("mode") == "use_ensemble":
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
    model = Ensembler(model_list, **ensembler_params)
  else:
    Model = cfg.get('model')
    print(f"Selected model: {Model.__name__}")
    model_params = MODEL_PARAMS_DICT[Model]

    model = Model(**model_params)
  
  # 3. Train/Validate model with full trainset
  model.fit(X_train, y_train)
  y_pred = model.predict(X_train)
  rmse = root_mean_squared_error(y_pred, y_train)
  print(f"RMSE Loss: {rmse}")

  # 4. Make submission file
  result = np.expm1(model.predict(X_test))
  print(result)
  test_origin = pd.read_csv("../data/test.csv")
  test_id = test_origin.Id.to_list()

  col_name = ["Id", "SalePrice"]
  list_df = pd.DataFrame(zip(test_id, result), columns=col_name)
  list_df.to_csv(files.get("submission_csv"), index=False)
  print('Save submission file completely')