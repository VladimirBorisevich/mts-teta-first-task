import pandas as pd
import numpy as np
import os
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import catboost as cat
import lightgbm as lgbm
from project.src.code.prep import read_yaml, read_data
import pickle


def linear(X, y, test, exp_id, run_name):
    run_id = mlflow.start_run(
        experiment_id=exp_id.experiment_id,
        run_name=run_name)
    linear = LinearRegression(normalize=True)
    linear.fit(X, y)
    pred = linear.predict(test.drop("price", axis=1))
    client.log_metric(run_id.info.run_id, "MAE", round(mean_absolute_error(test.price, pred)))
    client.log_metric(run_id.info.run_id, "r2", r2_score(test.price, pred))
    params = linear.get_params()
    client.log_param(run_id.info.run_id, "normalize", "True")
    # client.log_artifact(run_id.info.run_id,local_path='test.ipynb',
    #                            artifact_path='code')
    # mlflow.sklearn.log_model(linear,
    #                                 artifact_path="model",
    #                                 registered_model_name="mlruns/models/linear")
    mlflow.end_run()


def tree():
    pass


def forest():
    pass


def light():
    pass


def catboost():
    pass


def train_light(X_train, y_train, X_test, y_test, scaler_name):
    with open(f"scalers/{scaler_name}.pkl", 'rb') as file:
        scaler = pickle.load(file)
    params = {'bagging_fraction': 0.9615087447098526,
              'bagging_freq': 0,
              'boosting_type': 'gbdt',
              'colsample_bytree': None,
              'feature_fraction': 0.8456892704201968,
              'lambda_l1': 0.24842029273784616,
              'lambda_l2': 0.10134215415038501,
              'learning_rate': 0.08006161308805573,
              'metric': 'mae',
              'min_child_samples': None,
              'min_child_weight': 0.007497551098352097,
              'min_data_in_leaf': 150,
              'min_sum_hessian_in_leaf': None,
              'num_leaves': 152,
              'objective': 'regression',
              'seed': 123,
              'subsample_for_bin': 3500,
              'verbose': -1,
              'n_estimators': 5000}
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
    train_dataset = lgbm.Dataset(X_train, y_train,
                                 categorical_feature=['building_type', 'object_type', 'year', 'hour'])
    test_dataset = lgbm.Dataset(X_test, y_test,
                                categorical_feature=['building_type', 'object_type', 'year', 'hour'])
    model = lgbm.train(params=params,
                       train_set=train_dataset,
                       valid_sets=[train_dataset, test_dataset],
                       num_boost_round=20000,
                       early_stopping_rounds=50,
                       verbose_eval=100)
    with open("models/lgbm.pkl", 'wb') as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    config = read_yaml("config.yml")
    project_path = config["project_path"]

    mlflow.set_registry_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    client = mlflow.tracking.MlflowClient(tracking_uri="sqlite:///mlruns/mlflow.db")

    try:
        exp_id = client.create_experiment("test")
    except mlflow.exceptions.MlflowException:
        exp_id = client.get_experiment_by_name("test")

    train = read_data(os.path.join(project_path, *["data", "clean", "train_40_3.csv"]))
    test = read_data(os.path.join(project_path, *["data", "clean", "test_40_3.csv"]))
    train = train.drop(["date_time", "region"], axis=1)
    test = test.drop(["date_time", "region"], axis=1)
    y = train.price
    X = train.drop(["price"], axis=1)

    # linear(X, y, test, exp_id, "first try")
    train_light(X, y, test.drop("price", axis=1), test.price, config["stream"]["scaler_name"])
