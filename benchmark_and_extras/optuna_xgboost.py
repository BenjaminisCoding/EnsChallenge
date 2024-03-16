import optuna 
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import warnings
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import skew, kurtosis
from sklearn.model_selection import KFold
import optuna 
warnings.filterwarnings('ignore')

from utils import load_data
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm 
from functools import partial
import argparse
import random



def objective(trial, train_data, labels, n_splits):

    params = {
    'booster': 'gbtree',
    'tree_method':'gpu_hist',
    # 'n_estimators': 1000,
    'max_depth': trial.suggest_int('max_depth', 3, 10), 
    'learning_rate': 0.025,
    'objective': 'multi:softmax',
    # 'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric':'mlogloss',
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    'gamma': 0,
    'subsample': 0.5,
    'colsample_bytree': 1
    # 'eval_metric':'merror'
    }

    random.seed(42)
    np.random.seed(42)

    return KFOLD(train_data, labels, n_splits, params)

def compute_score(bst, X_valid, y_valid):

    X_valid_xgb = xgb.DMatrix(X_valid)
    predictions = bst.predict(X_valid_xgb, iteration_range=(0, bst.best_iteration))
    predictions = pd.DataFrame(predictions)
    return np.round(accuracy_score(predictions,y_valid), 4)

def KFOLD(train_data, labels, n_splits, params):

    L_res = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(kf.split(train_data)):
        X_train, X_valid = train_data.iloc[train_index], train_data.iloc[val_index]
        y_train, y_valid = labels.iloc[train_index], labels.iloc[val_index]
        d_train = xgb.DMatrix(X_train, y_train)
        d_valid = xgb.DMatrix(X_valid, y_valid)

        num_round = 20_000
        evallist = [(d_train, 'train'), (d_valid, 'eval')]
        bst = xgb.train(params, d_train, num_round, evallist, early_stopping_rounds=100, verbose_eval = False)
        res = compute_score(bst, X_valid, y_valid)
        L_res.append(res)
    return np.mean(L_res)

def write_results(study, name_exp):

    with open(f'./optuna/{name_exp}.txt', 'w') as file:
        file.write(str(study.best_params) + '\n')
        file.write(str(study.best_value))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name_exp', required=True)
    parser.add_argument('--n_trials', required = True)
    parser.add_argument('--n_splits', default = 10)
    args = parser.parse_args()

    train_data, labels = load_data()
    study = optuna.create_study(direction='maximize')
    study.optimize(partial(objective, n_splits = int(args.n_splits), train_data = train_data, labels = labels), n_trials=int(args.n_trials))
    write_results(study, args.name_exp)









