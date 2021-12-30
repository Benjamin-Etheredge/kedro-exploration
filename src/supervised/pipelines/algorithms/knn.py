from sklearn.neighbors import KNeighborsClassifier
from os import cpu_count
from typing import Dict
from .. import utils


def model(params):
    return KNeighborsClassifier(**params)

def train(
    train_x, 
    train_y,
    params=None
):
    params = params if params is not None else {}
    params = {**params, 'n_jobs': min(32, max(1, cpu_count()//5))}
    clf = model(params)
    clf.fit(train_x, train_y)

    return clf


def get_params(trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 100),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        #'algorithm': trial.suggest_categorical('algorithm', ['ball_tree', 'kd_tree', 'brute']),
        'leaf_size': trial.suggest_int('leaf_size', 10, 100),
        'p': trial.suggest_int('p', 1, 2),
        #'n_jobs': min(32, max(1, cpu_count()//5)), # TODO don't hardcode cv=5 here
        'n_jobs': 32
    }
    return params

def hyperparam_sweep(
    train_x, 
    train_y, 
    *sweep_args, 
    **sweep_kwargs) -> Dict:
   return utils.sweep(model, get_params, train_x, train_y, *sweep_args, **sweep_kwargs)
