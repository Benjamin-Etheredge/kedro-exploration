from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from os import cpu_count
import optuna
from .. import utils


def model(params):
    return DecisionTreeClassifier(**params)


def train(
    train_x, 
    train_y,
    params=None
):
    params = params if params is not None else {}
    clf = model(params)
    clf.fit(train_x, train_y)

    return clf


def get_params(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        #'max_features': trial.suggest_int('max_features', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }
    return params


def hyperparam_sweep(
    train_x, 
    train_y,
    *sweep_args,
    **sweep_kwargs
):
    return utils.sweep(model, get_params, train_x, train_y, *sweep_args, **sweep_kwargs)


'''
from sklearn.base import clone
def analysis(
    model,
    train_x, 
    train_y,
):
    clf = clone(model))
    pred_y = clf.predict_proba(test_x)[:, 1]
    return {
        'auc': roc_auc_score(test_y, pred_y),
        'params': params,
        'clf': clf
    }
)
'''