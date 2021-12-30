#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from os import cpu_count
import optuna
from .. import utils


def model(params):
    return GradientBoostingClassifier(**params)


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
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        #'max_features': trial.suggest_uniform('max_features', 0.01, 1),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        #'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        #'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        #'n_jobs': min(1, cpu_count()//5), # TODO don't hardcode cv=5 here
    }
    return params
def hyperparam_sweep(
    train_x, 
    train_y,
    *args, **kwargs
):
    return utils.sweep(model, get_params, train_x, train_y, *args, **kwargs)