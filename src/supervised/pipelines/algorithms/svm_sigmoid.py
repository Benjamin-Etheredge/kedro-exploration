from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
# TODO what is SGDClassifier relation to svms?
import optuna
from .. import utils


def model(params):
    #https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
    return SVC(**params, cache_size=8192)


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
    params = dict(
        #penalty=trial.suggest_categorical('penalty', ['l1', 'l2']),
        #kernel=trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid']),
        kernel='sigmoid',
        C=trial.suggest_loguniform('C', 1e-2, 1e2),
        #coef0=trial.suggest_uniform('coef0', -1, 1),
        random_state=4,
        #probability=True, # TODO research why this uses a form of folds
        # TODO why is this not documented in leanearSVC
    )
    #if params['kernel'] == 'poly':
        #params['degree'] = trial.suggest_int('degree', 1, 5)

    #if params['kernel'] == 'poly' or params['kernel'] == 'sigmoid':
        #params['coef0'] = trial.suggest_loguniform('coef0', 1e-5, 1e5)
    
    return params


def hyperparam_sweep(
    train_x, 
    train_y,
    *sweep_args,
    **sweep_kwargs
):
    return utils.sweep(model, get_params, train_x, train_y, *sweep_args, **sweep_kwargs)


def kernel_exploration(
    train_x, 
    train_y,
    params=None
):
    params = params if params is not None else {}

    
