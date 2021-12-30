from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from os import cpu_count
from .. import utils
from mlrose_hiive import NeuralNetwork


def model(params):
    return NeuralNetwork(
        activation='relu',
        max_iters=1000,
        bias=True,
        is_classifier=True,
        **params)
    # hidden_nodes = [], activation = 'sigmoid', \
    #                                 algorithm = 'random_hill_climb', max_iters = 1000, \
    #                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
    #                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
    #                                 random_state = 3)


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

    layers = trial.suggest_int('layers', 1, 3)
    nodes = trial.suggest_categorical('nodes', [2**i for i in range(1, 7)])
    params = {
        'hidden_nodes': [nodes for _ in range(layers)],
        #'algorithm': trial.suggest_categorical('algorithm', ['random_hill_climb', 'simulated_annealing', 'genetic_alg']),
        #'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling']),
        #'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-1),
        #'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        #'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
        #'max_iter': trial.suggest_int('max_iter', 1, 1000),
        #'activation': trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
        #'solver': trial.suggest_categorical('solver', ['sgd', 'adam']),
        #'verbose': trial.suggest_categorical('verbose', [0, 1, 2]),
        #'warm_start': trial.suggest_categorical('warm_start', [True, False]),
        'random_state': 4,
    }
    return params



# TODO pull out sweep
def hyperparam_sweep(
    train_x, 
    train_y,
    *sweep_args,
    **sweep_kwargs
):
    results = utils.sweep(model, get_params, train_x, train_y, *sweep_args, **sweep_kwargs)
    
    best_params = results['best_params']
    del results['best_params']

    layers = best_params['layers']
    del best_params['layers']

    nodes = best_params['nodes']
    del best_params['nodes']

    best_params['hidden_layer_sizes'] = [(nodes) for _ in range(layers)]
    return {
        'best_params': best_params,
        **results,
    }