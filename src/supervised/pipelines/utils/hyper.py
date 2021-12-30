import optuna
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.base import ClassifierMixin
from typing import Dict, Callable, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic


def create_objective(
    model_func: Callable[[Dict[str, Any]], ClassifierMixin], 
    params_func: Callable[[optuna.Trial], Dict[str, Any]],
    x_train, 
    y_train,
    folds: int = 5,
    scoring: str = 'f1', # TODO was this bug?
    n_jobs: int = -1,
) -> float:

    def objective(trial: optuna.Trial) -> float:
        params = params_func(trial)
        clf = model_func(params)
        return cross_val_score(clf, x_train, y_train, cv=folds, scoring=scoring, n_jobs=None).mean()

    return objective

from sklearn.base import clone


# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def learning_curve(
    estimator, 
    title, 
    X, 
    y, 
    axes=None, 
    ylim=None, 
    cv=10,
    n_jobs=-1, 
    #n_jobs=None, 
    train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    estimator = clone(estimator)

    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(15, 15))

    axes[0,0].set_title(title)
    if ylim is not None:
        axes[0,0].set_ylim(*ylim)
    axes[0,0].set_xlabel("Training examples")
    axes[0,0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, score_times = \
        model_selection.learning_curve(
                       estimator, X, y, 
                       cv=cv, 
                       n_jobs=n_jobs,
                       scoring='accuracy',
                       train_sizes=train_sizes,
                       random_state=4,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    score_times_mean = np.mean(score_times, axis=1)
    score_times_std = np.std(score_times, axis=1)

    # Plot learning curve
    axes[0,0].grid()
    axes[0,0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0,0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0,0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0,0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0,0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[0,1].grid()
    axes[0,1].plot(train_sizes, fit_times_mean, 'o-')
    axes[0,1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[0,1].set_xlabel("Training examples")
    axes[0,1].set_ylabel("fit_times")
    axes[0,1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[1,0].grid()
    axes[1,0].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[1,0].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[1,0].set_xlabel("fit_times")
    axes[1,0].set_ylabel("Score")
    axes[1,0].set_title("Performance of the model")

    # Plot n_samples vs score_times
    axes[1,1].grid()
    axes[1,1].plot(train_sizes, score_times_mean, 'o-')
    axes[1,1].fill_between(train_sizes, score_times_mean - score_times_std,
                         score_times_mean + score_times_std, alpha=0.1)
    axes[1,1].set_xlabel("Training examples")
    axes[1,1].set_ylabel("score_times")
    axes[1,1].set_title("Scalability of the model")
    #ic(train_scores_mean)

    return dict(
        plot=plot,
        report=dict(
            train_sizes=train_sizes.tolist(),
            train_scores_mean=train_scores_mean.tolist(),
            train_scores_std=train_scores_std.tolist(),
            test_scores_mean=test_scores_mean.tolist(),
            test_scores_std=test_scores_std.tolist(),
            fit_times_mean=fit_times_mean.tolist(),
            fit_times_std=fit_times_std.tolist(),
        )
    )


def sweep(
    model_func: Callable, 
    params_func: Callable, 
    train_x, 
    train_y, 
    folds: int = 5,
    study_name: Union[str, None] = None, 
    storage: Union[str, None] = None, 
    n_trials: int = 10, 
    timeout_seconds: int = 60,
    **objective_kwargs
) -> Dict[str, Any]:
    #ic(n_trials)
    #ic(timeout_seconds)

    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction='maximize')
    objective = create_objective(model_func, params_func, train_x, train_y, folds=folds, **objective_kwargs)
    study.optimize(objective, timeout=timeout_seconds, n_trials=n_trials)
    return dict(
        best_value=study.best_value,
        best_params=study.best_params,
    )


from sklearn.model_selection import validation_curve
def eval_neural_net(model, train_x, train_y, test_x, test_y):
    #return validation_curve(model, train_x, train_y, cv=5, scoring="accuracy")
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    return dict(
        train_score=model.score(train_x, train_y),
        test_score=model.score(test_x, test_y),
        pred_y=pred_y.tolist(),
    )