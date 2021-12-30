from typing import Callable, Any
from kedro.extras.datasets.pickle import pickle_dataset
from kedro.pipeline import Pipeline, node
from numpy.lib.npyio import save

from supervised.pipelines.utils.hyper import learning_curve

from . import knn, decision_tree, boosting, neural_net, svm_rbf, svm_sigmoid, svm_linear
from .. import utils
import time
from kedro.io import DataCatalog, PartitionedDataSet, MemoryDataSet
from kedro.extras.datasets import text, pandas, pickle, yaml, matplotlib

# loaded in with hooks.py
def create_datasets():
    datasets = {}

    data_names = ["mushroom", "mnist", "mnist_sub", "kddcup99", "adult"] # TODO maybe move to data or pull out

    data_elements = {
        "x_transformers": ["data/03_primary", pickle.PickleDataSet, "x_transformers",],
        "y_transformer": ["data/03_primary", pickle.PickleDataSet, "y_transformer",],
    }
    for data_name in data_names:
        path_base = f"data/03_primary/{data_name}"
        datasets[f"{data_name}_x_transformers"] = PartitionedDataSet(
            path=f"{path_base}/x_transformers",
            dataset=pickle.PickleDataSet
        )
        datasets[f"{data_name}_y_transformer"] = pickle.PickleDataSet(
            filepath=f"{path_base}/x_transformers.pkl",
        )

    models = ["svm_linear", "svm_rbf", "boosting", "decision_tree", "neural_network", "knn"]
    #models = ["svm_linear", "svm_rbf", "svm_sigmoid", "boosting", "decision_tree", "neural_network", "knn"]

    elements = {
        "baseline_model": ["data/06_models", pickle.PickleDataSet, "baseline_model.pkl"],
        "baseline_training_time": ["data/08_reporting", text.TextDataSet, "baseline_training_time.txt",],
        "baseline_report": ["data/08_reporting", yaml.YAMLDataSet, "baseline_report.yml",],
        "model": ["data/06_models", pickle.PickleDataSet, "model.pkl",],
        "report": ["data/08_reporting", yaml.YAMLDataSet, "baseline_report.yaml",],
        "params": ["data/05_model_input", yaml.YAMLDataSet, "params.yaml",],
        "training_time": ["data/08_reporting", text.TextDataSet, "baseline_training_time.txt",],
        "report": ["data/08_reporting", yaml.YAMLDataSet, "report.yml",],
        "learning_plot": ["data/08_reporting", matplotlib.MatplotlibWriter, "learning_plot.jpg",],
        "learning_report": ["data/08_reporting", yaml.YAMLDataSet, "learning_report.yml",],
        "name": ["data/08_reporting", text.TextDataSet, "name.txt",],
        "sql": ["data/08_reporting", text.TextDataSet, "sql_path.txt",],
    }


    for data_name in data_names:
        for model in models:
            #datasets[f"{data_name}_{model}_name"] = text.TextDataSet(
                #filepath=f"data/06_models/{data_name}/{model}/name.txt",
            #)

            #datasets[f"{data_name}_{model}_name"].save(f"{data_name}_{model}")

            for element, (base_dir, ds_func, file_name) in elements.items():
                key = f"{data_name}_{model}_{element}"
                datasets[key] = ds_func(filepath=f"{base_dir}/{data_name}/{model}/{file_name}")
            # Note: this creates a dir that needs to exist for sql to work
            datasets[f"{data_name}_{model}_name"].save(f"{data_name}_{model}")
            datasets[f"{data_name}_{model}_sql"].save(
                f"sqlite:///data/08_reporting/{data_name}/sql/{model}.db")
    

    # TODO move to data
    # TODO register param names

    #for name in data_names:

    return datasets


def create_algorithm_pipeline(
    model_name: str,
    data_name: str, 
    model_func: Callable,
    hyperparam_sweep_func: Callable,
    train_func: Callable,
    kedro_train_x: Any, 
    kedro_train_y: Any, 
    kedro_test_x: Any, 
    kedro_test_y: Any, 
) -> Pipeline:
    # TODO move to utils

    # format is data_model_other
    baseline_model = f"{data_name}_{model_name}_baseline_model"
    baseline_time = f"{data_name}_{model_name}_baseline_training_time"
    baseline_report = f"{data_name}_{model_name}_baseline_report"

    model = f"{data_name}_{model_name}_model"
    train_time = f"{data_name}_{model_name}_training_time"
    params = f"{data_name}_{model_name}_params"
    score = f"{data_name}_{model_name}_score" # memory cause I don't care about it
    report = f"{data_name}_{model_name}_report"

    learning_report = f"{data_name}_{model_name}_learning_report"
    learning_plot = f"{data_name}_{model_name}_learning_plot"

    # TODO make hook
    #def wrapped_train(*args, **kwargs):
        #start_time = time.monotonic()
        #if len(args) < 3 and 'params' not in kwargs:
            #kwargs['params'] = {}
        #return dict(
            #model=train_func(*args, **kwargs),
            #time=str(time.monotonic() - start_time,)
        #)

    return Pipeline([
        # Baseline
        node(
            train_func, 
            inputs=dict(
                train_x=kedro_train_x,
                train_y=kedro_train_y,
            ),
            outputs=baseline_model,
            tags=['base', data_name, model_name]
            #name='baseline'
        ),
        node(
            utils.grade,
            inputs=[baseline_model, kedro_train_x, kedro_train_y, kedro_test_x, kedro_test_y],
            outputs=baseline_report,
            tags=['base', 'report', data_name, model_name],
            #name="baseline_report"
        ),
        # TODO swap to passing in model and cloning it
        node(
            hyperparam_sweep_func,
            inputs=dict(
                train_x=kedro_train_x,
                train_y=kedro_train_y,
                folds="params:k_folds",
                study_name=f"{data_name}_{model_name}_name",
                storage=f"{data_name}_{model_name}_sql",
                timeout_seconds="params:hyper_timeout_seconds",
                n_trials="params:hyper_n_trials",
                scoring="params:hyper_scoring",
            ),
            outputs=dict(
                best_params=params,
                best_value=score,
            ),
            tags=['hyper', data_name, model_name],
        ),
        node(
            train_func, 
            inputs=[kedro_train_x, kedro_train_y, params],
            outputs=model,
            tags=['train', data_name, model_name],
        ),
        node(
            utils.learning_curve,
            inputs=dict(
                estimator=model, 
                title=f"{data_name}_{model_name}_name",
                X=kedro_train_x, 
                y=kedro_train_y),
            outputs=dict(
                plot=learning_plot,
                report=learning_report
            ),
            tags=['report', 'learning_curve', data_name, model_name],
        ),
        node(
            utils.grade,
            inputs=[model, kedro_train_x, kedro_train_y, kedro_test_x, kedro_test_y],
            outputs=report,
            tags=['report', data_name, model_name]
        )
    ],)
    #namespace=f"{data_name}_{model_name}"

def create_pipelines(
    data_name,
    kedro_train_x,
    kedro_train_y,
    kedro_test_x,
    kedro_test_y,
    **kwargs
) -> Pipeline:

    data_kwargs = dict(
        kedro_train_x=kedro_train_x,
        kedro_train_y=kedro_train_y,
        kedro_test_x=kedro_test_x,
        kedro_test_y=kedro_test_y,
    )

    pipelines = {
        f"{data_name}_{model_name}": create_algorithm_pipeline(
                    model_name=model_name,
                    data_name=data_name,
                    hyperparam_sweep_func=module.hyperparam_sweep,
                    train_func=module.train,
                    model_func=module.model,
                    **data_kwargs)
        for model_name, module in [
            ('knn', knn),
            ('decision_tree', decision_tree),
            ('boosting', boosting),
            ('neural_network', neural_net),
            ('svm_linear', svm_linear),
            ('svm_rbf', svm_rbf),
            #('svm_sigmoid', svm_sigmoid),
        ]
    }

    # TODO maybe param range file
    # output yaml
    return {
        **pipelines,
        f"{data_name}_all_algs": sum(pipelines.values(), Pipeline([])),
    }