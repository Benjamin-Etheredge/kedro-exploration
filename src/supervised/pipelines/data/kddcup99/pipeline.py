from kedro.pipeline import Pipeline, node

from .nodes import (
    build_feature_engineer,
    apply_feature_engineer,
    get_data,
    clean,
    split,
)
from .analysis import exploration

#from ..utils import split_data


def create_data_pipeline(**kwargs):
    data_pipeline = Pipeline([
        node(
            get_data, 
            inputs=None,
            outputs=dict(
                data="raw_kddcup99_data",
                targets="raw_kddcup99_targets"
            ),
        ),
        node(
            clean, 
            inputs=["raw_kddcup99_data", "raw_kddcup99_targets"],
            outputs=dict(
                data="clean_kddcup99_data",
                targets="clean_kddcup99_targets",
            ),
        ),
        node(
            split,
            inputs=["clean_kddcup99_data", 'clean_kddcup99_targets', 'params:test_ratio', 'params:seed'],
            outputs=dict(
                train_x="raw_kddcup99_train_x",
                train_y="raw_kddcup99_train_y",
                test_x="raw_kddcup99_test_x",
                test_y="raw_kddcup99_test_y",
            ),
        ),
        node(
            build_feature_engineer,
            inputs=["raw_kddcup99_train_x", "raw_kddcup99_train_y"],
            outputs=dict(
                x_transformers='x_transformers',
                y_transformer='y_transformer',
            )
        ),
        node(
            apply_feature_engineer,
            inputs=["raw_kddcup99_train_x", "raw_kddcup99_train_y", "x_transformers", "y_transformer"],
            outputs=dict(
                x="kddcup99_train_x",
                y="kddcup99_train_y",
            )
        ),
        node(
            apply_feature_engineer,
            inputs=["raw_kddcup99_test_x", "raw_kddcup99_test_y", "x_transformers", "y_transformer"],
            outputs=dict(
                x="kddcup99_test_x",
                y="kddcup99_test_y",
            )
        ),
    ])

    analaysis_pipeline = Pipeline([
        node(
            exploration,
            inputs=dict(data="raw_kddcup99_data", targets="raw_kddcup99_targets"),
            outputs="kddcup99_raw_report"
        ),
        node(
            exploration,
            inputs=dict(data="clean_kddcup99_data", targets="clean_kddcup99_targets"),
            outputs="kddcup99_clean_report"
        ),
        #node(
            #exploration,
            #inputs=dict(data="kddcup99_train_x", targets="kddcup99_train_y"),
            #outputs="kddcup99_train_report"
        #),
        #node(
            #exploration,
            #inputs=dict(data="kddcup99_test_x", targets="kddcup99_test_y"),
            #outputs="kddcup99_test_report"
        #),
    ])

    return data_pipeline + analaysis_pipeline 