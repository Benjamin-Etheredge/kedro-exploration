from kedro.pipeline import Pipeline, node

from .nodes import (
    clean, 
    feature_engineer,
    split,
    #apply_feature_engineer,
    #split_labels
)

#from ..utils import split_data


def create_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                clean, 
                ["uci_mushroom_data"],
                "clean_mushroom_data",
                name='clean_mushroom_data',
            ),
            # TODO make good splits
            node(
                split,
                ["clean_mushroom_data", "params:test_ratio", "params:seed"],
                dict(
                    train_x="clean_mushroom_train_x",
                    train_y="clean_mushroom_train_y",
                    test_x="clean_mushroom_test_x",
                    test_y="clean_mushroom_test_y"
                )
            ),
            node(
                feature_engineer,
                ["clean_mushroom_train_x", "clean_mushroom_train_y"],
                dict(
                    x="mushroom_train_x",
                    y="mushroom_train_y",
                )
            ),
            node(
                feature_engineer,
                ["clean_mushroom_test_x", "clean_mushroom_test_y"],
                dict(
                    x="mushroom_test_x",
                    y="mushroom_test_y",
                )
            ),
        ]
    )