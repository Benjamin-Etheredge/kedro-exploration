from kedro.pipeline import Pipeline, node

from .nodes import (
    feature_engineer,
    get_data,
    get_sub_data,
    split,
)

#from ..utils import split_data
def create_sub_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                split,
                inputs=["uci_mnist_test"],
                outputs=dict(
                    x="clean_mnist_sub_test_x",
                    y="clean_mnist_sub_test_y",
                ),
            ),
            node(
                split,
                inputs=["uci_mnist_train"],
                outputs=dict(
                    x="clean_mnist_sub_train_x",
                    y="clean_mnist_sub_train_y",
                ),
                tags=["mnist_sub", "data"],
            ),
            node(
                feature_engineer,
                inputs=["clean_mnist_sub_train_x", "clean_mnist_sub_train_y"],
                outputs=dict(
                    x="mnist_sub_train_x",
                    y="mnist_sub_train_y",
                ),
                tags=["mnist_sub", "data"],
            ),
            node(
                feature_engineer,
                inputs=["clean_mnist_sub_test_x", "clean_mnist_sub_test_y"],
                outputs=dict(
                    x="mnist_sub_test_x",
                    y="mnist_sub_test_y",
                ),
                tags=["mnist_sub", "data"],
            ),
        ]
    )


def create_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                get_data, 
                inputs=None,
                outputs=dict(
                    train_x="clean_mnist_train_x",
                    train_y="clean_mnist_train_y",
                    test_x="clean_mnist_test_x",
                    test_y="clean_mnist_test_y",
                    ds_info="mnist_info",
                ),
                name='get_mnist_data',
                tags=['data', 'mnist']
            ),
            
            node(
                feature_engineer,
                inputs=["clean_mnist_train_x", "clean_mnist_train_y"],
                outputs=dict(
                    x="mnist_train_x",
                    y="mnist_train_y",
                ),
                tags=['data', 'mnist']
            ),
            node(
                feature_engineer,
                inputs=["clean_mnist_test_x", "clean_mnist_test_y"],
                outputs=dict(
                    x="mnist_test_x",
                    y="mnist_test_y",
                ),
                tags=['data', 'mnist']
            ),
        ]
    )
"""
            node(
                get_data, 
                inputs=[],
                outputs=dict(
                    ds_train="mnist_train_ds",
                    ds_test="mnist_test_ds",
                    ds_info="mnist_info",
                ),
                name='get_mnist_data',
            ),
            node(
                convert_to_numpy,
                "mnist_train_ds",
                dict(
                    x="converted_mnist_train_x",
                    y="converted_mnist_train_y",
                )
            ),
            node(
                convert_to_numpy,
                "mnist_test_ds",
                dict(
                    x="converted_mnist_test_x",
                    y="converted_mnist_test_y",
                )
            ),
            """