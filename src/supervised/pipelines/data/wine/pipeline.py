from kedro.pipeline import Pipeline, node


from .nodes import (
    clean, 
    build_feature_engineer,
    split,
    apply_feature_engineer,
)


# TODO use module pipeline to avoid names in data pipeline

def create_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                clean, 
                ["uci_adult_data"],
                "clean_adult_data",
                name='clean_adult_data',
            ),
            # TODO make good splits
            node(
                split,
                ["clean_adult_data", "params:test_ratio", "params:seed"],
                dict(
                    train_x="clean_adult_train_x",
                    train_y="clean_adult_train_y",
                    test_x="clean_adult_test_x",
                    test_y="clean_adult_test_y"
                )
            ),
            node(
                build_feature_engineer,
                ["clean_adult_train_x", "clean_adult_train_y"],
                outputs=dict(
                    x_transformers="adult_x_transformers",
                    y_transformer="adult_y_transformer",
                )
            ),
            node(
                apply_feature_engineer,
                ["clean_adult_train_x", "clean_adult_train_y", "adult_x_transformers", "adult_y_transformer"],
                dict(
                    x="adult_train_x",
                    y="adult_train_y",
                )
            ),
            node(
                apply_feature_engineer,
                ["clean_adult_test_x", "clean_adult_test_y", "adult_x_transformers", "adult_y_transformer"],
                dict(
                    x="adult_test_x",
                    y="adult_test_y",
                )
            ),
        ]
    )