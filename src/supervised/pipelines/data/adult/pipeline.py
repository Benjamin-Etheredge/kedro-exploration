from kedro.pipeline import Pipeline, node


from .nodes import (
    clean, 
    build_feature_engineer,
    split,
    apply_feature_engineer,
)


# TODO use module pipeline to avoid names in data pipeline

def create_data_pipeline(**kwargs):
    data_pipeline = Pipeline(
        [
            node(
                clean, 
                ["uci_adult_data"],
                dict(
                    X="clean_adult_train_x",
                    y="clean_adult_train_y",
                ),
                tags=['data', 'adult']
            ),
            node(
                clean, 
                ["uci_adult_test"],
                dict(
                    X="clean_adult_test_x",
                    y="clean_adult_test_y",
                ),
                tags=['data', 'adult']
            ),
            node(
                build_feature_engineer,
                ["clean_adult_train_x", "clean_adult_train_y"],
                outputs=dict(
                    x_transformers="adult_x_transformers",
                    y_transformer="adult_y_transformer",
                ),
                tags=['data', 'adult']
            ),
            node(
                apply_feature_engineer,
                ["clean_adult_train_x", "clean_adult_train_y", "adult_x_transformers", "adult_y_transformer"],
                dict(
                    x="adult_train_x",
                    y="adult_train_y",
                ),
                tags=['data', 'adult']
            ),
            node(
                apply_feature_engineer,
                ["clean_adult_test_x", "clean_adult_test_y", "adult_x_transformers", "adult_y_transformer"],
                dict(
                    x="adult_test_x",
                    y="adult_test_y",
                ),
                tags=['data', 'adult']
            ),
        ]
    )

    analysis_pipeline = Pipeline(
        []
    )

    return data_pipeline