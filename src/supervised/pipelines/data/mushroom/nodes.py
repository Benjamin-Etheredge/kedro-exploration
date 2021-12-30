import pandas as pd
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import Dict


def feature_engineer(x, y):
    # TODO
    return dict(
        x=x,
        y=y.values.ravel(),
    )


def split(
    df: pd.DataFrame,
    test_ratio: float,
    seed: int
) -> pd.DataFrame:
    """Split the dataframe into a training and testing set.

    Args:
        df (pandas.DataFrame): The dataframe to split.

    Returns:
        pandas.DataFrame: The training and testing dataframes.

    """
    # Split the dataframe into a training and testing set
    train, test = sklearn.model_selection.train_test_split(
        df, 
        random_state=seed,
        test_size=test_ratio,
        stratify=df['class'])

    return dict(
        train_x=train.drop(columns=['class']),
        train_y=train['class'],
        test_x=test.drop(columns=['class']),
        test_y=test['class']
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataframe.

    Args:
        df (pandas.DataFrame): The dataframe to clean.

    Returns:
        pandas.DataFrame: The cleaned dataframe.

    """

    le = LabelEncoder()
    le.fit(df['class'])
    labels = le.transform(df['class'])

    df = df.drop(columns=['class'])

    ez_features = [
        "odor", 
        "spore-print-color",
        "stalk-surface-below-ring",
        "habitat",
        "population",
    ]

    df = df.drop(columns=ez_features)

    df = pd.get_dummies(df, drop_first=True, columns=df.columns)

    df['class'] = labels

    return df