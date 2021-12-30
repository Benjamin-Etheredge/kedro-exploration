from kedro.pipeline.pipeline import TRANSCODING_SEPARATOR
import pandas as pd
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.base import TransformerMixin
from typing import Dict, Union, Any
#from sklearn import datasets
import tensorflow_datasets as tfds
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import sklearn.datasets

import pandas as pd
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import Dict


def build_transformer(key, transformer):

    pass
    #return lambda x: transformer().transform(x[])


def build_feature_engineer(
    x: pd.DataFrame, 
    y: pd.Series
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:

    transformer = SelectKBest(chi2, k=2)
    #transformer = transformer.fit(x, y)
    # TODO
    x_transformers = {}
    for col in x.columns:
        # TODO handle other col types
        #if x[col].dtype == np.float64:
        scaler = StandardScaler()
        scaler.fit(x[col].values.reshape(-1, 1))
        x_transformers[col] = scaler
        


    le = LabelEncoder()
    le.fit(y)

    return dict(
        x_transformers=x_transformers,
        y_transformer=le,
    )

def apply_feature_engineer(
    x: pd.DataFrame,
    y: pd.Series,
    x_transformers: Dict[str, TransformerMixin],
    y_transformer: TransformerMixin
):
    new_x = x.copy()
    for key, transformer in x_transformers.items():
        #transformer = transformer_callable()  # kedro requires this step
        new_x[key] = transformer.transform(x[key].values.reshape(-1, 1))

    new_y = y_transformer.transform(y.labels.ravel())
    return dict(
        x=new_x,
        y=new_y,
    )
    new_x = x.copy()
    new_y = y.copy()
    for key, transformer in x_transformers.items():
        new_x[key] = transformer()(x[key])
        


def split(
    data: pd.DataFrame,
    targets: pd.Series,
    test_ratio: float,
    seed: int
) -> Dict:
    assert len(data) > 0
    assert len(targets) > 0
    assert len(targets) == len(data)
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
        data, 
        targets,
        random_state=seed,
        test_size=test_ratio,
        stratify=targets)

    return dict(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y
    )


def get_data() -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    data, targets = sklearn.datasets.fetch_kddcup99(
        percent10=True, 
        as_frame=True, 
        return_X_y=True
    )
    return dict(
        data=data,
        targets=targets
    )


def clean(
    data: pd.DataFrame, 
    targets: pd.Series
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:

    df = data.drop(columns=['protocol_type', 'service', 'flag'])
    df = df.drop(columns=['land', 'logged_in', 'is_host_login', 'is_guest_login'])
    df = df.drop(columns=[
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
        'dst_host_srv_rerror_rate', 'wrong_fragment'])

    # targets
    targets_df = targets.copy()
    smurf_mask = targets_df.labels == "b'smurf.'"
    normal_mask = targets_df.labels == "b'normal.'"

    # Filter down
    targets_df = targets_df[smurf_mask | normal_mask]
    df = df[smurf_mask | normal_mask]

    # Recompute masks as rows have changed 
    targets_df[targets_df.labels == "b'smurf.'"] = 1
    targets_df[targets_df.labels == "b'normal.'"] = 0

    return dict(
        data=df,
        targets=targets_df
    )


'''
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


def get_data() -> pd.DataFrame:

    data = sklearn.datasets.kddcup99.fetch_kddcup99(percent10=True))

    return data
        
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


"""
def get_data() -> Dict[str, Union[np.ndarray, Any]]:
    (ds_train, ds_test), ds_info = tfds.load(
        'kddcup99',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True,
    )
    result = convert_to_numpy(ds_train)
    x_train, y_train = result['X'], result['y']

    result = convert_to_numpy(ds_test)
    x_test, y_test = result['X'], result['y']
    return dict(
        train_x=x_train,
        train_y=y_train,
        test_x=x_test,
        test_y=y_test,
        ds_info=ds_info,
    )
    

def feature_engineer(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    # TODO

        #x=x/255.,
    return dict(
        x=x,
        y=y,
    )


def convert_to_numpy(ds):
    # TODO
    array = np.vstack(tfds.as_numpy(ds))
    X = array[:, 0]
    #X = [img.flatten() for img in X]
    #X = np.vstack(img.flatten() for img in X)
    X = np.vstack([img.flatten() for img in X])
    y = array[:, 1].astype(np.uint8)

    return dict(
        X=X,
        y=y,
    )
"""
'''