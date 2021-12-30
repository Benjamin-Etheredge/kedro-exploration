import pandas as pd
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import Dict, Union, Any
#from sklearn import datasets
import tensorflow_datasets as tfds
import numpy as np


def get_sub_data() -> Dict[str, Union[np.ndarray, Any]]:
    data, targets = sklearn.datasets.load_digits(
        as_frame=True,
        return_X_y=True
    )

    return dict(
        data=data.values,
        targets=targets.values,
    )


def split(
    data: pd.DataFrame,
) -> Dict:
    assert len(data) > 0
    assert len(data.columns) == 65

    return dict(
        x=data.drop(columns=['label']).values,
        y=data['label'].values,
    )


###############################################################################

def get_data() -> Dict[str, Union[np.ndarray, Any]]:
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
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
    # TODO probably fine 
    value_range = np.max(x)

    return dict(
        x=x/value_range,
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
