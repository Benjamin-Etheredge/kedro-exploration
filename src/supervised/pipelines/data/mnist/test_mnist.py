import pandas as pd
import numpy as np
from .nodes import get_data

def validate_data(train_x, train_y, test_x, test_y):
    assert train_x.shape == (60000, 784)
    assert train_y.shape == (60000,)
    assert test_x.shape == (10000, 784)
    assert test_y.shape == (10000,)

    for y in [train_y, test_y]:
        for value in y:
            assert value in range(10)
            assert type(value) == np.uint8
    


# NOTE: might as well auto test to be sure columns are right values
def test_get_data():
    """
    Args:
        df (pandas.DataFrame): The dataframe to test.

    Returns:
        (pandas.DataFrame): The testing dataframe.
    """
    result = get_data()
    assert type(result) == dict

    assert len(result.keys()) == 5
    for key in ["train_x", "train_y", "test_x", "test_y"]:
        assert key in result.keys()
        assert type(result[key]) == np.ndarray

    train_x = result['train_x']
    train_y = result['train_y']
    test_x = result['test_x']
    test_y = result['test_y']
    validate_data(train_x, train_y, test_x, test_y)


#from kedro.io import DataCatalog 
#def test_catalog():
    #train_x = DataCatalog.load('mnist_train_x')
    #train_y = DataCatalog.load('mnist_train_y')
    #test_x = DataCatalog.load('mnist_test_x')
    #test_y = DataCatalog.load('mnist_test_y')



    