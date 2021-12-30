import pandas as pd
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from typing import Dict, Union
from sklearn.base import TransformerMixin


def build_feature_engineer(
    x: pd.DataFrame, 
    y: pd.Series
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:

    # X
    x_transformers = {}
    for col in CONTINUOUS_FEATURES:
        # TODO handle other col types
        #if x[col].dtype == np.float64:
        scaler = StandardScaler()
        scaler.fit(x[col].values.reshape(-1, 1))
        x_transformers[col] = scaler
    
    for col in ENCODE_FEATURES:
        # TODO handle other col types
        #if x[col].dtype == np.float64:
        encoder = OneHotEncoder(drop='first', sparse=False)
        encoder.fit(x[col].values.reshape(-1, 1))
        x_transformers[col] = encoder

    # Y
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
) -> Dict:
    new_x = x.copy()
    new_x.reset_index(drop=True, inplace=True) # reset index due to shuffling during split
    
    for key, transformer in x_transformers.items():
        transformer = transformer() # TODO why wasn't this needed for in memory ds
        encodings = transformer.transform(new_x[key].values.reshape(-1, 1))
        encodings_df = pd.DataFrame(encodings, columns=[key + '_' + str(i) for i in range(encodings.shape[1])])
        encodings_df.reset_index()
        new_x = pd.concat([new_x, encodings_df], axis=1)
        new_x.reset_index()
        new_x = new_x.drop(columns=[key])
        new_x.reset_index()

    new_y = y_transformer.transform(y.values.ravel())
    return dict(
        x=new_x,
        y=new_y,
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

LABEL = 'class'
ENCODE_FEATURES = [
        #"workclass",
        #"sex",
        #"marital-status",
        #"occupation",
        #"race" TODO
    ]

LATER = [
    #"native-country",
    #"capital-gain",
    #"capital-loss", # TODO maybe use this
    #"race",
    #"relationship",
    #"education",
    #"fnlwgt"
]

CONTINUOUS_FEATURES = [
    "age",
    "education-num",
    "hours-per-week",
]


def clean(in_df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataframe.

    Args:
        df (pandas.DataFrame): The dataframe to clean.

    Returns:
        pandas.DataFrame: The cleaned dataframe.

    """
    df = in_df.copy()

    #le = LabelEncoder()
    #le.fit(df['class'])
    #labels = le.transform(df['class'])


    #df = df.drop(columns=['class'])

    # TODO scaling in feature engineering

    #df = df.drop(columns=ez_features)


    #df = pd.get_dummies(df, drop_first=True, columns=ENCODE_FEATURES)
    df = df.drop(columns=LATER)

    #df['class'] = labels

    return df