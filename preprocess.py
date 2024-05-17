import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression


def setup_dataframe_1(dataset_address):
    df = pd.read_csv(dataset_address)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    X_values = X.values
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X_values)
    X = pd.DataFrame(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def setup_dataframe_2(dataset_address):
    df = pd.read_csv(dataset_address)
    # df = df.drop(columns=[])
    # df = pd.get_dummies(df, columns=[])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    X_values = X.values
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X_values)
    X = pd.DataFrame(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def setup_dataframe_3():
    ionosphere = fetch_ucirepo(id=52)

    X = ionosphere.data.features
    y = ionosphere.data.targets

    class_mapping = {'b': 0, 'g': 1}
    y.loc[:, 'Class'] = y['Class'].map(class_mapping)

    X_values = X.values
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X_values)
    X = pd.DataFrame(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def setup_dataframe_4(dataset_address):
    dataset_file = arff.loadarff(dataset_address)
    df = pd.DataFrame(dataset_file[0])

    df = df.applymap(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else x)
    df['Result'] = df['Result'].map({-1: 1, 0: 2, 1: 3})

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def setup_dataframe_5(dataset_address):
    dataset_file = arff.loadarff(dataset_address)
    df = pd.DataFrame(dataset_file[0])

    df = df.applymap(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else x)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    X_values = X.values
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X_values)
    X = pd.DataFrame(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test
