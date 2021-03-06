import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import svm

def predict(csv_file, model_file):
    # read the data from csv_file 
    # and model from model_file
    df = pd.read_csv(csv_file)
    X, _ = preprocessing(df) # not necessary of the same scale
    with open(model_file, 'rb') as f:
        reg = pickle.load(f)
    return reg.predict(X)

def score(estimator, X_test, Y_test):
    Y_predict = estimator.predict(X_test)
    return np.sqrt(mean_squared_error(Y_test, Y_predict))

def svr_fit(X, Y, _C=80, _kernel='linear'):
    reg = svm.SVR(C=_C, kernel=_kernel)
    scores = cross_val_score(reg, X, Y, cv=5, scoring=score)
    return np.mean(scores)

def read_raw_data():
    df = pd.read_csv('flats_moscow_a.csv')
    return df

def _normalize(df, column_name):
    min_max_scaler = MinMaxScaler()
    df[column_name] = min_max_scaler.fit_transform(df[[column_name]])

def _get_data_and_label(df):
    try:
        Y = df['price'].values
        df.drop('price', axis=1, inplace=True)
    except:
        Y = None
    df.drop(df.columns[0], axis=1, inplace=True)
    X = df.values
    return (X, Y)

def preprocessing(df, K=4, n_components=30):
    '''
        Parameters
        ----------
        df: Pandas dataframe
        K: number of bins used to discretize lastownage

        Returns
        -------
        X: array-like object, number of samples X number of features
        Y: predicted values
    '''
    # convert columns with category data with right type
    categorical_class = ['brick', 'metro', 'code', 'owners',
                         'parking', 'rating', 'murder', 'class']
    for i in categorical_class:
        df[i] = df[i].astype('category')
    # remove unrelated features
    df.drop('lastownage', axis=1, inplace=True)
    # remove entry with missing values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # normalize continuous features
    continuous_class = ['kitsp', 'dist', 'walk', 'floor',
                        'livesp', 'totsp']
    for i in continuous_class:
        _normalize(df, i)
    # implement one-hot encoding
    df_new = pd.get_dummies(df)
    # finally, we return the numpy array as data, price as label
    X, Y = _get_data_and_label(df_new)
    return (X, Y)

def model(X, Y):
    '''
    evaluated the model which using X to predict Y
    return the MSE loss
    '''
    return svr_fit(X, Y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quite', default=False, const=True, nargs='?')
    args = parser.parse_args()
    df = read_raw_data()
    X, Y = preprocessing(df)
    mse_loss = model(X, Y)
    if not args.quite:
        print(mse_loss)

if __name__ == '__main__':
    main()
