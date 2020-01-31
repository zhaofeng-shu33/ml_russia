import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def score(estimator, X_test, Y_test):
    Y_predict = estimator.predict(X_test)
    return mean_squared_error(Y_test, Y_predict)

def linear_fit(X, Y):
    reg = linear_model.LinearRegression()
    scores = cross_val_score(reg, X, Y, cv=5, scoring=score)
    return np.mean(scores)

def read_raw_data():
    df = pd.read_csv('flats_moscow_a.csv')
    return df

def _normalize(df, column_name):
    min_max_scaler = MinMaxScaler()
    df[column_name] = min_max_scaler.fit_transform(df[[column_name]])

def _remove_abnormal(df, i):
    # to do added
    return

def _get_data_and_label(df):
    Y = df['price'].values
    df.drop('price', axis=1, inplace=True)
    df.drop(df.columns[0], axis=1, inplace=True)
    X = df.values
    return (X, Y)

def preprocessing(df, K=4):
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
    categorical_class = ['brick', 'metro', 'floor',
        'code','owners', 'parking', 'rating', 'murder', 'class']
    for i in categorical_class:
        df[i] = df[i].astype('category')
    # discretize some features
    df['lastownage'] = pd.cut(df.lastownage, bins=4)
    # remove entry with missing values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # normalize continuous features
    continuous_class = ['kitsp', 'dist', 'walk',
        'livesp', 'totsp']
    for i in continuous_class:
        _normalize(df, i)
    _normalize(df, 'price')
    # assume each continuous features conforms to standard normal
    # distribution, treat observations with feature larger than 3
    # sigma as abnormal and remove them in later computation
    for i in continuous_class:
        _remove_abnormal(df, i)
    # implement one-hot encoding
    df_new = pd.get_dummies(df)
    # finally, we return the numpy array as data, price as label
    return _get_data_and_label(df_new)

def model(X, Y):
    '''
    evaluated the model which using X to predict Y
    return the MSE loss
    '''
    return linear_fit(X, Y)

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
