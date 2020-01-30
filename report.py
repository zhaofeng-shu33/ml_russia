import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_raw_data():
    df = pd.read_csv('flats_moscow_a.csv')
    return df

def _normalize(df, column_name):
    min_max_scaler = MinMaxScaler()
    df[column_name] = min_max_scaler.fit_transform(df[[column_name]])

def preprocessing(df, K=4):
    '''
        K: number of bins used to discretize lastownage
    '''
    # convert columns with category data with right type
    categorical_class = ['brick', 'metro', 'floor',
        'code','owners', 'parking', 'rating', 'murder', 'class']
    for i in categorical_class:
        df[i] = df[i].astype('category')
    # discretize some features
    df['lastownage'] = pd.cut(df.lastownage, bins=4)
    # remove entry with missing values
    df.dropna()
    # normalize continuous features
    continuous_class = ['kitsp', 'dist', 'walk',
        'livesp', 'totsp']
    for i in continuous_class:
        _normalize(df, i)
    # assume each continuous features conforms to standard normal
    # distribution, treat observations with feature larger than 3
    # sigma as abnormal and remove them in later computation

def main():
    df = read_raw_data()
    preprocessing(df)

if __name__ == '__main__':
    main()
