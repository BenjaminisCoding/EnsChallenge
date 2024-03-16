import pandas as pd
import numpy as np

def round(x):

    if x - int(x) <= .5: return int(x)
    else: return int(x) + 1

def load_data():

    train_home_team_statistics_df = pd.read_csv('./datas_final/train_home_team_statistics_df.csv', index_col=0)
    train_away_team_statistics_df = pd.read_csv('./datas_final/train_away_team_statistics_df.csv', index_col=0)

    train_scores = pd.read_csv('./datas_final/Y_train.csv', index_col=0)

    train_home = train_home_team_statistics_df.iloc[:,2:]
    train_away = train_away_team_statistics_df.iloc[:,2:]

    train_home.columns = 'HOME_' + train_home.columns
    train_away.columns = 'AWAY_' + train_away.columns

    train_data =  pd.concat([train_home,train_away],join='inner',axis=1)
    train_scores = train_scores.loc[train_data.index]

    train_data = train_data.replace({np.inf:np.nan,-np.inf:np.nan})

    labels = train_scores.copy()
    labels['class'] = labels.idxmax(axis=1)
    label_mapping = {'HOME_WINS': 0, 'DRAW': 1, 'AWAY_WINS': 2}
    labels['class'] = labels['class'].map(label_mapping)
    labels = labels.drop(['HOME_WINS', 'DRAW', 'AWAY_WINS'], axis=1)

    return train_data, labels