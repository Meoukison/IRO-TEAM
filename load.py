import pandas as pd, numpy as np
D='../data/'
def load(split):
    X=pd.read_csv(D+f'X_{split}.csv').merge(pd.read_csv(D+f'official_on_off_{split}.csv').rename(columns={'on':'on_o','off':'off_o'}),on='Index').merge(pd.read_csv(D+f'rater_id_{split}.csv'),on='Index')
    if split=='train': X=X.merge(pd.read_csv(D+'y_train.csv'),on='Index')
    X=X.sort_values(['patient_id','age']).reset_index(drop=True)
    X['dur']=X.age-X.age_at_diagnosis
    return X
