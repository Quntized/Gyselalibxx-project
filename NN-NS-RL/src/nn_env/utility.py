from sklearn.model_selection import train_test_split
import random, torch, os
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
from typing import Literal, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def preparing_0D_dataset(
    df: pd.DataFrame,
    cols_0D: List,
    cols_ctrl: List,
    scaler: Literal['Robust', 'Standard', 'MinMax'] = 'Robust'
):
    # nan interpolation
    df.interpolate(method='linear', limit_direction='forward')
    
    # [REMOVED: df = df[df.shot > 19000] - This was deleting all your Vlasov data!]
    ts_cols = cols_0D + cols_ctrl

    # float type
    for col in ts_cols:
        df[col] = df[col].astype(np.float32)
    
    # shot sampling
    shot_list = np.unique(df.shot.values)
    
    
    print("# of shot : {}".format(len(shot_list)))
    
    # train / valid / test data split
    from sklearn.model_selection import train_test_split
    
    shot_train, shot_test = train_test_split(shot_list, test_size=0.2, random_state=42)
    shot_train, shot_valid = train_test_split(shot_train, test_size=0.25, random_state=42)
    
    df_train = df[df.shot.isin(shot_train)].copy().reset_index(drop=True)
    df_valid = df[df.shot.isin(shot_valid)].copy().reset_index(drop=True)
    df_test = df[df.shot.isin(shot_test)].copy().reset_index(drop=True)
    
    if scaler == 'Standard':
        scaler_0D = StandardScaler()
        scaler_ctrl = StandardScaler()
    elif scaler == 'Robust':
        scaler_0D = RobustScaler()
        scaler_ctrl = RobustScaler()
    elif scaler == 'MinMax':
        scaler_0D = MinMaxScaler()
        scaler_ctrl = MinMaxScaler()
  
    # scaler training
    scaler_0D.fit(df_train[cols_0D].values)
    scaler_ctrl.fit(df_train[cols_ctrl].values)
        
    return df_train, df_valid, df_test, scaler_0D, scaler_ctrl
def get_range_of_output(df : pd.DataFrame, cols_0D : List):
    
    range_info = {}

    for col in cols_0D:
        min_val = df[col].min()
        max_val = df[col].max()
        range_info[col] = [min_val, max_val]
    
    return range_info