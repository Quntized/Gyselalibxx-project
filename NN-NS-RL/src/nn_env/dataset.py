import os
import numpy as np
import pandas as pd
import torch
import random
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional, Dict, List, Union, Literal

DEFAULT_COLS = [
    'density_max', 'density_mean', 'temperature_max', 'velocity_max'
]
DEFAULT_CTRL_COLS = [
    'beam_temp', 'beam_vel', 'perturb_amp'
]


class DatasetFor0D(Dataset):
    def __init__(
        self,
        ts_data: pd.DataFrame,
        seq_len: int = 4,
        seq_len_ctrl: int = 5,
        pred_len: int = 1,
        cols_0D: List = DEFAULT_COLS,
        cols_ctrl: List = DEFAULT_CTRL_COLS,
        interval: int = 3,
        scaler_0D=None,
        scaler_ctrl=None,
    ):
        self.ts_data = ts_data
        self.seq_len = seq_len
        self.seq_len_ctrl = seq_len_ctrl
        self.cols_0D = cols_0D
        self.cols_ctrl = cols_ctrl
        self.pred_len = pred_len
        self.interval = interval
        self.scaler_0D = scaler_0D
        self.scaler_ctrl = scaler_ctrl
        self.input_indices = []
        self.target_indices = []
        self.shot_list = np.unique(self.ts_data.shot.values).tolist()
        self.preprocessing()
        self._generate_index()

    def preprocessing(self):
        self.ts_data[self.cols_ctrl] = self.ts_data[self.cols_ctrl].fillna(0)
        shot_ignore = []
        for shot in tqdm(self.shot_list, desc='extract null data'):
            df_shot = self.ts_data[self.ts_data.shot == shot]
            null_check = df_shot[self.cols_0D + self.cols_ctrl].isna().sum()
            for c in null_check:
                if c > 0.5 * len(df_shot):
                    shot_ignore.append(shot)
                    break

        shot_list_new = [s for s in self.shot_list if s not in shot_ignore]
        self.shot_list = shot_list_new

        for shot in tqdm(self.shot_list, desc='replace nan value'):
            df_shot = self.ts_data[self.ts_data.shot == shot].copy()
            self.ts_data.loc[self.ts_data.shot == shot, self.cols_0D] = df_shot[self.cols_0D].ffill()

        if self.scaler_0D:
            self.ts_data[self.cols_0D] = self.scaler_0D.transform(self.ts_data[self.cols_0D])
        if self.scaler_ctrl:
            self.ts_data[self.cols_ctrl] = self.scaler_ctrl.transform(self.ts_data[self.cols_ctrl])

        self.ts_data = self.ts_data.reset_index(drop=True)

    def _generate_index(self):
        for shot in tqdm(self.shot_list, desc='Dataset Indices generation'):
            input_indices = []
            target_indices = []
            df_shot = self.ts_data[self.ts_data.shot == shot].reset_index(drop=True)
            n = len(df_shot)
            idx_last = n - self.seq_len - self.pred_len
            if idx_last < 0:
                continue

            idx = 0
            while idx <= idx_last:
                input_indices.append((shot, idx))
                target_indices.append((shot, idx + self.seq_len))
                idx += self.interval

            self.input_indices.extend(input_indices)
            self.target_indices.extend(target_indices)

    def __getitem__(self, idx: int):
        shot, input_pos = self.input_indices[idx]
        _, target_pos = self.target_indices[idx]
        df_shot = self.ts_data[self.ts_data.shot == shot].reset_index(drop=True)

        data_0D = df_shot[self.cols_0D].iloc[input_pos + 1: input_pos + 1 + self.seq_len].values
        data_ctrl = df_shot[self.cols_ctrl].iloc[input_pos + 1: input_pos + 1 + self.seq_len_ctrl].values


        target_0D = df_shot[self.cols_0D].iloc[target_pos: target_pos + self.pred_len].values
        target_ctrl = df_shot[self.cols_ctrl].iloc[target_pos: target_pos + self.pred_len].values

        label = df_shot[self.cols_0D].iloc[target_pos + 1: target_pos + 1 + self.pred_len].values


        data_0D = torch.from_numpy(data_0D).float()
        data_ctrl = torch.from_numpy(data_ctrl).float()
        target_0D = torch.from_numpy(target_0D).float()
        target_ctrl = torch.from_numpy(target_ctrl).float()
        label = torch.from_numpy(label).float()

        return data_0D, data_ctrl, target_0D, target_ctrl, label

    def __len__(self):
        return len(self.input_indices)