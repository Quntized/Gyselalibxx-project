''' Custom Environment for tokamak operation control
    This enviornment is based on Neural Network which predict the next state of the tokamak plasma from being trained by KSTAR dataset
    Current version : prediction of the 0D parameters such as beta, q95, li, ne
    Later version : prediction of the 0D paramters + Magnetic surface (Grad-Shafranov solver)
    Reference
    - https://www.gymlibrary.dev/content/environment_creation/
    - https://github.com/openai/gym-soccer/blob/master/gym_soccer/envs/soccer_env.py
    - https://medium.com/cloudcraftz/build-a-custom-environment-using-openai-gym-for-reinforcement-learning-56d7a5aa827b
    - https://github.com/notadamking/Stock-Trading-Environment
    - https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
'''
import gymnasium as gym
import os, subprocess, time, signal, gc
import torch
import torch.nn as nn
import pandas as pd
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from src.rl.rewards import RewardSender
from typing import Dict, Optional, Literal, List
from src.config import Config
import logging
class NeuralEnv(gym.Env):
    metadata = {'render.modes':['human']} 
    
    def __init__(
        self, 
        predictor : nn.Module, 
        device : str, 
        reward_sender, 
        seq_len : int, 
        pred_len : int, 
        range_info : Dict, 
        t_terminal : float = 4.0, 
        dt : float = 0.01, 
        cols_control = None,
        limit_ctrl_rate : bool = False,
        rate_range_info : Optional[Dict] = None,
        scaler_0D = None,
        scaler_ctrl = None,
        use_stochastic : bool = False,
        noise_mean_0D : float = 0,
        noise_std_0D : float = 1.0,
        noise_mean_ctrl : float = 0,
        noise_std_ctrl : float = 1.0,
        noise_scale_0D : float = 1.0,
        noise_scale_ctrl : float = 1.0,
        gamma : float = 0.995
        ):
        
        super().__init__()
        
        # predictor : output as next state of plasma
        self.predictor = predictor.to(device)
        
        # scaler
        self.scaler_0D = scaler_0D
        self.scaler_ctrl = scaler_ctrl
        
        self.predictor.eval()
        self.device = device
        
        # reward engineering
        self.reward_sender = reward_sender
        
        # initialize state, action, targets, done, time released
        self.done = False
        self.init_state = None
        self.init_action = None
        self.target_0D = None
        self.target_ctrl = None
        
        # current state 
        self.t_released = 0 
        self.current_state = None
        self.current_action = None
        
        # information for virtual operation
        self.dt = dt 
        self.t_terminal = t_terminal 
        self.seq_len = seq_len
        self.pred_len = pred_len    
        
        # range information about action space
        self.action_space = {
            'low' : [range_info[col][0] for col in range_info.keys()],
            'upper' : [range_info[col][1] for col in range_info.keys()],
        }
        
        self.original_shot = None
        self.cols_control = cols_control
        self.limit_ctrl_rate = limit_ctrl_rate
        
        if limit_ctrl_rate:
            self.action_space['rate-low'] = [rate_range_info[col][0] for col in rate_range_info.keys()]
            self.action_space['rate-upper'] = [rate_range_info[col][1] for col in rate_range_info.keys()]
            
        # stochastic
        self.use_stochastic = use_stochastic
        self.noise_mean_0D = noise_mean_0D
        self.noise_mean_ctrl = noise_mean_ctrl
        self.noise_std_0D = noise_std_0D
        self.noise_std_ctrl = noise_std_ctrl
        self.noise_scale_0D = noise_scale_0D
        self.noise_scale_ctrl = noise_scale_ctrl
        self.gamma = gamma
        
    def update_ls_weight(self, weights:List):
        self.reward_sender.update_target_weight(weights)
    
    def add_noise(self, x : torch.Tensor, choice : Literal['state','action']):
        if choice == 'state':
            noise = torch.ones_like(x).to(x.device) * self.noise_mean_0D + torch.randn(x.size()).to(x.device) * self.noise_std_0D
            noise *= self.noise_scale_0D
            x += noise
        elif choice == 'action':
            noise = torch.ones_like(x).to(x.device) * self.noise_mean_ctrl + torch.randn(x.size()).to(x.device) * self.noise_std_ctrl
            noise *= self.noise_scale_ctrl
            x += noise
        return x
        
    def load_shot_info(self, df : pd.DataFrame):
        self.original_shot = df
    
    def update_init_state(self, init_state : torch.Tensor, init_action : torch.Tensor, target_0D : torch.Tensor, target_ctrl : torch.Tensor):
        if init_state.ndim == 2: init_state = init_state.unsqueeze(0)
        if init_action.ndim == 2: init_action = init_action.unsqueeze(0)
        if target_0D.ndim == 2: target_0D = target_0D.unsqueeze(0)
        if target_ctrl.ndim == 2: target_ctrl = target_ctrl.unsqueeze(0)
            
        self.init_state = init_state
        self.init_action = init_action
        self.target_0D = target_0D
        self.target_ctrl = target_ctrl
        
        self.current_state = init_state
        self.current_action = init_action
        
    def update_state(self, next_state : torch.Tensor):
        state = self.current_state
        if next_state.ndim == 2:
            next_state = next_state.unsqueeze(0)
            
        next_state = torch.cat([state, next_state], dim=1)
        self.current_state = next_state[:,-self.seq_len:,:]
        self.t_released += self.dt * self.pred_len
        
    def update_action(self, next_action : torch.Tensor):
        action = self.current_action
        if next_action.ndim == 2:
            next_action = next_action.unsqueeze(0)
            
        next_action = torch.cat([action, next_action], dim=1)
        self.current_action = next_action[:,-self.seq_len-self.pred_len:,:]
        
    def get_state(self):
        return self.current_state

    def get_action(self):
        return self.current_action
    
    def reset(self):
        self.done = False
        self.current_state = self.init_state
        self.current_action = self.init_action
        self.t_released = 0
        return self.current_state

    def step(self, action : torch.Tensor):
        state = self.get_state()
        
        if state.ndim == 2:
            state = state.unsqueeze(0)
        if action.ndim == 2:
            action = action.unsqueeze(0)
            
        self.update_action(action)
        action = self.get_action()
        
        if self.use_stochastic is True:
            state = self.add_noise(state, 'state')
            action = self.add_noise(action, 'action')
            
        # FIXED: Passing all 4 required arguments to the predictor
        next_state = self.predictor(
            state.to(self.device), 
            action.to(self.device),
            self.target_0D.to(self.device),
            self.target_ctrl.to(self.device)
        ).detach().cpu()
        
        reward = self.reward_sender(next_state)
        
        self.check_terminal_state(next_state)
        self.update_state(next_state)
        
        return next_state, reward, self.done, {}

    def get_reward(self, next_state : torch.Tensor):
        return self.reward_sender(next_state)
    
    def check_terminal_state(self, next_state : torch.Tensor):
        if torch.isnan(next_state).sum() > 0:
            self.done = True
        if self.t_released >= self.t_terminal:
            self.done = True
            
    def close(self):
        if self.current_state is not None: self.current_state.cpu()
        if self.init_state is not None: self.init_state.cpu()
        if self.current_action is not None: self.current_action.cpu()
        if self.init_action is not None: self.init_action.cpu()
        if self.target_0D is not None: self.target_0D.cpu()
        if self.target_ctrl is not None: self.target_ctrl.cpu()
        
        self.predictor.cpu()
        
        self.current_state = None
        self.init_state = None
        self.current_action = None
        self.init_action = None
        self.target_0D = None
        self.target_ctrl = None
        
        gc.collect()
        torch.cuda.empty_cache()