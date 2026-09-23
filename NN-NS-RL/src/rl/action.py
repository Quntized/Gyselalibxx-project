import gym
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict
class NormalizedActions(gym.ActionWrapper):
    def action(self, action: torch.Tensor):
        low_bound = self.action_space['low']
        upper_bound = self.action_space['upper']
        
        low_bound_ = torch.tensor(low_bound, device=action.device, dtype=action.dtype).unsqueeze(0).unsqueeze(0)
        upper_bound_ = torch.tensor(upper_bound, device=action.device, dtype=action.dtype).unsqueeze(0).unsqueeze(0)
        
        action = low_bound_ + (action + 1.0) * 0.5 * (upper_bound_ - low_bound_)
        
        return action

    def reverse_action(self, action: torch.Tensor):
        low_bound = self.action_space['low']
        upper_bound = self.action_space['upper']
        
        low_bound_ = torch.tensor(low_bound, device=action.device, dtype=action.dtype).unsqueeze(0).unsqueeze(0)
        upper_bound_ = torch.tensor(upper_bound, device=action.device, dtype=action.dtype).unsqueeze(0).unsqueeze(0)
        
        action = 2 * (action - (low_bound_ + upper_bound_)/2) / (upper_bound_ - low_bound_)
        
        return action



class ClippingActions(gym.ActionWrapper):
    
    def action(self, action: torch.Tensor):
        low_bound = self.action_space['rate-low']
        upper_bound = self.action_space['rate-upper']
        
        low_bound = torch.tensor(low_bound, device=action.device, dtype=action.dtype).view(1, 1, -1)
        upper_bound = torch.tensor(upper_bound, device=action.device, dtype=action.dtype).view(1, 1, -1)
        
        action_prev = self.get_action()[:, -1, :].unsqueeze(1)
        
        action = torch.clip(action, action_prev + low_bound, action_prev + upper_bound)
        
        return action
        
    def reverse_action(self, action: torch.Tensor):
        return action