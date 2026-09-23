import numpy as np
import random, os, pickle
import torch
from src.rl.SumTree import SumTree
from collections import namedtuple
from typing import Optional, Union, List

Transition = namedtuple(
    'Transition',
    ('state', 'action','next_state','reward','done')
)

class PER(object):
    
    # initial values
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001    
    
    def __init__(self, capacity : int):
        self.tree = SumTree(capacity)
        self.capacity = capacity
    
    def _get_priority(self, err):
        if isinstance(err, torch.Tensor):
            err = err.detach().cpu().numpy()
        return (np.abs(err) + self.e) ** self.a
            
    def push(self, *args):
        prios = self.tree.max() if self.tree.n_entries > 0 else 1.0
        
        clean_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                clean_args.append(arg.detach().cpu())
            else:
                clean_args.append(arg)
                
        sample = Transition(*clean_args)
        self.tree.add(prios, sample)
        
    def sample(self, batch_size : int):
        batch = []
        indice = []
        segment = self.tree.total() / batch_size
        priorities = []
        
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i+1)
            
            while True:
                s = random.uniform(a,b)
                (idx, p, data) = self.tree.get(s)
                
                if not isinstance(data, int):
                    break
            
            priorities.append(p)
            batch.append(data)
            indice.append(idx)

        sampling_probs = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        is_weight /= is_weight.max()
        
        is_weight = torch.tensor(is_weight, dtype=torch.float32)
        
        return batch, indice, is_weight

    def update(self, idx : Union[int, List, np.ndarray], err : Union[float, List, np.ndarray, torch.Tensor]):
        if isinstance(idx, (list, np.ndarray)):
            if isinstance(err, torch.Tensor):
                err = err.detach().cpu().numpy()
            for i, e in zip(idx, err):
                p = self._get_priority(e)
                self.tree.update(i, p)
        else:
            p = self._get_priority(err)
            self.tree.update(idx, p)
            
    def save_buffer(self, env_name : str, tag : str = "", save_path : Optional[str] = None):
            
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/', exist_ok=True)

        if save_path is None:
            save_path = "checkpoints/PER_{}_{}".format(env_name, tag)
            
        print("Process : saving buffer to {}".format(save_path))
        
        with open(save_path, "wb") as f:
            pickle.dump(self.tree, f)
        
    def load_buffer(self, save_path : str):
        print("Process : loading buffer from {}".format(save_path))
        
        with open(save_path, 'rb') as f:
            self.tree = pickle.load(f)
            
    def clear(self):
        self.tree._clear()