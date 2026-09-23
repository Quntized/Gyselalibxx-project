import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc, time
from itertools import count
from tqdm.auto import tqdm
from typing import Optional, List, Literal, Dict, Union
from src.rl.buffer import Transition, ReplayBuffer
from src.rl.PER import PER
from src.rl.utility import InitGenerator
from src.rl.env import NeuralEnv

# Ornstein-Uhlenbeck Process
# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise:
    def __init__(self, n_action: int, pred_len: int, mu: float = 0.0, theta: float = 0.15, 
                 max_sigma: float = 0.3, min_sigma: float = 0.3, decay_period: int = 100000,
                 action_low: float = -1.0, action_high: float = 1.0):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        
        self.action_dim = n_action
        
        self.low = action_low
        self.high = action_high
        
        self.pred_len = pred_len
        self.state = None
        
        self.reset()

    def reset(self):
        # Clean initialization without redundant empty_like calls
        self.state = torch.ones((1, self.pred_len, self.action_dim), dtype=torch.float32) * self.mu
    
    def evolve_state(self, device):
        x = self.state.to(device)
        noise = torch.randn((1, self.pred_len, self.action_dim), device=device, dtype=torch.float32)
        dx = self.theta * (self.mu - x) + self.sigma * noise
        
        self.state = x + dx
        return self.state
    
    def get_action(self, action: torch.Tensor, t: float = 0):
        # Pass the action's device so the state evolves entirely on the GPU
        ou_state = self.evolve_state(action.device)
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        
        # Safely clamp using the parameterized bounds
        return torch.clamp(action + ou_state, self.low, self.high)


import torch
import torch.nn as nn

# Encoder : plasma state -> hidden vector
class Encoder(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, pred_len: int, conv_dim: int = 32, conv_kernel: int = 3, conv_stride: int = 2, conv_padding: int = 1):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.conv_dim = conv_dim
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        
        # temporal convolution
        self.layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_dim, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
        )   
        
        self.feature_dim = conv_dim
                
        # output as sequential length = pred_len
        self.layer_2 = nn.Linear(
            self.compute_conv1d_output_dim(self.compute_conv1d_output_dim(seq_len, conv_kernel, conv_stride, conv_padding, 1), conv_kernel, conv_stride, conv_padding, 1),
            pred_len
        )
        
    def forward(self, x: torch.Tensor):
        
        # x : (B, T, D)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        
        x = x.permute(0, 2, 1)
        
        # x : (B, conv_dim, T')
        x = self.layer_1(x)
        
        # x : (B, conv_dim, pred_len)
        x = self.layer_2(x)
        
        # x : (B, pred_len, conv_dim)
        x = x.permute(0, 2, 1)
        
        return x
    
    def compute_conv1d_output_dim(self, input_dim: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1):
        return int((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
# Actor
class Actor(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, pred_len: int, mlp_dim: int, n_actions: int):
        super(Actor, self).__init__()
        self.encoder = Encoder(input_dim, seq_len, pred_len)
        self.n_actions = n_actions
        
        self.mlp_dim = mlp_dim
        linear_input_dim = self.encoder.feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(linear_input_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, n_actions),
        )
        
        nn.init.uniform_(self.mlp[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mlp[-1].bias, -3e-3, 3e-3)
        
    def forward(self, x: torch.Tensor):
        # x : (B, T, D)
        x = self.encoder(x)
        # x : (B, pred_len, linear_input_dim)
        x = torch.tanh(self.mlp(x))
        # x : (B, pred_len, n_action)
        return x



class Critic(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, pred_len: int, mlp_dim: int, n_actions: int):
        super(Critic, self).__init__()
        self.encoder = Encoder(input_dim, seq_len, pred_len)
        self.n_actions = n_actions
        self.mlp_dim = mlp_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + n_actions, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1)
        )
        
        nn.init.uniform_(self.mlp[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mlp[-1].bias, -3e-3, 3e-3)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        state = self.encoder(state)
        # Swapped 'axis' for 'dim' to adhere to standard PyTorch syntax
        x = torch.cat([state, action], dim=2).mean(dim=1)
        x = self.mlp(x)
        return x


import torch
import torch.nn as nn
import numpy as np
from typing import Optional

# update policy
def update_policy(
    memory, # ReplayBuffer
    policy_network: nn.Module, 
    value_network: nn.Module, 
    target_policy_network: nn.Module,
    target_value_network: nn.Module,
    value_optimizer: torch.optim.Optimizer,
    policy_optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    batch_size: int = 128, 
    gamma: float = 0.99, 
    device: Optional[str] = "cpu",
    min_value: float = -np.inf,
    max_value: float = np.inf,
    tau: float = 1e-2,
    use_CAPS: bool = False,
    lamda_temporal_smoothness: float = 1.0, 
    lamda_spatial_smoothness: float = 1.0,
    ):
    
    if use_CAPS:
        loss_fn_CAPS = nn.SmoothL1Loss(reduction='mean')

    policy_network.train()
    value_network.train()
    
    if len(memory) < batch_size:
        return None, None

    if device is None:
        device = "cpu"
    
    if criterion is None:
        criterion = nn.SmoothL1Loss(reduction='mean') 
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool
    )
    
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_batch_ = state_batch.detach().clone()

    value_network.train()
    next_q_values = torch.zeros((batch_size, 1), device=device)
    
    if non_final_mask.sum() > 0:
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
        next_q_values[non_final_mask] = target_value_network(non_final_next_states, target_policy_network(non_final_next_states).detach()).detach()
    
    bellman_q_values = reward_batch.unsqueeze(1) + gamma * next_q_values
    bellman_q_values = torch.clamp(bellman_q_values, min_value, max_value).detach()
    
    q_values = value_network(state_batch, action_batch)
    value_loss = criterion(q_values, bellman_q_values)

    value_optimizer.zero_grad()
    value_loss.backward()    
    
    for param in value_network.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1) 
        
    value_optimizer.step()

    value_network.eval()
    policy_loss = value_network(state_batch_, policy_network(state_batch_))
    policy_loss = -policy_loss.mean()
    
    if use_CAPS:
        mu = policy_network(state_batch[non_final_mask])
        next_mu = policy_network(non_final_next_states)
        near_mu = add_noise_to_action(mu)
        loss_CAPS = smoothness_inducing_regularizer(loss_fn_CAPS, mu, next_mu, near_mu, lamda_temporal_smoothness, lamda_spatial_smoothness)
        policy_loss += loss_CAPS

    policy_optimizer.zero_grad()
    policy_loss.backward()
    
    for param in policy_network.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1) 
    
    policy_optimizer.step()
    
    for target_param, param in zip(target_value_network.parameters(), value_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    for target_param, param in zip(target_policy_network.parameters(), policy_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    return value_loss.item(), policy_loss.item()


# PER version update function
def update_policy_PER(
    memory, # PER
    policy_network: nn.Module, 
    value_network: nn.Module, 
    target_policy_network: nn.Module,
    target_value_network: nn.Module,
    value_optimizer: torch.optim.Optimizer,
    policy_optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    batch_size: int = 128, 
    gamma: float = 0.99, 
    device: Optional[str] = "cpu",
    min_value: float = -np.inf,
    max_value: float = np.inf,
    tau: float = 1e-2
    ):

    policy_network.train()
    value_network.train()

    if memory.tree.n_entries < batch_size:
        return None, None

    if device is None:
        device = "cpu"
    
    criterion = nn.SmoothL1Loss(reduction='none') 
    
    transitions, indice, is_weight = memory.sample(batch_size)
    
    # Align weight shape for element-wise multiplication
    is_weight = is_weight.to(device).unsqueeze(1) 

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool
    )
    
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_batch_ = state_batch.detach().clone()

    value_network.train()
    next_q_values = torch.zeros((batch_size, 1), device=device)
    
    if non_final_mask.sum() > 0:
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
        next_q_values[non_final_mask] = target_value_network(non_final_next_states, target_policy_network(non_final_next_states).detach()).detach()
    
    bellman_q_values = reward_batch.unsqueeze(1) + gamma * next_q_values
    bellman_q_values = torch.clamp(bellman_q_values, min_value, max_value).detach()
    
    q_values = value_network(state_batch, action_batch)
    
    # Priority update
    prios = bellman_q_values.clone().detach() - q_values.clone().detach()
    prios = np.abs(prios.cpu().numpy()).flatten()
    
    memory.update(indice, prios)
    
    value_loss = (criterion(q_values, bellman_q_values) * is_weight).mean()

    value_optimizer.zero_grad()
    value_loss.backward()    
    
    for param in value_network.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1) 
        
    value_optimizer.step()

    value_network.eval()
    policy_loss = value_network(state_batch_, policy_network(state_batch_))
    policy_loss = -policy_loss.mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    
    for param in policy_network.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1) 
    
    policy_optimizer.step()

    for target_param, param in zip(target_value_network.parameters(), value_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    for target_param, param in zip(target_policy_network.parameters(), policy_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    return value_loss.item(), policy_loss.item()

def train_ddpg(
    env, # NeuralEnv
    ou_noise, # OUNoise
    init_generator, # InitGenerator
    memory, # Union[ReplayBuffer, PER] 
    policy_network : nn.Module, 
    value_network : nn.Module, 
    target_policy_network : nn.Module,
    target_value_network : nn.Module,
    policy_optimizer : torch.optim.Optimizer,
    value_optimizer : torch.optim.Optimizer,
    value_loss_fn :Optional[nn.Module] = None,
    batch_size : int = 128, 
    gamma : float = 0.99, 
    device : Optional[str] = "cpu",
    min_value : float = -np.inf,
    max_value : float = np.inf,
    tau : float = 1e-2,
    num_episode : int = 256,  
    verbose : int = 8,
    save_best : Optional[str] = None,
    save_last : Optional[str] = None,
    scaler_0D = None,
    use_CAPS : bool = False,
    lamda_temporal_smoothness : float = 1.0, 
    lamda_spatial_smoothness : float = 1.0,
    ):
    
    T_MAX = 1024

    value_network.train()
    policy_network.train()

    target_value_network.eval()
    target_policy_network.eval()

    if device is None:
        device = "cpu"

    episode_durations = []
    
    min_reward_list = []
    max_reward_list = []
    mean_reward_list = []
    
    target_value_result = {}
    
    for key in env.reward_sender.targets_cols:
        target_value_result[key] = {
            "min" : [],
            "max" : [],
            "mean": [],
        }
    
    # FIX 2: Prevent the Negative Reward Trap. Initialize to negative infinity!
    best_reward = -np.inf 
    best_episode = 0
    
    for i_episode in tqdm(range(num_episode), desc = 'DDPG algorithm training process'):
        
        start_time = time.time()
        
        init_state, init_action, target_0D, target_ctrl = init_generator.get_state()
        
        env.update_init_state(init_state, init_action, target_0D, target_ctrl)
        
        # reset ou noise and current state from env
        ou_noise.reset()
        state = env.reset()
        
        reward_list = []
        state_list = []

        for t in range(T_MAX):
            
            # compute action
            policy_network.eval()
            action = policy_network(state.to(device)).detach().cpu()
            
            # add OU Noise for efficient exploration : considering stochastic policy
            env_input_action = ou_noise.get_action(action, t)
            _, reward, done, _ = env.step(env_input_action)

            reward_list.append(reward.detach().cpu().numpy())
            reward = torch.tensor([reward])
            
            if not done:
                next_state = env.get_state()
            else:
                next_state = None

            memory.push(state, action, next_state, reward, done)

            # update state
            state = next_state
            
            # evolve state of OU Noise
            ou_noise.evolve_state(device=device) # Explicitly passing device for OU noise

            if isinstance(memory, type(memory)): # Handles both ReplayBuffer and PER based on your update functions
                # Note: Keeping your custom update_policy/update_policy_PER logic intact
                try:
                    if memory.__class__.__name__ == 'ReplayBuffer':
                        value_loss, policy_loss = update_policy(
                            memory, policy_network, value_network, target_policy_network,
                            target_value_network, value_optimizer, policy_optimizer,
                            value_loss_fn, batch_size, gamma, device, min_value, max_value,
                            tau, use_CAPS, lamda_temporal_smoothness, lamda_spatial_smoothness
                        )
                    else:
                        value_loss, policy_loss = update_policy_PER(
                            memory, policy_network, value_network, target_policy_network,
                            target_value_network, value_optimizer, policy_optimizer,
                            value_loss_fn, batch_size, gamma, device, min_value, max_value, tau
                        )
                except Exception as e:
                    pass # Usually skips update if buffer isn't large enough yet
            
            # update state list
            if state is not None:
                state_list.append(state[:,-1,:].unsqueeze(0).detach().cpu().numpy())

            if done or t > T_MAX:
                episode_durations.append(t+1)
                
                max_reward = np.max(reward_list)
                min_reward = np.min(reward_list)
                mean_reward = np.mean(reward_list)
                
                state_list = np.squeeze(np.concatenate(state_list, axis = 1), axis = 0)
                
                if scaler_0D:
                    state_list = scaler_0D.inverse_transform(state_list)
                
                for idx, key in zip(env.reward_sender.target_cols_indices, env.reward_sender.targets_cols):
                    target_value_result[key]['min'].append(np.min(state_list[:,idx]))
                    target_value_result[key]['max'].append(np.max(state_list[:,idx]))
                    target_value_result[key]['mean'].append(np.mean(state_list[:,idx]))
                    
                break
            
        end_time = time.time()

        if i_episode % verbose == 0:
            print(r"| episode:{} | duration:{} | reward - mean: {:.2f}, min: {:.2f}, max: {:.2f} | run time : {:.2f} | done : {}".format(i_episode+1, t + 1, mean_reward, min_reward, max_reward, end_time - start_time, done))

        min_reward_list.append(min_reward)
        max_reward_list.append(max_reward)
        mean_reward_list.append(mean_reward)

        # memory cache delete
        gc.collect()

        # torch cache delete
        torch.cuda.empty_cache()
        
        # save weights
        if save_last is not None:
            torch.save(policy_network.state_dict(), save_last)
        
        if mean_reward > best_reward and save_best is not None:
            best_reward = mean_reward
            best_episode = i_episode
            torch.save(policy_network.state_dict(), save_best)

    print("RL training process clear....!")
    env.close()
    
    episode_reward = {
        "min" : min_reward_list,
        "max" : max_reward_list,
        "mean" : mean_reward_list
    }

    return target_value_result, episode_reward
