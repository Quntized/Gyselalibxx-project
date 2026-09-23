import torch
import argparse, os
import pandas as pd
import warnings
from src.rl.env import NeuralEnv
from src.nn_env.transformer import Transformer
from src.nn_env.NS_Transformer import NStransformer
from src.nn_env.forgetting import DFwrapper
from src.rl.rewards import RewardSender
from src.rl.utility import InitGenerator, preparing_initial_dataset, get_range_of_output, plot_rl_status
from src.rl.ddpg import Actor, Critic, train_ddpg, OUNoise
from src.rl.buffer import ReplayBuffer
from src.rl.PER import PER
from src.rl.action import NormalizedActions, ClippingActions
from src.config import Config

warnings.filterwarnings(action = 'ignore')

def parsing():
    parser = argparse.ArgumentParser(description="Training RL algorithms for 1D1V plasma")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "")
    parser.add_argument("--algorithm", type = str, default = "DDPG", choices=['DDPG'])
    parser.add_argument("--save_dir", type = str, default = "./result")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # scenario for training
    parser.add_argument("--shot_random", type = bool, default = True)
    parser.add_argument("--t_init", type = float, default = 0.0)
    parser.add_argument("--t_terminal", type = float, default = 10.0)
    parser.add_argument("--dt", type = float, default = 0.05)
    
    # ReplayBuffer setting
    parser.add_argument("--capacity", type = int, default = 50000)
    parser.add_argument("--use_PER", type = bool, default = False)
    
    
    # training setup
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--num_episode", type = int, default = 5000)  
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--gamma", type = float, default = 0.995)
    parser.add_argument("--min_value", type = float, default = -10.0)
    parser.add_argument("--max_value", type = float, default = 10.0)
    parser.add_argument("--tau", type = float, default = 0.01)
    parser.add_argument("--verbose", type = int, default = 4)
    parser.add_argument("--use_CAPS", type = bool, default=False)
    parser.add_argument("--lamda_temporal_smoothness", type = float, default = 4.0)
    parser.add_argument("--lamda_spatial_smoothness", type = float, default = 1.0)
    
    # environment setup
    parser.add_argument("--stochastic", type = bool, default = False)
    parser.add_argument("--use_normalized_action", type = bool, default = False)
    parser.add_argument("--use_clip_action", type = bool, default=False)
    parser.add_argument("--env_noise_scale_0D", type = float, default = 0.1)
    parser.add_argument("--env_noise_scale_ctrl", type = float, default = 0.1)
    parser.add_argument("--env_noise_mean_0D", type = float, default = 0)
    parser.add_argument("--env_noise_mean_ctrl", type = float, default = 0)
    parser.add_argument("--env_noise_std_0D", type = float, default = 1.0)
    parser.add_argument("--env_noise_std_ctrl", type = float, default = 1.0)
    
    # predictor config
    parser.add_argument("--model", type = str, default = 'NStransformer', choices=['NStransformer'])
    parser.add_argument("--predictor_weight", type = str, default = "./weights/NStransformer_seq_10_pred_1_interval_3_Robust_best.pt")
    parser.add_argument("--use_DF", type = bool, default = False)
    parser.add_argument('--scale_DF', type = float, default = 0.1)
    parser.add_argument("--seq_len", type = int, default = 10)
    parser.add_argument("--pred_len", type = int, default = 1)
    
    args = vars(parser.parse_args())

    return args
print("=============== Device setup ===============")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())
torch.cuda.init()
torch.cuda.empty_cache()

if __name__ == "__main__":
    
    # parsing
    args = parsing()
    
    tag = "{}_{}".format(args['algorithm'], args['model'])
    save_dir = args['save_dir']
    batch_size = args['batch_size']
    num_episode = args['num_episode']
    seq_len = args['seq_len']
    pred_len = args['pred_len']
    t_init = args['t_init']
    lr = args['lr']
    gamma = args['gamma']
    min_value = args['min_value']
    max_value = args['max_value']
    tau = args['tau']
    verbose = args['verbose']
    
    # tag correction
    if args['use_PER']:
        tag = "{}_PER".format(tag)
        
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
        
    config = Config()

    # columns for predictor
    cols_0D = config.input_params['state']
    cols_control = config.input_params['control']
    model = NStransformer(
        n_layers = config.model_config[args['model']]['n_layers'], 
        n_heads = config.model_config[args['model']]['n_heads'], 
        dim_feedforward = config.model_config[args['model']]['dim_feedforward'], 
        dropout = config.model_config[args['model']]['dropout'],        
        input_0D_dim = len(cols_0D),
        input_seq_len = seq_len,                                         
        input_ctrl_dim = len(cols_control),
        output_pred_len = pred_len,                                      
        output_0D_dim = len(cols_0D),
        feature_dim = config.model_config[args['model']]['feature_0D_dim'], 
        #range_info = range_info,
        noise_mean = config.model_config[args['model']]['noise_mean'],
        noise_std = config.model_config[args['model']]['noise_std'],
        kernel_size = config.model_config[args['model']]['kernel_size']
    ).to(device)
    if args['use_DF']:
        model = DFwrapper(model, args['scale_DF'])
        tag = "{}_DF".format(tag)
    model.to(device)
    model.load_state_dict(torch.load(args['predictor_weight']))
    targets_dict = config.control_config["target"]
    reward_sender = RewardSender(targets_dict, total_cols = cols_0D)
    df = pd.read_csv("/home/sajid/NS_Vlasov_2D/plasma_0D_dataset.csv").reset_index()
    df, scaler_0D, scaler_ctrl = preparing_initial_dataset(df, cols_0D, cols_control, 'Robust')
    init_generator = InitGenerator(df, t_init, cols_0D, cols_control, seq_len, pred_len, True, None)
    range_info = get_range_of_output(df, cols_control)

    if args['stochastic']:
        tag = "{}_stochastic".format(tag)

    env = NeuralEnv(
        predictor=model, 
        device = device, 
        reward_sender = reward_sender, 
        seq_len = seq_len, 
        pred_len = pred_len, 
        range_info = range_info, 
        t_terminal = args['t_terminal'], 
        dt = args['dt'], 
        cols_control=cols_control,
        use_stochastic=args['stochastic'],
        noise_mean_0D=args['env_noise_mean_0D'], noise_mean_ctrl=args['env_noise_mean_ctrl'],
        noise_std_0D=args['env_noise_std_0D'], noise_std_ctrl=args['env_noise_std_ctrl'],
        noise_scale_0D=args['env_noise_scale_0D'], noise_scale_ctrl=args['env_noise_scale_ctrl']
    )
    # action rapper
    if args['use_normalized_action']:
        env = NormalizedActions(env)
        tag = "{}_normalized".format(tag)
        
    if args['use_clip_action']:
        env = ClippingActions(env)
        tag = "{}_clipping".format(tag)
    
    # Replay Buffer
    if args['use_PER']:
        memory = PER(capacity=args['capacity'])
    else:
        memory = ReplayBuffer(capacity=args['capacity'])
    input_dim = len(cols_0D)
    n_actions = len(cols_control)
    if args['use_CAPS']:
        tag = "{}_CAPS".format(tag)
        lamda_temporal_smoothness = args['lamda_temporal_smoothness']
        lamda_spatial_smoothness = args['lamda_spatial_smoothness']
    else:
        lamda_temporal_smoothness = 0
        lamda_spatial_smoothness = 0
    
    if len(args['tag']) > 0:
        tag = "{}_{}".format(tag, args['tag'])

    ou_noise = OUNoise(n_actions, pred_len, mu = 0, theta = 0.15, max_sigma = 0.5, min_sigma = 0.1, decay_period=10000)
    # policy and value network
    policy_network = Actor(input_dim, seq_len, pred_len, config.control_config[args['algorithm']]['mlp_dim'], n_actions)
    target_policy_network = Actor(input_dim, seq_len, pred_len, config.control_config[args['algorithm']]['mlp_dim'], n_actions)
        
    value_network = Critic(input_dim, seq_len, pred_len, config.control_config[args['algorithm']]['mlp_dim'], n_actions)
    target_value_network = Critic(input_dim, seq_len, pred_len, config.control_config[args['algorithm']]['mlp_dim'], n_actions)

    #gpu allocation
    policy_network.to(device)
    target_policy_network.to(device)

    value_network.to(device)
    target_value_network.to(device)
        
    # optimizer
    value_optimizer = torch.optim.AdamW(value_network.parameters(), lr = lr)
    policy_optimizer = torch.optim.AdamW(policy_network.parameters(), lr = lr)

    # loss function for critic network
    if args['use_PER']:
        value_loss_fn = torch.nn.SmoothL1Loss(reduction = 'none')
    else:
        value_loss_fn = torch.nn.SmoothL1Loss(reduction = 'mean')
    # optimization
    print("=========== DDPG algorithm training process ===========")
    save_best = os.path.join("./weights/", "{}_best.pt".format(tag))
    save_last = os.path.join("./weights/", "{}_last.pt".format(tag))
    target_value_result, episode_reward = train_ddpg(
            env, 
            ou_noise,
            init_generator,
            memory,
            policy_network,
            value_network,
            target_policy_network,
            target_value_network,
            policy_optimizer,
            value_optimizer,
            value_loss_fn,
            batch_size,
            gamma,
            device,
            min_value,
            max_value,
            tau,
            num_episode,
            verbose,
            save_best,
            save_last,
            scaler_0D,
            args['use_CAPS'],
            lamda_temporal_smoothness,
            lamda_spatial_smoothness
        )
    
    
    
    # Evaluation
    print("=============== Evaluation process ===============")
    
    save_path = "./result/{}_episode_reward.png".format(tag)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plot_rl_status(target_value_result, episode_reward, tag, config.COL2STR, save_path)