import torch
import argparse
import numpy as np
import pandas as pd
from src.nn_env.dataset import DatasetFor0D
from src.nn_env.utility import preparing_0D_dataset, get_range_of_output
from src.nn_env.transformer import Transformer
from src.nn_env.NS_Transformer import NStransformer
from src.config import Config
from src.nn_env.train import train
from src.nn_env.forgetting import DFwrapper
from src.nn_env.evaluate import evaluate
#from src.nn_env.predict import generate_shot_data_from_real, generate_shot_data_from_self
from torch.utils.data import DataLoader
import warnings
import os

warnings.filterwarnings(action = 'ignore')

def parsing():
    parser = argparse.ArgumentParser(description="training NN based environment - 2D Vlasov")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--tag", type = str, default = "")
    parser.add_argument("--lr", type = float, default= 2e-4)
    parser.add_argument("--verbose", type=int, default=4)
    parser.add_argument("--test_shot_num", type = int, default = 999)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--max_norm_grad", type = float, default = 1.0)
    parser.add_argument("--gpu_num", type = int, default = 0)
    parser.add_argument("--model", type = str, default = "NStransformer", choices = ['NStransformer'])
    parser.add_argument("--gamma",type = float, default = 0.95)
    parser.add_argument("--step_size", type = int, default=8)
    parser.add_argument("--root_dir", type=str, default = "./weights/")
    parser.add_argument("--seq_len", type = int, default = 10)
    parser.add_argument("--pred_len", type = int, default = 1)
    parser.add_argument("--interval", type = int, default = 3)
    parser.add_argument("--use_forgetting", type = bool, default = False)
    parser.add_argument("--scale_forgetting", type = float, default = 0.1)
    parser.add_argument("--use_scaler", type = bool, default = True)
    parser.add_argument("--num_epoch", type = int, default = 32)
    parser.add_argument("--scaler", type = str, default = 'Robust', choices = ['Standard', 'Robust', 'MinMax'])
    args = vars(parser.parse_args())
    return args

print("=============== Device setup ===============")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

torch.cuda.init()
torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parsing()

    if(torch.cuda.device_count() >= 1):
        device = "cuda:{}".format(args['gpu_num'])
    else:
        device = 'cpu'

    df = pd.read_csv("/home/sajid/NS_Vlasov_2D/plasma_0D_dataset.csv").reset_index()
    config = Config()
    cols_0D = config.input_params['state']
    cols_control = config.input_params['control']
    ts_train, ts_valid, ts_test, scaler_0D, scaler_ctrl = preparing_0D_dataset(df, cols_0D, cols_control, args['scaler'])

    seq_len = args['seq_len']
    pred_len = args['pred_len']
    interval = args['interval']
    batch_size = args['batch_size']
    pred_cols = cols_0D
    train_data = DatasetFor0D(ts_train.copy(deep = True), seq_len, seq_len + pred_len, pred_len, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl)
    valid_data = DatasetFor0D(ts_valid.copy(deep = True), seq_len, seq_len + pred_len, pred_len, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl)
    test_data = DatasetFor0D(ts_test.copy(deep = True), seq_len, seq_len + pred_len, pred_len, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl)

    print("=============== Dataset information ===============")
    print("train data : ", train_data.__len__())
    print("valid data : ", valid_data.__len__())
    print("test data : ", test_data.__len__())

    train_loader = DataLoader(train_data, batch_size = batch_size, num_workers = args['num_workers'], shuffle = True, pin_memory = True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = args['num_workers'], shuffle = False, pin_memory = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = args['num_workers'], shuffle = False, pin_memory = True)

    ts_data = pd.concat([train_data.ts_data, valid_data.ts_data, test_data.ts_data], axis = 0)
    range_info = get_range_of_output(ts_data, cols_0D)

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
        range_info = range_info,
        noise_mean = config.model_config[args['model']]['noise_mean'],
        noise_std = config.model_config[args['model']]['noise_std'],
        kernel_size = config.model_config[args['model']]['kernel_size']
    ).to(device)

    if args['use_forgetting']:
        model = DFwrapper(model, args['scale_forgetting'])

    #model.summary()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args['step_size'], gamma=args['gamma'])
    tag = "{}_seq_{}_pred_{}_interval_{}".format(args['model'], args['seq_len'], args['pred_len'], args['interval'])
    if config.model_config[args['model']]['RIN']:
        tag = "{}_RevIN".format(tag)
    if args['use_forgetting']:
        tag = "{}_DF".format(tag)
    if args['use_scaler']:
        tag = "{}_{}".format(tag, args['scaler'])
    if len(args['tag']) > 0:
        tag = "{}_{}".format(tag, args['tag'])
    save_best_dir = os.path.join(args['root_dir'], "{}_best.pt".format(tag))
    save_last_dir = os.path.join(args['root_dir'], "{}_last.pt".format(tag))
    tensorboard_dir = os.path.join("./runs/", "tensorboard_{}".format(tag))
    loss_fn = torch.nn.MSELoss(reduction = 'mean')
    if os.path.exists(save_last_dir):
        pass
    print("=============== Training process ===============")
    print("Process : {}".format(tag))
    train_loss, valid_loss = train(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        args['num_epoch'],
        args['verbose'],
        save_best = save_best_dir,
        save_last = save_last_dir,
        max_norm_grad = args['max_norm_grad'],
        tensorboard_dir = tensorboard_dir,
        test_for_check_per_epoch = test_loader,
    )

    model.load_state_dict(torch.load(save_best_dir))
    test_loss, mse, rmse, mae, r2 = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device
    )
