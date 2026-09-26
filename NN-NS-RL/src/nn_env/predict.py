import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from src.nn_env.dataset import DatasetFor0D

def predict_from_self_tensorboard(
    model: torch.nn.Module,
    test_data, # Your DatasetFor0D object
    device: str = 'cpu',
):
    shot_list = np.unique(test_data.ts_data.shot.values)
    
    # Adapt to use the dataset's parameters instead of assuming the model has them
    seq_len_0D = test_data.seq_len
    pred_len_0D = test_data.pred_len
    seq_len_ctrl = test_data.seq_len_ctrl
    
    is_shot_valid = False
    while not is_shot_valid:
        shot_num = random.choice(shot_list)
        
        # reset_index(drop=True) guarantees a clean 0, 1, 2... index for safe .iloc slicing
        df_shot = test_data.ts_data[test_data.ts_data.shot == shot_num].reset_index(drop=True)
        
        # Ensure the shot has enough frames for an initial sequence + at least 32 prediction steps
        idx_max = len(df_shot) - pred_len_0D - seq_len_0D
        if idx_max >= 32:
            is_shot_valid = True
            
    model.to(device)
    model.eval()
    
    time_x = df_shot['time']
    cols_0D = test_data.cols_0D
    cols_ctrl = test_data.cols_ctrl
    
    data_0D = df_shot[cols_0D]
    data_ctrl = df_shot[cols_ctrl]
    
    predictions = []
    idx = 0
    
    # Initialize the state buffer using .iloc for standard Python slicing
    state_list = torch.from_numpy(data_0D.iloc[0:seq_len_0D].values)
    previous_state = torch.Tensor([])
    
    while idx < idx_max:
        with torch.no_grad():
            if idx == 0:
                # First step uses pure ground truth for the 0D input
                input_0D = torch.from_numpy(data_0D.iloc[idx : idx+seq_len_0D].values).unsqueeze(0).float()
            else:
                # Subsequent steps use the model's own previous predictions (Autoregressive)
                input_0D = previous_state.unsqueeze(0).float()
                
            # Control parameters are always known (ground truth)
            input_ctrl = torch.from_numpy(data_ctrl.iloc[idx : idx+seq_len_ctrl].values).unsqueeze(0).float()
            target_0D = torch.from_numpy(data_0D.iloc[idx+seq_len_0D: idx+seq_len_ctrl].values).unsqueeze(0).float()
            target_ctrl = torch.from_numpy(data_ctrl.iloc[idx+seq_len_0D: idx+seq_len_ctrl].values).unsqueeze(0).float()
            #dummy_target_0D = torch.zeros(1, pred_len_0D, len(cols_0D)).float()
            next_state = model(input_0D.to(device), input_ctrl.to(device), target_0D.to(device), target_ctrl.to(device)) 
            
        # Advance the time step
        idx += pred_len_0D
        
        # Append the new prediction to our running state buffer using torch.cat
        state_list = torch.cat([state_list, next_state.cpu().squeeze(0)], dim=0)
        
        # Slice the last 'seq_len_0D' frames to act as the input for the next loop iteration
        previous_state = state_list[-seq_len_0D:, :]
        
        # Save prediction for plotting
        prediction = next_state.detach().cpu().squeeze(0).numpy()
        predictions.append(prediction)
            
    predictions = np.concatenate(predictions, axis=0)
        
    # Align ground truth arrays with the prediction window for plotting
    plot_time = time_x.iloc[seq_len_0D : seq_len_0D + len(predictions)].values
    actual = data_0D.iloc[seq_len_0D : seq_len_0D + len(predictions)].values
    
    # Inverse transform if a scaler was used so the plots show real physics values
    if test_data.scaler_0D:
        predictions = test_data.scaler_0D.inverse_transform(predictions)
        actual = test_data.scaler_0D.inverse_transform(actual)
        
    # Generate the 4-panel plot
    fig, axes = plt.subplots(len(cols_0D), 1, figsize=(10, 8), sharex=True, facecolor='white')
    plt.suptitle(f"Autoregressive Rollout | Shot: {shot_num:04d}", fontsize=14)
    
    for i, (ax, col) in enumerate(zip(axes.ravel(), cols_0D)):
        ax.plot(plot_time, actual[:, i], 'k', label="Actual")
        ax.plot(plot_time, predictions[:, i], 'b--', label="Predicted")
        
        # Automatically format 'density_max' to 'Density Max'
        ax.set_ylabel(col.replace('_', ' ').title())
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax.legend(loc="upper right")

    fig.tight_layout()
    return fig

def predict_tensorboard(
    model: torch.nn.Module,
    test_data: DatasetFor0D,
    device: str = 'cpu',
):
    model.to(device)
    model.eval()

    seq_len_0D = model.input_seq_len
    pred_len_0D = model.output_pred_len
    seq_len_ctrl = seq_len_0D + pred_len_0D
    shot_list = np.unique(test_data.ts_data.shot.values)

    is_shot_valid = False
    while not is_shot_valid:
        shot_num = random.choice(shot_list)
        df_shot = test_data.ts_data[test_data.ts_data.shot == shot_num].reset_index(drop=True)
        idx_max = len(df_shot) - pred_len_0D - seq_len_0D
        is_shot_valid = idx_max >= 130   # magic-number floor, kept as in original

    cols_0D = test_data.cols_0D
    cols_ctrl = test_data.cols_ctrl
    time_x = df_shot['time']
    data_0D = df_shot[cols_0D]
    data_ctrl = df_shot[cols_ctrl]

    predictions = []
    idx = 0
    while idx < idx_max:
        with torch.no_grad():
            input_0D = torch.from_numpy(
                data_0D.iloc[idx + 1: idx + 1 + seq_len_0D].values
            ).float().unsqueeze(0)
            input_ctrl = torch.from_numpy(
                data_ctrl.iloc[idx + 1: idx +1 + seq_len_ctrl].values
            ).float().unsqueeze(0)

            # boundary / decoder seed  matches dataset.__getitem__ target_0D
            target_0D = torch.from_numpy(
                data_0D.iloc[idx + seq_len_0D: idx + seq_len_0D + pred_len_0D].values
            ).float().unsqueeze(0)
            # known future control  matches dataset.__getitem__ target_ctrl
            target_ctrl = torch.from_numpy(
                data_ctrl.iloc[idx + seq_len_0D : idx + seq_len_0D + pred_len_0D].values
            ).float().unsqueeze(0)

            outputs = model(
                input_0D.to(device), input_ctrl.to(device),
                target_0D.to(device), target_ctrl.to(device),
            ).squeeze(0).cpu().numpy()

        predictions.append(outputs)
        idx += pred_len_0D

    predictions = np.concatenate(predictions, axis=0)
    time_x = time_x.iloc[seq_len_0D + 1: seq_len_0D + 1 + len(predictions)].values
    actual = data_0D.iloc[seq_len_0D + 1: seq_len_0D  + 1 + len(predictions)].values

    if test_data.scaler_0D:
        predictions = test_data.scaler_0D.inverse_transform(predictions)
        actual = test_data.scaler_0D.inverse_transform(actual)

    fig, axes = plt.subplots(len(cols_0D), 1, figsize=(10, 6), sharex=True, facecolor='white')
    plt.suptitle("shot : {} - walk-forward (true-history) prediction".format(shot_num))
    for ax, col, i in zip(axes.ravel(), cols_0D, range(len(cols_0D))):
        ax.plot(time_x, actual[:, i], 'k', label="actual")
        ax.plot(time_x, predictions[:, i], 'b', label="pred")
        ax.set_ylabel(col.replace('_', ' ').title())
        ax.legend(loc="upper right")

    fig.tight_layout()
    return fig