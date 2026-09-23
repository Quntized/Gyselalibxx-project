class Config():

    input_params = {
        "state":['density_max', 'density_mean', 'temperature_max', 'velocity_max'],
        "control":['beam_temp', 'beam_vel', 'perturb_amp']
    }

    model_config = {
        "NStransformer":{
            "n_layers": 4, 
            "n_heads":8,
            "dim_feedforward" : 1024,
            "dropout" : 0.1,
            "RIN" : False,
            "feature_0D_dim" : 128,
            "feature_ctrl_dim": 128,
            "noise_mean" : 0,
            "noise_std" : 1.96,
            "kernel_size" : 3,
        }
    }
    control_config = {
        "DDPG": {
            "mlp_dim": 128
        },
        "target": {
            "temperature_max": 0.2123244
        }
    }
    COL2STR = {
    "temperature_max": "Maximum Temperature (keV)" 
}