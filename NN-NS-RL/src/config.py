class Config():

    input_params = {
        "state":['mean_te', 'mean_ti', 'mean_ne', 'mean_ni'],
        "control":['krook_amplitude', 'kin_energy', 'kin_extent','kin_stiffness','krook_extent','krook_stiffness','nustar0','epsilon_bot','temperature_bot','mean_velocity_bot','perturb_amplitude']
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
            "noise_std" : 0.01,
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
