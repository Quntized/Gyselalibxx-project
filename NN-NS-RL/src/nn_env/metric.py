
import numpy as np
from typing import Union, List, Optional
# Import the new root_mean_squared_error function
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

def MSE(gt: np.ndarray, pt: np.ndarray):
    return mean_squared_error(gt, pt)

def RMSE(gt: np.ndarray, pt: np.ndarray):
    return root_mean_squared_error(gt, pt)

def MAE(gt: np.ndarray, pt: np.ndarray):
    return np.mean(np.abs((gt - pt)))

def R2(gt: np.ndarray, pt: np.ndarray):
    return r2_score(gt, pt)
def compute_metrics(gt : Union[np.ndarray, List], pt : Union[np.ndarray, List], algorithm : Optional[str] = None, is_print : bool = True):
    
    if gt.ndim == 3:
        gt = gt.reshape(-1, gt.shape[2])
        pt = pt.reshape(-1, pt.shape[2])
    
    mse = MSE(gt, pt)
    rmse = RMSE(gt, pt)
    mae = MAE(gt, pt)
    r2 = R2(gt, pt)
    
    if is_print:
        if algorithm:
            print("| {} | mse : {:.3f} | rmse : {:.3f} | mae : {:.3f} | r2-score : {:.3f}".format(algorithm, mse, rmse, mae, r2))
        else:
            print("| mse : {:.3f} | rmse : {:.3f} | mae : {:.3f} | r2-score : {:.3f}".format(mse, rmse, mae, r2))
            
    return mse, rmse, mae, r2