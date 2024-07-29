import bz2
import pickle
import os
import numpy as np

# Pickle a file and then compress it into a file with extension 
def save_and_compress_pickle(title: str, data: object):
    with bz2.BZ2File(title, "wb") as f: 
        pickle.dump(data, f)

 # Load any compressed pickle file
def load_and_decompress_pickle(file: str) -> object:
    data = bz2.BZ2File(file, "rb")
    data = pickle.load(data)
    return data

def check_need_calculate(path:str, skip_exist:bool) -> bool:
    """Check if calculation needs to be done

    Args:
        path (str): Output of calculation
        skip_exist (bool): If we should recalculate if it already exists

    Returns:
        bool: Wether calculation is needed
    """
    if not skip_exist:
        return True
    return not os.path.exists(path=path)

def create_if_not_exist(path:str) -> None:
    """Generates file path if it does not exist already

    Args:
        path (str): Path to directory
    """
    if not os.path.exists(path):
        os.makedirs(path)

def check_if_exist(path:str) -> bool:
    return os.path.exists(path)

def structural_similarity_index(x,y):
    K1 = 0.01
    K2 = 0.03
    L = 255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2


    x_mu = np.average(x, axis=(1, 2))
    x_mu = np.expand_dims(x_mu, (1, 2))

    y_mu = np.average(y, axis=(1, 2))
    y_mu = np.expand_dims(y_mu, (1, 2))

    x_std = np.std(x, axis=(1, 2))
    x_std = np.expand_dims(x_std, (1, 2))
    
    y_std = np.std(y, axis=(1, 2))
    y_std = np.expand_dims(y_std, (1, 2))
    
    xy_std = np.average(np.multiply((x - x_mu), (y - y_mu)), axis=(1, 2))
    xy_std = np.expand_dims(xy_std, axis=(1, 2))
    
    ssim_xy = ((2 * x_mu * y_mu + C1) * (2 * xy_std + C2)) / (
        (x_mu ** 2 + y_mu ** 2 + C1) * (x_std ** 2 + y_std ** 2 + C2))
        
    return ssim_xy
