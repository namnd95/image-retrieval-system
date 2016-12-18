import pyflann
import numpy as np
from . import settings

def init_data():
    # dump data
    dataset = np.zeroes
    return dataset

def run():
    flann = pyflann.FLANN()
    params = flann.build_index(dataset, algorithm="autotuned", target_precision=0.9, log_level = "info")
    flann.save_index(settings.INDEX_FILE)
    np.save(settings.PARAMS, params)
    
