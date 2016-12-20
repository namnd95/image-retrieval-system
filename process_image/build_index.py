import pyflann
import numpy as np
import settings
from extract_feature import extract_feature
import glob
import os


def get_list_data(ext):    
    data = glob.glob(settings.MEDIA_DIR + '*.' + ext)
    return data
    
data = sorted(get_list_data('jpg'))

def init_data(list_data):
    #dataset = extract_feature(list_data)
    for i, name in enumerate(list_data):
        data_name = name + '.npy'
        if not os.path.exists(data_name):
            cur = extract_feature([name])
            np.save(name+'.npy', cur)
        print i, name
    #dataset = dataset.reshape(len(list_data), -1)

def init_dataset():
    list_file = sorted( get_list_data('npy') )
    result = None
    for name in list_file:
        cur = np.load(name)
        if result is None:
            result = cur
        else:
            result = np.append(result, cur, axis=0)
    print result
    return result
    
    

def run():
    flann = pyflann.FLANN()
    init_data(data)
    dataset = init_dataset()
    params = flann.build_index(dataset, algorithm="autotuned", target_precision=0.9, log_level = "info")
    flann.save_index(settings.INDEX_FILE)
    np.save(settings.DATASET, dataset)
    np.save(settings.PARAMS, params)
    
if __name__ == "__main__":
    run()

    
