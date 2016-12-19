import pyflann
import numpy as np
import settings
from extract_feature import extract_feature

test = ['../media/test/ashmolean_3.jpg', '../media/test/balliol_1.jpg', '../media/test/all_souls_3.jpg', '../media/test/balliol_3.jpg', '../media/test/balliol_4.jpg', '../media/test/balliol_5.jpg', '../media/test/all_souls_1.jpg', '../media/test/ashmolean_4.jpg', '../media/test/all_souls_5.jpg', '../media/test/balliol_2.jpg', '../media/test/all_souls_4.jpg', '../media/test/all_souls_2.jpg', '../media/test/ashmolean_5.jpg', '../media/test/ashmolean_1.jpg', '../media/test/ashmolean_2.jpg']

data = ['../media/christ_church_4.jpg', '../media/ashmolean_3.jpg', '../media/magdalen_3.jpg', '../media/cornmarket_1.jpg', '../media/bodleian_1.jpg', '../media/radcliffe_camera_5.jpg', '../media/keble_2.jpg', '../media/hertford_4.jpg', '../media/radcliffe_camera_1.jpg', '../media/balliol_1.jpg', '../media/pitt_rivers_4.jpg', '../media/all_souls_3.jpg', '../media/cornmarket_4.jpg', '../media/hertford_3.jpg', '../media/bodleian_4.jpg', '../media/balliol_3.jpg', '../media/christ_church_3.jpg', '../media/pitt_rivers_5.jpg', '../media/pitt_rivers_2.jpg', '../media/cornmarket_2.jpg', '../media/radcliffe_camera_3.jpg', '../media/cornmarket_5.jpg', '../media/bodleian_5.jpg', '../media/christ_church_2.jpg', '../media/keble_1.jpg', '../media/balliol_4.jpg', '../media/christ_church_1.jpg', '../media/cornmarket_3.jpg', '../media/bodleian_2.jpg', '../media/magdalen_2.jpg', '../media/magdalen_5.jpg', '../media/pitt_rivers_1.jpg', '../media/balliol_5.jpg', '../media/keble_3.jpg', '../media/all_souls_1.jpg', '../media/magdalen_1.jpg', '../media/magdalen_4.jpg', '../media/ashmolean_4.jpg', '../media/hertford_2.jpg', '../media/radcliffe_camera_2.jpg', '../media/all_souls_5.jpg', '../media/keble_5.jpg', '../media/radcliffe_camera_4.jpg', '../media/keble_4.jpg', '../media/balliol_2.jpg', '../media/all_souls_4.jpg', '../media/bodleian_3.jpg', '../media/all_souls_2.jpg', '../media/pitt_rivers_3.jpg', '../media/ashmolean_5.jpg', '../media/ashmolean_1.jpg', '../media/hertford_5.jpg', '../media/ashmolean_2.jpg', '../media/hertford_1.jpg', '../media/christ_church_5.jpg']

def init_data(list_data):
    dataset = extract_feature(list_data)
    #dataset = dataset.reshape(len(list_data), -1)
    return dataset

def run():
    flann = pyflann.FLANN()
    dataset = init_data(data)
    params = flann.build_index(dataset, algorithm="autotuned", target_precision=0.9, log_level = "info")
    flann.save_index(settings.INDEX_FILE)
    np.save(settings.DATASET, dataset)
    np.save(settings.PARAMS, params)
    
# run()

    
