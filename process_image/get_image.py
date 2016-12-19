import pyflann
import numpy as np
import settings
from extract_feature import extract_feature

class GetNeareastNeighbor:

    flann = pyflann.FLANN()
    params = None
    preRun = False
    
    def init(self):
        if GetNeareastNeighbor.preRun:
            return
            
        GetNeareastNeighbor.preRun = True
        # index = np.load(settings.INDEX_FILE).item()
        dataset = np.load(settings.DATASET)
        GetNeareastNeighbor.flann.load_index(settings.INDEX_FILE, dataset)
        GetNeareastNeighbor.params = np.load(settings.PARAMS).item()
    
    def find(self, image, num):
        self.init()
        data = extract_feature([image])
        # print data
        result, dists = GetNeareastNeighbor.flann.nn_index(
            data, num, 
            checks=GetNeareastNeighbor.params["checks"]
        )
        return result
        

# g = GetNeareastNeighbor()
# print g.find('../media/test/ashmolean_3.jpg', 5)
