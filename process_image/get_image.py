import pyflann
import numpy as np
from . import settings

class GetNeareastNeighbor:

    flann = pyflann.FLANN()
    params = None
    preRun = False
    
    def init(self):
        if GetNeareastNeighbor.preRun
            return
            
        GetNeareastNeighbor.preRun = True
        GetNeareastNeighbor.flann.load_index(settings.INDEX_FILE, settings.DATASET)
        GetNeareastNeighbor.params = np.load(settings.PARAMS).item()
    
    def find(self, image, num, params):
        self.init()
        result, dists = GetNeareastNeighbor.flann.nn_index(
            image, num, 
            checks=GetNeareastNeighbor.params["checks"]
        )
        return result