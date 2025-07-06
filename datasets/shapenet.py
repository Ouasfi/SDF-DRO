from datasets import utils
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import torch
import open3d as o3d
import trimesh 
import datasets
@datasets.register('shapenet')
class SNDataset(datasets.BaseDataset):
    def __init__(self, config):
        super().__init__( config)
   
    def load_pointcloud(self, datapath):
        """
        params:
        ------
        datapath: path to the data directory
        
        returns a dict containing the points, occupancy grid, pointcloud, and normals.
        """
        datashape = np.load(datapath + self.config.shape_name + 'pointcloud.npz')
        pointcloud_tgt = datashape['points'].astype(np.float32)    
    
        normals_tgt = datashape.get('normals', np.zeros_like(pointcloud_tgt)).astype(np.float32) 
        bound_min ,bound_max = pointcloud_tgt.min(0) , pointcloud_tgt.max(0)

        #normals_tgt =   np.zeros_like(pointcloud_tgt).astype(np.float32) 
        data = {
            'pc': pointcloud_tgt,
            'normals': normals_tgt,
            'bounds': (bound_min ,bound_max)
            }
        return data
