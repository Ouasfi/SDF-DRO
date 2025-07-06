import utils
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import torch
import trimesh
import datasets
@datasets.register('3dscenes')
class SceneDataset(datasets.BaseDataset):
    def __init__(self,   config):
        super().__init__(   config)
   
    def load_pointcloud(self, datapath):
        """
        params:
        ------
        datapath: path to the data directory
        
        returns a dict containing the points, occupancy grid, pointcloud, and normals.
        """
        mesh = trimesh.load(datapath + f'{self.config.shape_name}.ply'  )
        bounds= mesh.bounds
        mesh = utils.normalize_mesh(mesh)
        pointcloud_tgt = mesh.vertices

        
        normals_tgt =   np.zeros_like(pointcloud_tgt).astype(np.float32) 
        data = {
            'pc': pointcloud_tgt,
            'normals': normals_tgt,
            'bounds': bounds
            }
        return data
