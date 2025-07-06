from datasets import utils
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import torch
import open3d as o3d
import trimesh 
import datasets
@datasets.register('semanticposs')
class SMPDataset(datasets.BaseDataset):
    def __init__(self,   config):
        super().__init__(  config)
   
    def load_pointcloud(self, datapath):
        """
        params:
        ------
        datapath: path to the data directory
        
        returns a dict containing the points, occupancy grid, pointcloud, and normals.
        """
        pointcloud = utils.read_lidar_points(datapath + f'{self.config.shape_name}.bin',voxel_size = self.config.voxel_size)
  
        mesh = trimesh.Trimesh(vertices = pointcloud   )
        bound_min ,bound_max = pointcloud.min(0) , pointcloud.max(0)
        bbox_padding = 0.05
        loc = (bound_min + bound_max) / 2
        scale = (bound_max- bound_min).max() / (1 - bbox_padding)
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)
        pointcloud_tgt = mesh.vertices

        normals_tgt =   np.zeros_like(pointcloud_tgt).astype(np.float32) 
        data = {
            'pc': pointcloud_tgt,
            'normals': normals_tgt,
            'bounds': (bound_min ,bound_max)
            }
        return data
