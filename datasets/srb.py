from datasets import utils
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import torch
import trimesh
import datasets
@datasets.register('srb')
class SRBDataset(datasets.BaseDataset):
    def __init__(self,  config):
        super().__init__(   config)
   
    def load_pointcloud(self, datapath):
        """
        params:
        ------
        datapath: path to the data directory
        
        returns a dict containing the points, occupancy grid, pointcloud, and normals.
        """
        ## Load scan pointcloud
        mesh = trimesh.load(datapath + f'{self.config.shape_name}.ply'  )
        ## Load clean model
        gt_pc = trimesh.load(datapath.replace('scans','ground_truth' ) + f'{self.config.shape_name}.xyz'  ).vertices
        ## save the scan bounds to rescale the predicted mesh before evaluation
        bounds= mesh.bounds
        ## Normalize scan pointcloud for training
        mesh = utils.normalize_mesh(mesh)
        pointcloud_tgt = mesh.vertices

        
        normals_tgt =   np.zeros_like(pointcloud_tgt).astype(np.float32) 
        data = {
            'pc': pointcloud_tgt,
            'gt_pc': gt_pc, 
            'normals': normals_tgt,
            'bounds': bounds
            }
        return data
    def bbox_unscale(self, pc):
        bbox = self.data['bounds']
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 -0.00)
        mesh = trimesh.Trimesh(vertices = pc)
            # Transform input mesh
        mesh.apply_scale(scale)
        mesh.apply_translation(loc)
        return mesh.vertices