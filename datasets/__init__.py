datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, config):
    
    dataset = datasets[name](config)
    return dataset

from torch.utils.data import Dataset, DataLoader
import numpy as np 
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import random
import os

KDTREE = 'ckdtree' 

def __from_env():
    import os
    
    global KDTREE
    env_kdtree_backend = os.environ.get('KDTREE') 

    if env_kdtree_backend is not None and env_kdtree_backend in ['ckdtree', 'napf']:
        KDTREE = env_kdtree_backend
        
    print(f" KDTREE: {KDTREE}")
        

__from_env()
from . import utils

class BaseDataset(Dataset):
    def __init__(self, config):
        self.dataset_path = config.dataset_path
        self.config = config
        self.data = self.load_pointcloud( self.dataset_path)
        self.preprocess()
    def preprocess(self):
        clean_points, normals = utils.sample_pointcloud(self.data, N = self.config.n_points)
        noisy_points = utils.add_gaussian_noise(clean_points, self.config.sigma )
        processed_data= utils.fast_process_data(noisy_points,self.config.n_queries)
        self.rho_estimate = utils.get_sigmas(processed_data['point']).mean()
        targets = np.asarray(processed_data['sample_near']).reshape(-1,3)
        self.val_points =   np.asarray(noisy_points  ).astype(np.float32)
        self.query_points = torch.from_numpy( np.asarray(processed_data['sample']).reshape(-1,3) ).to(torch.float32)#.to(device)
        self.target_points = torch.from_numpy( targets ).to(torch.float32)#.to(device)
        self.bounds = targets.min(0)-0.05, targets.max(0)+0.05
        (self.n_pointcloud, self.n_queries) = (processed_data['sample_near'].shape[1], processed_data['sample'].shape[0])
    def load_pointcloud(self, datapath):
        """
        params:
        ------
        datapath: path to the data directory
        
        returns a dict containing the points, occupancy grid, pointcloud, and normals.
        """
        
        return NotImplementedError

    def __len__(self):
        return len(self.val_points)
    
    def __getitem__(self, index):
        # index_coarse = np.random.choice(10, 1)
        # index_fine = np.random.choice((self.query_points.shape[0]-1)//10 , self.config.batch_size, replace = False)
        # index = index_fine * 10 + index_coarse

        return { 
            'queries' : self.query_points[index],
            'targets' : self.target_points[index],
            'index': index
        }

class PCSampler(Sampler) :
    def __init__(self, dataset, n_poincloud, n_queries):
        assert len(dataset) > 0
        self.dataset = dataset
        self.n_p = n_poincloud # input points per batch
        self.n_q = n_queries   # query points per batch
        assert self.n_p < len(dataset), f"number of input points per batch should be less than pointcloud size {len(dataset)}"
    
    def __iter__(self):
        idx = 0
        n_queries = self.dataset.n_queries
        n_pointcloud = self.dataset.n_pointcloud
        while True:
 
            index_p = np.random.choice(n_pointcloud, self.n_p)
            index_q = np.random.choice(n_queries , self.n_q)
            index = (index_q.reshape(-1,1)*n_pointcloud + index_p.reshape(1,-1)) .flatten()
            yield  index
            idx += 1
            if idx == n_pointcloud:
                idx = 0
    # def __iter__(self):
    #     idx = 0
    #     n_queries = self.dataset.n_queries
    #     n_pointcloud = self.dataset.n_pointcloud
        
    #     while True:

    #         n_pointcloud = 10
    #         index_p = np.random.choice(n_pointcloud, 1)
    #         index_q = np.random.choice((self.dataset.query_points.shape[0]-1)//n_pointcloud , self.n_p * self.n_q , replace = False)
    #         index = (index_q.reshape(-1,1)*n_pointcloud + index_p.reshape(1,-1)) .flatten()
    #         yield  index
    #         idx += 1
    #         if idx == n_pointcloud:
    #             idx = 0





from . import shapenet, sfm, semanticPoss