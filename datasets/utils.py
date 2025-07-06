import numpy as np
import torch
import glob 
import trimesh
import torch.nn.functional as F
import math
import random
import secrets
import scipy
#from pyhocon import ConfigFactory
import torch
import open3d as o3d
import napf

from   datasets import KDTREE
if KDTREE == 'ckdtree':
    from scipy.spatial import cKDTree as KDTree
elif KDTREE == 'napf':
    class KDTree:
        def __init__(self, pc):
            super().__init__()
            self.tree = napf.KDT(tree_data=pc, metric=2 )
        def query(self, queries,k=1, **kwargs):
            dist, idx = self.tree.knn_search(
                                queries=queries,
                                kneighbors=k,
                                nthread=50)
            return np.sqrt(dist), idx
else:
    raise ValueError(f"Unknown KDTREE module: {KDTREE}")

def fix_seeds():
    """
    Fix the seeds of numpy, torch and random to ensure reproducibility across
    different runs. This is useful when you want to compare the results of
    different experiments, or when you want to reproduce the results of a paper.
    """
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
def load_conf(path):
    """
    params:
    ------
    path: path to the neural pull config file
    
    returns the config namespace
    """
    f = open(path)
    conf_text = f.read()
    f.close()

    return ConfigFactory.parse_string(conf_text)

def load_pointcloud(datapath):
    """
    params:
    ------
    datapath: path to the data directory
    
    returns a dict containing the points, occupancy grid, pointcloud, and normals.
    """
    try: 
        dataspace = np.load(datapath + 'points.npz')
        points_tgt = dataspace['points'].astype(np.float32)
        occ_tgt = np.unpackbits(dataspace['occupancies']).astype(np.float32)
    except : 
        points_tgt, occ_tgt = None, None
    datashape = np.load(datapath + 'pointcloud.npz')
    pointcloud_tgt = datashape['points'].astype(np.float32)
    

    
    normals_tgt = datashape.get('normals', np.zeros_like(pointcloud_tgt)).astype(np.float32) 
    data = {
        'points': points_tgt,
        'occ' : occ_tgt,
        'pc': pointcloud_tgt,
        'normals': normals_tgt,
           }
    data['bounds']= datashape.get('bounds')
    return data

def sample_pointcloud(data, N):
 
    """
    params:
    ------
    data: dict containing pc,  normals.
    N : int number of points to sample.
    
    returns sampled points and normals
    """
    scr = 183965288784846061718375689149290307792 #secrets.randbits(128)
    rng = np.random.default_rng( scr )     
    pointcloud = data['pc']
    normals_tgt = data['normals']
    if N<10:
        N = pointcloud.shape[0]
    point_idx = rng.choice(pointcloud.shape[0], N, replace = False)
    return pointcloud[point_idx,:], normals_tgt[point_idx,:]

 
def add_gaussian_noise(points, sigma ):
    """
    params:
    ------
    points: clean input points of size( N, 3)
    sigma: std.
    
    returns noisy input points.
    """
    return points + sigma* np.random.randn(points.shape[0],points.shape[-1])
 


def get_sigmas(noisy_data):
    
    """
    Compute the local sigmas for a given noisy pointcloud.
    
    The local sigmas are computed as the distance to the 50th nearest neighbor of each point in the pointcloud.
    
    Parameters
    ----------
    noisy_data : torch Tensor of shape (N, 3)
        The noisy pointcloud.
    
    Returns
    -------
    local_sigma : torch Tensor of shape (N) containing the local sigmas.
    """
    sigma_set = []

    ptree = KDTree(noisy_data)

    for p in np.array_split(noisy_data, 100, axis=0):
        d = ptree.query(p, 50 + 1)
        sigma_set.append(d[0][:, -1])

    sigmas = np.concatenate(sigma_set)
    local_sigma = torch.from_numpy(sigmas).float().cuda()
    return local_sigma

from packaging.version import parse as parse_version

def get_scipy_kwargs(num_workers=10):
    scipy_version = scipy.__version__

    if parse_version(scipy_version) >= parse_version("1.9.0"):
        return {'workers': num_workers}
    else:
        return {'n_jobs': num_workers}
def fast_process_data(pointcloud, n_queries = 1):
 

    dim = pointcloud.shape[-1]
    scr = 183965288784846061718375689149290307792 #secrets.randbits(128)
    rng = np.random.default_rng( scr ) 
    pointcloud_ = pointcloud 
    POINT_NUM, POINT_NUM_GT,  = pointcloud.shape[0] // 60 , pointcloud.shape[0] // 60 * 60 
    QUERY_EACH = int(n_queries*1000000//POINT_NUM_GT)
    #print(POINT_NUM,POINT_NUM_GT,QUERY_EACH)
    scale = 0.25 * np.sqrt(POINT_NUM_GT / 20000)
    # Subsample to n_points_gt
    point_idx = rng.choice(pointcloud.shape[0], POINT_NUM_GT, replace = False)
    pointcloud = pointcloud_[point_idx,:]
   
    #ptree = cKDTree(pointcloud)
    ptree = KDTree(pointcloud)
    scipy_kwargs = get_scipy_kwargs (num_workers = 10)

    sigmas = ptree.query(pointcloud,51,**scipy_kwargs)[0][:,-1]
    ## Compute NN per input 
    sample = pointcloud.reshape(1,POINT_NUM_GT,dim) + scale*np.expand_dims(sigmas,-1) * rng.normal(0.0, 1.0, size=(QUERY_EACH, POINT_NUM_GT, dim))
    n_idx = ptree.query(sample.reshape(-1,dim),1,**scipy_kwargs)[1]
    sample_near =  pointcloud[n_idx].reshape((QUERY_EACH, POINT_NUM_GT, dim))
    return { "sample": sample, 'point' : pointcloud,'gt_point' : pointcloud_, 
            'sample_near' : sample_near, 'idx': point_idx, 'rho_idx': n_idx}

def normalize_mesh(mesh):
    bbox = mesh.bounds
    bbox_padding = 0.
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max() / (1 - bbox_padding)

    # Transform input mesh
    mesh.apply_translation(-loc)
    mesh.apply_scale(1 / scale)
    return mesh

def read_lidar_points(bin_file, voxel_size = 0.5):
        points = np.fromfile(bin_file, dtype = np.float32)
        points = np.reshape(points,(-1,4)) # x,y,z,intensity
        pointcloud = points[:,0:3]
        p = np.median (pointcloud, axis = 0) 
        dists = np.sum((pointcloud -p)**2, axis = 1 ) 
        pointcloud = pointcloud[dists < np.median(dists)*4]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        downpcd = pcd.voxel_down_sample( voxel_size=voxel_size)
        pointcloud = np.asarray(downpcd.points)
        return pointcloud   




def sample_batch(n_queries, batch_size, device = 'cuda'):
    index_coarse = np.random.choice(10, 1)
    index_fine = np.random.choice((n_queries-1)//10 , batch_size, replace = False)
    index = index_fine * 10 + index_coarse
    return  index
