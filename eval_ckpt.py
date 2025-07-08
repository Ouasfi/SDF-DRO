import open3d as o3d
import trimesh, os
from datasets import utils
import numpy as np
import models
import datasets

import torch
import pandas as pd
#from diso import DiffMC#,DiffDMC
#from joblib import Parallel, delayed
import trimesh
import numpy as np
import concurrent.futures
import time
import models
import evaluation
from utils import config as fg  
import argparse
from train import   fix_seeds
def load_inputs_with_bounds(shapepath, sigma, n_points):

    """
    Load input points and compute the bounding box of the pointcloud.

    Parameters
    ----------
    shapepath : str
        path to the shape data
    sigma : float
        level of noise to add to the pointcloud
    n_points : int
        number of points to sample from the pointcloud

    Returns
    -------
    shapedata : dict
        dictionary containing the pointcloud, occupancy grid, and pointcloud
    points_clean : (n_points, 3) array
        clean input points
    bounds : tuple of (3,) arrays
        bounding box of the pointcloud
    """
    shapedata = utils.load_pointcloud(shapepath)
    points_clean, _ = utils.sample_pointcloud(shapedata, N=n_points)
    noisy_points = utils.add_gaussian_noise(points_clean, sigma)
    bound_min = np.array([
        np.min(noisy_points[:, 0]), np.min(noisy_points[:, 1]),
        np.min(noisy_points[:, 2])
    ]) - 0.05
    bound_max = np.array([
        np.max(noisy_points[:, 0]), np.max(noisy_points[:, 1]),
        np.max(noisy_points[:, 2])
    ]) + 0.05
    return shapedata, points_clean, (bound_min, bound_max)

strip_prefix = lambda k, prefix : k[len(prefix):] if k.startswith(prefix) else  k
def load_sdf_network( conf, ckpt, results_dir):
    sdf_network = models.make(conf.model.name, conf.model.sdf_network).cuda()
    path = f'{results_dir}/step_{ckpt}000_weights.pth' if ckpt is not None else f'{results_dir}/weights.pth'
    state_dict = torch.load (path, map_location=torch.device("cuda"))
    state_dict = {strip_prefix(k, '_orig_mod.'):v for k,v in state_dict.items()}
    sdf_network.load_state_dict( state_dict)
    return sdf_network

def load_state_dict(  ckpt, results_dir):
    path = f'{results_dir}/step_{ckpt}000_weights.pth' if ckpt is not None else f'{results_dir}/weights.pth'
    statedict = torch.load (path, map_location=torch.device("cuda"))
    return statedict

@torch.no_grad()
def sdf_inference (occ_network, pts):
    out = occ_network.sdf(pts.cuda()) 
    return out

def select_ckpt(conf, args,input_points, bound_min, bound_max, ckpts):
    """
    Evaluate the given checkpoint numbers and return the one with the lowest chamfer distance wrt tot he input pointcloud.

    Parameters
    ----------
    conf : Config
        configuration object
    args : Namespace
        parsed command line arguments
    input_points : (n_points, 3) array
        input points
    bound_min : (3,) array
        lower bound of the bounding box
    bound_max : (3,) array
        upper bound of the bounding box
    ckpts : list of int
        list of checkpoint numbers to evaluate

    Returns
    -------
    best_ckpt : dict
        dictionary containing the best checkpoint number, chamfer distance, hausdorff distance, and the predicted mesh
    """
    def val_ckpt(ckpt):
        occ_network = load_sdf_network( conf, ckpt, args.results_dir).cuda()
        occ_network.load_state_dict( load_state_dict(  ckpt, args.results_dir) )
        sdf_function = lambda pts: -sdf_inference (occ_network, pts)

        metrics_dict, mesh, rec_p = evaluation.CD_from_SDF(
                                    bound_min,bound_max, sdf_function,
                                    point_gt=input_points ,
                                    resolution=args.mc_resolution,
                                    threshold=0
                                )
        metrics_dict['mesh'] = mesh
        return metrics_dict
    
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(val_ckpt, ckpt) for ckpt in ckpts]
        print('submit took: {:.2f} sec'.format(time.time() - start))

        start = time.time()
        scores = [future.result() for future in futures]
        print('result took: {:.2f} sec'.format(time.time() - start))
    #scores = [ val_ckpt(38)]
    #print(scores)
    return min(scores, key=lambda x: x['cd']),scores
def val_ckpt(conf, args,input_points, bound_min, bound_max, ckpt,median_iso_factor = 0):
    """
    Evaluate the given checkpoint numbers and return the one with the lowest chamfer distance wrt tot he input pointcloud.

    Parameters
    ----------
    conf : Config
        configuration object
    args : Namespace
        parsed command line arguments
    input_points : (n_points, 3) array
        input points
    bound_min : (3,) array
        lower bound of the bounding box
    bound_max : (3,) array
        upper bound of the bounding box
    ckpts : list of int
        list of checkpoint numbers to evaluate

    Returns
    -------
    best_ckpt : dict
        dictionary containing the best checkpoint number, chamfer distance, hausdorff distance, and the predicted mesh
    """
    def val_ckpt_(ckpt):
        occ_network = load_sdf_network( conf, ckpt, conf.trainer.log.log_dir).cuda()
        #occ_network.load_state_dict( load_state_dict(  ckpt, conf.trainer.log.log_dir) )
        sdf_function = lambda pts: -sdf_inference (occ_network, pts)
        inputs = torch.from_numpy(input_points).float()
        median_iso_level = sdf_function (inputs ).median().detach().cpu().numpy()*median_iso_factor
        metrics_dict, mesh, rec_p = evaluation.CD_from_SDF(
                                    bound_min,bound_max, sdf_function,
                                    point_gt=input_points ,
                                    resolution=args.mc_resolution,
                                    threshold=median_iso_level
                                )
        metrics_dict['mesh'] = mesh
        return metrics_dict
    
 
        print('result took: {:.2f} sec'.format(time.time() - start))
   # scores = [val_ckpt_(ckpt)]
    #print(scores)
    return val_ckpt_(ckpt)

def eval_pred_mesh(mesh, pointcloud_gt, normals_gt, n_points):
    """
    Evaluate a predicted mesh wrt a ground truth pointcloud.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        predicted mesh
    pointcloud_gt : (n_points, 3) array
        ground truth pointcloud
    normals_gt : (n_points, 3) array
        ground truth normals
    n_points : int
        number of points to sample from the mesh

    Returns
    -------
    scores : pandas.DataFrame
        a dataframe containing the chamfer-L1, chamfer-L2, and normals scores
    """
    pred_mesh_o3d = o3d.geometry.TriangleMesh( o3d.utility.Vector3dVector( mesh.vertices),
                              o3d.utility.Vector3iVector( mesh.faces) )
    pointcloud_pred = pred_mesh_o3d.sample_points_uniformly(n_points,use_triangle_normal=True)
    normals_pred = np.array(pointcloud_pred.normals).astype(np.float32)
    pointcloud_pred =np.array( pointcloud_pred.points).astype(np.float32)
    out_dict = evaluation.eval.eval_pointcloud(pointcloud_pred, pointcloud_gt, normals_pred, normals_gt)
    return pd.DataFrame(out_dict, index = ["1"])[['chamfer-L1','chamfer-L2', 'normals']]

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Train a model with a given configuration.")
    parser.add_argument("config", type=str, help="Path to the configuration file.") 
    #parser = opts.neural_pull_opts()
    parser.add_argument('--results_dir','-r', type=str, default=None)
    parser.add_argument('--shapename', '-s',type=str, default=None)
    parser.add_argument('--mc_resolution', '-res',type=int, default=256)
    parser.add_argument('--ckpt', '-ckpt',type=int, default=None)
    parser.add_argument('--device', '-d',type=int, default=0)
    args = parser.parse_args()
    conf = fg.Config(fg.load_config(args.config, vars(args)))
    #args.device
    os.environ['CUDA_VISIBLE_DEVICES']= str(args.device)
    fix_seeds()
    if args.shapename:
        conf["dataset"]["shape_name"] = args.shapename

    conf["trainer"]['log']["log_dir"] = conf.trainer.log.log_dir.replace("${dataset.name}", conf.dataset.name).replace("${dataset.shape_name}", conf.dataset.shape_name) if not args.results_dir else args.results_dir
    device = 'cuda'
    dataset = datasets.make(conf.dataset.name, conf.dataset)
    bound_min, bound_max = dataset.bounds
    best_score  = val_ckpt(conf, args, dataset.data['pc'], bound_min, bound_max, ckpt = args.ckpt,median_iso_factor = 0)
    print('CD:', best_score['cd'], 'HD:', best_score['hd'])

    best_score['mesh'].export(conf.trainer.log.log_dir + '/bestmesh.obj')
    ## As the mesh was normalized to the unit cube before training scale it back using the input pc bounds
    best_score['mesh'].vertices = dataset.bbox_unscale(best_score['mesh'].vertices)
    #print(best_score['mesh'].vertices.min(), best_score['mesh'].vertices.max(), dataset.data['gt_pc'].min(), dataset.data['gt_pc'].max())
    scores = eval_pred_mesh(best_score['mesh'], dataset.data.get('gt_pc',dataset.data['pc']),dataset.data.get('gt_pc_normals', None) , 1000000)
    print(scores)
    pd.DataFrame(scores).to_csv(conf.trainer.log.log_dir + '/val_scores.csv')

    
    