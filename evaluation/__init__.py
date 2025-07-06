
import torch
import numpy as np
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

from . import extract, eval


def CD_from_SDF(bound_min, bound_max, query_func, point_gt=None, resolution=64, threshold=0.0, n_samples=100000):
    bound_min = torch.tensor(bound_min, dtype=torch.float32)
    bound_max = torch.tensor(bound_max, dtype=torch.float32)
    mesh = extract.MC(bound_min, bound_max, resolution=resolution, threshold=threshold, query_func=query_func)
    
    if mesh.is_empty:
        print("Mesh extraction failed. Mesh is empty.")
        return {'cd': np.inf, 'hd': np.inf}, None, None
    
    recon_points = mesh.sample(n_samples).astype(np.float32)
    cd, hd = eval.cd_ckdtree(point_gt , recon_points )
    metrics = {
        'cd': cd,
        'hd': hd,
    }
    return metrics, mesh, recon_points

