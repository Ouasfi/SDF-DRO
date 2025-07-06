import numpy as np
import napf
from evaluation import KDTREE
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


import torch
def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    print('points_src', points_src.dtype)
    print('points_tgt', points_tgt.dtype)
    recon_kd_tree = napf.KDT(tree_data=points_tgt.astype(np.float32), metric=2)  
    dist, idx =   recon_kd_tree.knn_search(
                            queries=points_src.astype(np.float32),
                            kneighbors=1,
                            nthread=50) 
 
    #idx = np.squeeze(idx)
    idx = idx[..., 0]  
    dist = np.sqrt(torch.from_numpy(dist).numpy())

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to method not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32
        )
    del recon_kd_tree
    return dist, normals_dot_product
EMPTY_PCL_DICT = {
    'completeness': np.sqrt(3),
    'accuracy': np.sqrt(3),
    'completeness2': 3,
    'accuracy2': 3,
    'chamfer': 6,
}

EMPTY_PCL_DICT_NORMALS = {
    'normals completeness': -1.,
    'normals accuracy': -1.,
    'normals': -1.,
}
def eval_pointcloud(pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None,
                        thresholds=np.linspace(1./1000, 1, 1000)):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
            thresholds (numpy array): threshold values for the F-score calculation
        '''
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            #logger.warn('Empty pointcloud / mesh detected!')
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        #recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness**2
        hausdorf_completeness=   np.max(completeness)
        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        #precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy**2
        hausdorf_accuracy =   np.max(accuracy)
        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()
        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)
        hausdorfL1 = np.max((hausdorf_accuracy, hausdorf_completeness))
        # F-Score
        #F = [
         #   2 * precision[i] * recall[i] / (precision[i] + recall[i])
         #   for i in range(len(precision))
        #]

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
            'hausdorf-L1': hausdorfL1,
            #'f-score': F[9], # threshold = 1.0%
            #'f-score-15': F[14], # threshold = 1.5%
            #'f-score-20': F[19], # threshold = 2.0%
        }
        return out_dict


def cd_flann(recon_points, gt_points):
    recon_kd_tree = napf.KDT(tree_data=recon_points, metric=2) 
    gt_kd_tree = napf.KDT(tree_data=gt_points, metric=2)
    re2gt_distances, indices = recon_kd_tree.knn_search(
                            queries=gt_points,
                            kneighbors=1,
                            nthread=50)
    gt2re_distances, indices = gt_kd_tree.knn_search(
                            queries=recon_points,
                            kneighbors=1,
                            nthread=50)
    
    re2gt_distances = np.sqrt(re2gt_distances)
    gt2re_distances = np.sqrt(gt2re_distances)
    cd_re2gt = np.mean(re2gt_distances)
    cd_gt2re = np.mean(gt2re_distances)
    hd_re2gt = np.max(re2gt_distances)
    hd_gt2re = np.max(gt2re_distances)
    chamfer_dist = 0.5* (cd_re2gt + cd_gt2re)
    hausdorff_distance = np.max((hd_re2gt, hd_gt2re))
    return chamfer_dist , hausdorff_distance

def cd_ckdtree(recon_points, gt_points):
    recon_kd_tree = KDTree(recon_points)
    gt_kd_tree = KDTree(gt_points)
    re2gt_distances, re2gt_vertex_ids = recon_kd_tree.query(gt_points, n_jobs=10)
    gt2re_distances, gt2re_vertex_ids = gt_kd_tree.query(recon_points, n_jobs=10)

    cd_re2gt = np.mean(re2gt_distances)
    cd_gt2re = np.mean(gt2re_distances)
    hd_re2gt = np.max(re2gt_distances)
    hd_gt2re = np.max(gt2re_distances)
    chamfer_dist = 0.5* (cd_re2gt + cd_gt2re)
    hausdorff_distance = np.max((hd_re2gt, hd_gt2re))
    return chamfer_dist , hausdorff_distance