import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
import trainers
import models
import math
import trainers.utils as utils
import evaluation  

def batched_parwise_distance(x):
    xx = torch.bmm(x, x.transpose(1, 2))  # (B, C, C)
    rx = torch.diagonal(xx, dim1=1, dim2=2)  # (B, C)
    # Compute pairwise squared distances
    rx_row = rx.unsqueeze(-1)  # (B, C, 1)
    rx_col = rx.unsqueeze(-2)  # (B, 1, C)
    dxx = rx_row + rx_col - 2 * xx  # (B, C, C)
    #unbiased estimate
    dxx = dxx/(dxx.size(-1)-1)
    return dxx #.mean()
@trainers.register('pdiff')
class SDROTrainer (trainers.base.BaseTrainer):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def __init__(self, config):
        super().__init__(config)
        # self.dataset = models.make( config.dataset.name, config.dataset)
        # self.model = models.make( config.model.name, self.config.model)
        self.model = torch.compile(self.model)
        # self.config = config
        self.prepare()
    
    def prepare(self):
        ## create optimizer and scheduler
        # set hyperparams
        self.rho , self.lambda_wasserstain = (self.dataset.rho_estimate/self.config.rho).cuda() , self.config.lambda_wasserstain 
        self.rho_lambda =  self.rho  * self.lambda_wasserstain
        self.m_dro = self.config.m_dro
        self.scheduler = self.configure_optimizers()['lr_scheduler']
        self.alpha = self.config.alpha

    def forward(self, batch):
 
        z = torch.randn_like(batch['queries'])
        x = torch.cat([batch['queries'], z], 1)
        sdf, grad =  self.model.sdf_with_grad(x)
        return  {'sdf' : sdf, "grad" : F.normalize(grad.squeeze()[...,:3], dim=1), "grad_z": grad.squeeze()[...,3:] }
    
 
    def npull_loss (self,  batch):
        output = self(batch)
        p , q , sdf_q, grad_q = batch['targets'] , batch['queries'], output['sdf'], output['grad']
        return torch.linalg.norm((p - (q - grad_q*sdf_q )), ord=2, dim=-1).mean() 
    #@torch.compile
    def pdiff_loss (self, batch):
        N,D = batch['queries'].shape
        batch['queries'] = batch['queries'].expand(self.m_dro,*batch['queries'].shape ).reshape(-1,D).contiguous()
        output = self(batch)
        # sample q' from Q_rho = a normal distribution when the cost is the eucleadian distance
        p , q , = batch['targets'] , batch['queries'].detach()
        ## compute sdf and grad with the new queries
        sdf_q, grad_q = output['sdf'], output['grad']
        pulled_q =  (q - grad_q*sdf_q ).view(self.m_dro,N, D )
        matching_loss = torch.linalg.norm((p.unsqueeze(0) - pulled_q), ord=2, dim=-1) .mean()
        dist_loss = -batched_parwise_distance(pulled_q.permute(1,0,2)) .mean() #+output['grad_z'].pow(2).sum(-1).mean()
        return matching_loss,dist_loss
        #sdro_loss    = torch.log(torch.mean(torch.exp(q_Q_rho_loss), dim=0)) * self.rho_lambda#.squeeze(0).squeeze(-1)
        #sdro_loss =  log_mean_exp(q_Q_rho_loss, dim=0)*self.rho_lambda
        #sdro_loss = (q_Q_rho_loss.logsumexp(dim=0) - math.log (self.m_dro)).squeeze() *self.rho_lambda
    def training_step(self, batch, batch_idx):
        self.scheduler.update_learning_rate(batch_idx)
        #sdf_loss =  self.npull_loss( batch)
        sdf_loss,dist_loss  = self.pdiff_loss ( batch)
        loss = self.alpha*sdf_loss+ (1-self.alpha)*dist_loss
        self.scheduler.optimizer .zero_grad()
        loss.backward()
        self.scheduler.optimizer.step()
        logs = { 'loss': loss.item(), 'sdf_loss': sdf_loss.item(), 'dro_loss': dist_loss.item() }
        return   logs
    def validation_step(self, dataset):
        b_min, b_max = dataset.bounds
        mc_threshold = 0
        gt_points = np.asarray(dataset.data['pc']  ).astype(np.float32)
        metrics_dict, mesh, rec_p = evaluation.CD_from_SDF(
            b_min, b_max, lambda pts: -self.model(torch.cat([pts.cuda(), torch.zeros_like(pts).cuda()], 1) ),
            point_gt=gt_points ,
            resolution=256,
            threshold=mc_threshold
    )
        return metrics_dict, mesh, rec_p

