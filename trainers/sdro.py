import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
import trainers
import models
import math
import trainers.utils as utils
@trainers.register('sdro')
class SDROTrainer (trainers.base.BaseTrainer):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def __init__(self, config):
        super().__init__(config)

        self.model = torch.compile(self.model)

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

        sdf, grad =  self.model.sdf_with_grad(batch['queries'])
        return  {'sdf' : sdf, "grad" : F.normalize(grad.squeeze(), dim=1) }
    
    def npull_loss (self,  batch):
        output = self(batch)
        p , q , sdf_q, grad_q = batch['targets'] , batch['queries'], output['sdf'], output['grad']
        return torch.linalg.norm((p - (q - grad_q*sdf_q )), ord=2, dim=-1).mean() 
    #@torch.compile
    def sdro_loss (self, batch):
        # sample q' from Q_rho = a normal distribution when the cost is the eucleadian distance
        p , q , = batch['targets'] , batch['queries'].detach()
        q_Q_rho  = q.unsqueeze(0) + torch.sqrt(self.rho) * torch.randn([self.m_dro, q.size(0), q.size(1)], device = 'cuda') 
        q_Q_rho = q_Q_rho.reshape([-1,q.size(1)])
        ## compute sdf and grad with the new queries
        batch['queries'] = q_Q_rho
        output = self(batch)
        sdf_q_Q_rho, grad_q_Q_rho = output['sdf'], output['grad']
        pulled_q_Q_rho =  (q_Q_rho - grad_q_Q_rho*sdf_q_Q_rho ).view(-1, q.size(0), q.size(1) )
        q_Q_rho_loss = torch.linalg.norm((p.unsqueeze(0) - pulled_q_Q_rho), ord=2, dim=-1)/self.rho_lambda
        sdro_loss =  utils.log_mean_exp(q_Q_rho_loss, dim = 0)
        return sdro_loss.mean()

    def training_step(self, batch, batch_idx):
        self.scheduler.update_learning_rate(batch_idx)
        sdf_loss =  self.npull_loss( batch)
        dro_loss  = self.sdro_loss ( batch)
        loss = self.alpha*sdf_loss+ (1-self.alpha)*dro_loss
        self.scheduler.optimizer .zero_grad()
        loss.backward()
        self.scheduler.optimizer.step()
        logs = { 'loss': loss.item(), 'sdf_loss': sdf_loss.item(), 'dro_loss': dro_loss.item() }
        return   logs


