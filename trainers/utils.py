from torch.optim import lr_scheduler
import sys
import math
import torch
import numpy as np
from trainers import LME
def log_mean_exp_max_trick(x, dim=0, **kwargs):
    """Numerically stable log(mean(exp(x)))"""
    x_max = torch.max(x, dim=dim, keepdim=True)[0]#.detach()
    return x_max.squeeze(dim)  + torch.log(torch.mean(torch.exp(x - x_max), dim=dim))  .squeeze()
def log_mean_exp_sum_trick(x, dim = 0, m= None ):
    return  (x.logsumexp(dim=dim) - math.log (x.shape[dim])).squeeze() 
def log_mean_exp_torch (x, dim = 0):
    return torch.log(torch.mean(torch.exp(x), dim=dim))
log_mean_exp = {'torch':log_mean_exp_torch, 'sum': log_mean_exp_sum_trick, 'max': log_mean_exp_sum_trick}[LME] 

custom_schedulers = []

def sample_batch(n_queries, batch_size, device = 'cuda'):
    index_coarse = np.random.choice(10, 1)
    index_fine = np.random.choice((n_queries-1)//10 , batch_size, replace = False)
    index = index_fine * 10 + index_coarse
    return  torch.from_numpy(index)

def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    elif name in custom_schedulers:
        return getattr(sys.modules[__name__], name)
    else:
        raise NotImplementedError
    
def parse_scheduler(config, optimizer):
    scheduler_args = {key: value for key, value in config.to_dict().items() if key != "name"}  # Extract arguments
    scheduler =  Scheduler(optimizer, **scheduler_args)
    return scheduler

def parse_optimizer(config, model):
    """
    Parse and create an optimizer based on the configuration.

    Args:
        config (dict): Optimizer configuration.
        model (torch.nn.Module): Model whose parameters will be optimized.

    Returns:
        torch.optim.Optimizer: The created optimizer.
    """
    params = model.parameters()
    optimizer_name = config.name  # Default to Adam if no name is provided
    optimizer_args = {key: value for key, value in config.to_dict().items() if key != "name"}  # Extract arguments
    optim = getattr(torch.optim, optimizer_name)(params, **optimizer_args)
    return optim

class Scheduler:
    def __init__(self, optimizer, maxiter, learning_rate, warm_up_end):
        self.warm_up_end = warm_up_end
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.optimizer = optimizer
    def get_lr(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr
        return lr
    def update_learning_rate (self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr