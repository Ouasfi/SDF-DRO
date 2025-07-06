import models
import datasets
import trainers.utils as utils
import evaluation  
import log_utils
import torch
import numpy as np
import time

class BaseTrainer (torch.nn.Module):
    """
    Base class for trainers.
    """
    def __init__(self, config):
        super().__init__()
        self.dataset = datasets.make(config.dataset.name, config.dataset)
        self.model = models.make(config.model.name, config.model.sdf_network).cuda()
        log_dir = config.trainer.log.log_dir.replace("${dataset.name}", config.dataset.name).replace("${dataset.shape_name}", config.dataset.shape_name)
        config.trainer.log.log_dir = log_dir
        self.config = config.trainer
        self.logger = log_utils.parse_logger(self.config.log.logger, self.config)
        self.prepare()

    def prepare(self):
        pass

    def forward(self, batch):
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def train(self, dataloader = None):
        n_queries = self.dataset.query_points.shape[0]
        batch_size = 5000
        device = 'cuda'
        for batch_idx in range(self.config.maxiter):
            batch = next(dataloader) if dataloader is not None else self.dataset[ utils.sample_batch(n_queries, batch_size, device = 'cuda')]
            #batch = self.dataset[ utils.sample_batch(n_queries, batch_size, device = 'cuda')]
            batch = {k: v.squeeze().to(device, non_blocking=True) for k, v in batch.items()}  # remove batch dimension
            start_time = time.monotonic()
            losses = self.training_step(batch, batch_idx)
            losses['time'] = (time.monotonic()-start_time)
            self.logger.log_training_step(batch_idx, losses)
            if batch_idx % self.config.validata_every == 0 and batch_idx != 0:
                metrics_dict, mesh, rec_p = self.validation_step(self.dataset)
                # Log validation metrics and mesh
                self.logger.log_validation_step(batch_idx, metrics_dict)
                self.logger.log_mesh(batch_idx, mesh, save = self.config.save_mesh)
                self.logger.log_network_weights(batch_idx, self.model, save  =   self.config.save_state_dict)
                self.logger.save_logs()
    def validation_step(self, dataset):
        b_min, b_max = dataset.bounds
        mc_threshold = 0
        gt_points = np.asarray(dataset.data['pc']  ).astype(np.float32)
        metrics_dict, mesh, rec_p = evaluation.CD_from_SDF(
            b_min, b_max, lambda pts: -self.model(pts.cuda()),
            point_gt=gt_points ,
            resolution=256,
            threshold=mc_threshold
        )
        return metrics_dict, mesh, rec_p

    def export(self, dataset, mc_threshold=0):
        b_min, b_max = dataset.bounds
        return evaluation.extract.MC(
            b_min, b_max,
            resolution=self.config.resolution,
            threshold=mc_threshold,
            query_func=lambda pts: -self.model(pts.cuda())
        )

    def configure_optimizers(self):
        optim = utils.parse_optimizer(self.config.optimizer, self.model)
        ret = {
            'optimizer': optim,
        }
        ret.update({
            'lr_scheduler': utils.parse_scheduler(self.config.scheduler, optim),
        })
        return ret
