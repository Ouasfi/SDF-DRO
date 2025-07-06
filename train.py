import os
os.environ['KDTREE'] = 'napf'   # Can be  ckdtree or napf 
os.environ['LME'] = 'torch'     # can be 'torch' , 'sum' or 'max'
import argparse
import json
import torch
import datasets
import trainers
from utils import config as fg  
import random
import numpy as np
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
 

def main(args):
    # Load configuration and apply overrides
    config = fg.Config(fg.load_config(args.config, vars(args)))

    # Override dataset path and log directory if provided
    if args.shapename:
        config["dataset"]["shape_name"] = args.shapename
    if args.log_dir:
        config["system"]["log_dir"] = args.log_dir

    # Create trainer using the make function
    fix_seeds()
    trainer = trainers.make(config["trainer"]["name"], config )
    trainer.train()
    trainer.logger.merge_logs()
    best = min (trainer.logger.validation_logs, key = lambda x : x['cd'] )
    print('CD:', best['cd'], 'HD:', best['hd'])
    torch.save(best['state_dict'], os.path.join(trainer.logger.log_dir, f"weights.pth"))
    best['mesh'].export(os.path.join(trainer.logger.log_dir, f"mesh.obj")) 

    # # Initialize dataloader
    # sampler = datasets.PCSampler(trainer.dataset, 500, 10)
    # dataloader = iter(torch.utils.data.DataLoader(dataset=trainer.dataset, sampler=sampler, num_workers=1))
    
    # # Start training
    # fix_seeds()
    # trainer.train(dataloader)



    #Initialize dataloader
    # sampler = datasets.PCSampler(trainer.dataset, 1000, 5)
    # #fix_seeds()
    # train_loader = torch.utils.data.DataLoader(
    #                             trainer.dataset, 
    #                             sampler=sampler,
    #                             num_workers=4, # Use multiple workers
    #                             pin_memory=True,  # For GPU transfer
    #                             persistent_workers=True,  # Keep workers alive
    #                             prefetch_factor=4)  # Prefetch batches)
    # dataloader = iter(train_loader)
    #dataloader = None
    # Start training
    #fix_seeds()
    #trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with a given configuration.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    parser.add_argument("--shapename", type=str, help="Override dataset path in the configuration.")
    parser.add_argument("--log_dir", type=str, help="Override log directory in the configuration.")
    parser.add_argument("--max_iters", type=int, help="Override max_iters in the configuration.")
    parser.add_argument("--validata_every", type=int, help="Override validata_every in the configuration.")
    parser.add_argument("--alpha", type=float, help="Override alpha in the configuration.")
    parser.add_argument("--rho", type=float, help="Override rho in the configuration.")
    parser.add_argument("--lambda_wasserstain", type=float, help="Override lambda_wasserstain in the configuration.")
    parser.add_argument("--m_dro", type=int, help="Override m_dro in the configuration.")
    parser.add_argument("--resolution", type=int, help="Override resolution in the configuration.")
    args = parser.parse_args()

    main(args)
