systems = {}


def register(name):
    def decorator(cls):
        systems[name] = cls
        return cls
    return decorator


def make(name, config, load_from_checkpoint=None):
    if load_from_checkpoint is None:
        system = systems[name](config)
    else:
        system = systems[name].load_from_checkpoint(load_from_checkpoint, strict=False, config=config)
    return system

LME = 'torch'  #log mean exp implementation

def __from_env():
    import os
    
    global LME
    env_lme_backend = os.environ.get('LME') 

    if env_lme_backend is not None and env_lme_backend in ['torch', 'sum', 'max']:
        LME = env_lme_backend
        
    print(f" LME: {LME}")
__from_env()
from .base import BaseTrainer
from . import sdro, pdiff