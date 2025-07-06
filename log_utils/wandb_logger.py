import wandb
from log_utils.logger import Logger

class WandbLogger(Logger):
    """
    Logger class to log metrics and losses using Weights & Biases (wandb).
    """
    def __init__(self, project_name, config=None):
        """
        Initialize the wandb logger.

        Args:
            project_name (str): Name of the wandb project.
            config (dict, optional): Configuration dictionary to log as wandb config.
        """
        self.project_name = project_name
        wandb.init(project=project_name, config=config)

    def log_training_step(self, step, losses):
        """
        Log training step information to wandb.

        Args:
            step (int): Current training step.
            losses (dict): Dictionary of loss values at the current step.
        """
        wandb.log({'step': step, **losses})

    def log_validation_step(self, step, metrics):
        """
        Log validation step information to wandb.

        Args:
            step (int): Current validation step.
            metrics (dict): Dictionary of validation metrics.
        """
        wandb.log({'step': step, **metrics})

    def finish(self):
        """
        Finish the wandb logging session.
        """
        wandb.finish()
