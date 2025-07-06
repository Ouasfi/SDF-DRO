from log_utils.logger import Logger
from log_utils.wandb_logger import WandbLogger

def parse_logger(logger_type, config):
    """
    Parse and initialize the appropriate logger based on the configuration.

    Args:
        logger_type (str): Type of logger to use ('default' or 'wandb').
        config (dict): Configuration dictionary containing logger settings.

    Returns:
        Logger: An instance of the selected logger class.
    """
    if logger_type == 'wandb':
        project_name = config.log.project_name
        return WandbLogger(project_name, config=config)
    elif logger_type == 'default':
        log_dir = config.log.log_dir
        return Logger(log_dir)
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")
