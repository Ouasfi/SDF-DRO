import os
import json
import pandas as pd
import torch
import trimesh

class Logger:
    """
    Logger class to save metrics, losses, meshes, and network weights during training and validation steps.
    """
    def __init__(self, log_dir):
        """
        Initialize the logger with a directory to save logs.

        Args:
            log_dir (str): Directory where logs will be saved.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.train_logs = []
        self.validation_logs = []
        self.meshes = []
        self.weights = []
    def log_training_step(self, step, losses):
        """
        Log training step information.

        Args:
            step (int): Current training step.
            losses (dict): Dictionary of loss values at the current step.
        """
        self.train_logs.append({'step': step, **losses})

    def log_validation_step(self, step, metrics):
        """
        Log validation step information.

        Args:
            step (int): Current validation step.
            metrics (dict): Dictionary of validation metrics.
        """
        self.validation_logs.append({'step': step, **metrics})

    def save_logs(self):
        """
        Save all logs to CSV files in the log directory.
        """
        train_log_path = os.path.join(self.log_dir, 'train_logs.csv')
        validation_log_path = os.path.join(self.log_dir, 'validation_logs.csv')

        # Convert training logs to DataFrame and save as CSV
        train_df = pd.DataFrame(self.train_logs)
        train_df.to_csv(train_log_path, index=False)

        # Convert validation logs to DataFrame and save as CSV
        validation_df = pd.DataFrame(self.validation_logs)
        validation_df.to_csv(validation_log_path, index=False)

    def log_mesh(self, step, mesh, save = False, filename="mesh.obj"):
        """
        Save a mesh to the log directory.

        Args:
            step (int): Current step.
            mesh (trimesh.Trimesh): Mesh object to save.
            filename (str): Name of the file to save the mesh.
        """
        mesh_path = os.path.join(self.log_dir, f"step_{step}_{filename}")
        self.meshes.append({'step': step,'mesh': mesh})
        if  isinstance (mesh, trimesh.Trimesh) and save:
            mesh.export(mesh_path)

    def log_network_weights(self, step, model, save = False, filename="weights.pth"):
        """
        Save network weights to the log directory.

        Args:
            step (int): Current step.
            model (torch.nn.Module): PyTorch model whose weights are to be saved.
            filename (str): Name of the file to save the weights.
        """
        weights_path = os.path.join(self.log_dir, f"step_{step}_{filename}")
        state_dict = model.state_dict()
        self.weights.append({'step': step,'state_dict':  {k: v.cpu() for k, v in state_dict.copy().items()}})
        if  save:
            torch.save(state_dict, weights_path)

        
    def merge_logs (self):
        for i, step_log in enumerate(self.validation_logs):
            step_log.update(self.meshes[i])
            step_log.update(self.weights[i])
        return self.validation_logs