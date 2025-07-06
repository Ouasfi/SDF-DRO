

import json
def load_config(config_path, overrides):
    """
    Load configuration from a JSON file and apply overrides.

    Args:
        config_path (str): Path to the configuration file.
        overrides (dict): Dictionary of parameters to override.

    Returns:
        dict: Updated configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    config.update(overrides)
    return config
class Config:
    """
    A recursive wrapper class for configuration dictionaries that allows both key-based
    and attribute-based access.
    """
    def __init__(self, config_dict):
        """
        Initialize the Config object.

        Args:
            config_dict (dict): The configuration dictionary to wrap.
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def __getitem__(self, key):
        """
        Allow key-based access to dictionary values.

        Args:
            key (str): The key to access.

        Returns:
            The value associated with the key.
        """
        return self.__dict__[key]

    def __setitem__(self, key, value):
        """
        Allow key-based assignment to dictionary values.

        Args:
            key (str): The key to set.
            value: The value to assign.
        """
        self.__dict__[key] = value

    def to_dict(self):
        """
        Convert the Config object back to a standard dictionary.

        Returns:
            dict: The wrapped configuration dictionary.
        """
        def recursive_dict(obj):
            if isinstance(obj, Config):
                return {key: recursive_dict(value) for key, value in obj.__dict__.items()}
            return obj

        return recursive_dict(self)
