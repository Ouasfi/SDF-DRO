models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, config):
    """
    Create a model instance using the registered model class.

    Args:
        name (str): Name of the model.
        config (dict or Config): Configuration for the model.

    Returns:
        object: An instance of the model class.
    """
    #if isinstance(config, Config):
    config = config.to_dict()  # Convert Config object to dictionary
    #print(f"Creating model {name} with config: {config}")
    model = models[name](**config)
    return model


from . import npull
