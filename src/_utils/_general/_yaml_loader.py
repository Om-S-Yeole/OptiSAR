import yaml


def _load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dictionary.
    Args:
        path (str): The file path to the YAML file.
    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be instance of str. Got {type(path)}.")

    with open(path, "r") as file:
        data = yaml.safe_load(file)

    return data
