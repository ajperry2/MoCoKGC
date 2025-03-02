import yaml
from pathlib import Path

from typing import Dict, List
from mocokgc.data.wikidata_5m import WikiData5M

def load_config(filepath: Path) -> Dict:
    """
    Load a YAML configuration file.
    :param filepath: Path to the YAML configuration file.
    :return: Dictionary with the configuration.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Config file {filepath} not found.")

    with filepath.open() as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
    print(config_data)
    return Config(config_data)

class DatasetConfig:
    path: Path
    name: str

    def __init__(self, path: str, name: str):
        self.path = Path(path)
        self.name = name

class Config:
    dataset: DatasetConfig
    seed: int

    def __init__(self, config_information: Dict):
        self.dataset = DatasetConfig(**config_information["dataset"])
        self.seed = config_information["seed"]