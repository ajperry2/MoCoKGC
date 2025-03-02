from pathlib import Path
from typing import Optional

from mocokgc.utils.config import load_config, Config
from mocokgc.utils import set_seed
from mocokgc.data.wikidata_5m import WikiData5M
from mocokgc.models.mlp import MLP



def train(config_path: str):
    """
    Train the model.
    """
    # Load the configuration file
    config: Config = load_config(Path(config_path))

    # Set the random seed for reproducibility
    set_seed(config.seed)

    # Load the dataset
    training_dataset = WikiData5M(mode="train")
    validation_dataset = WikiData5M(mode="valid")
    test_dataset = WikiData5M(mode="test")
    