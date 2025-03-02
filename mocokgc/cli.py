"""CLI interface for mocokgc project.

Be creative! do whatever you want!

- Install click or typer and create a CLI app
- Use builtin argparse
- Start a web application
- Import things from your .base module
"""
import typer
from typing_extensions import Annotated
from pathlib import Path

from mocokgc.scripts.train import train as train_internal

app = typer.Typer()

@app.command("train")
def train(config_path: Annotated[str, typer.Argument()] = Path(__file__).parents[0] / "config.yaml"):
    print(config_path)
    train_internal(config_path)

@app.command("test")
def test( config_path: Annotated[str, typer.Argument()] = None):
    print(config_path)
    train_internal(config_path)
