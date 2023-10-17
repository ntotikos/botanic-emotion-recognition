"""
This is the example code from Weights & Biases for logging a
new project/experiment.
"""

import wandb
import random
from src.utils.constants import LOGS_DIR

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project= "wandb-test-run",
    dir=LOGS_DIR,
    name="test-run-custom-name",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

wandb.finish()
