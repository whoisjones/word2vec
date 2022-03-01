import argparse
import yaml
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from src.data import get_dataloader
from src.model import SkipGramModel
from src.trainer import Trainer


def train(config):
    if not os.path.isdir(config["model_dir"]):
        os.makedirs(config["model_dir"])

    train_dataloader, vocab = get_dataloader(
        model_name=config["model_name"],
        dataset_name=config["dataset"],
        split="train",
        batch_size=config["batch_size"],
        window_size=config["window_size"],
        return_vocab=True,
    )
    val_dataloader, _ = get_dataloader(
        model_name=config["model_name"],
        dataset_name=config["dataset"],
        split="valid",
        batch_size=config["batch_size"],
        window_size=config["window_size"],
        return_vocab=False,
    )

    vocab_size = len(vocab.get_stoi())
    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    lr_lambda = lambda epoch: (config["epochs"] - epoch) / config["epochs"]
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
    )

    trainer.train()

    trainer.save_model()
    trainer.save_loss()

    config_path = os.path.join(config["model_dir"], "config.yaml")
    with open(config_path, "w") as stream:
        yaml.dump(config, stream)

    vocab_path = os.path.join(config["model_dir"], "vocab.pt")
    torch.save(vocab, vocab_path)

    # TODO implement analogy task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    train(config)
