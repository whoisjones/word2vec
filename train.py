import argparse
import yaml
import os

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def train(config):
    os.makedirs(config["model_dir"])

    #TODO create dataloader function
    #TODO where to place vocab
    train_dataloader = get_dataloader(split="train")
    val_dataloader = get_dataloader(split="val")

    #TODO model
    model = model(vocab)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1)

    #TODO where to get epochs
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #TODO implement trainer
    trainer = Trainer()

    trainer.train()

    trainer.save_model()
    trainer.save_loss()

    #TODO implement
    save_vocab(vocab, path)
    save_config(config, path)

    #TODO implement analogy task

if __name__ == "__main__":
    #TODO adapt config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config)
