import click
import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ShipDataset
from model import Classifier
from torchvision.transforms import v2


@click.command()
@click.option("--freeze_layers", default=False)
@click.option("--adaptive_lr", default=False)
@click.option("--adaptive_aug", default=False)
def train(freeze_layers, adaptive_lr, adaptive_aug):
    run = wandb.init(
        project="spip_classification",
        config={
            "freeze_layers":freeze_layers,
            "adaptive_lr": adaptive_lr,
            "adaptive_aug": adaptive_aug
        }

    )
    model = Classifier(freeze=freeze_layers, adaptive_lr=adaptive_lr)
    trainer = pl.Trainer()
    train_transforms = v2.Compose([
        ...
    ])
    train_dataset = ShipDataset("./data/train", adaptive_aug=adaptive_aug)
    val_dataset = ShipDataset("./data/valid")
    test_dataset = ShipDataset("./data/test")
    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=4),
        DataLoader(val_dataset, batch_size=32))
    trainer.test(model, DataLoader(test_dataset))


if __name__ == "__main__":
    train()
