import click
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from dataset import FolderDatasetAdaptiveAug
from model import Classifier
from torchvision.transforms import v2


@click.command()
@click.option("--freeze_layers", default=False)
@click.option("--adaptive_lr", default=False)
@click.option("--adaptive_aug", default=False)
def train(freeze_layers, adaptive_lr, adaptive_aug):
    model = Classifier(freeze=freeze_layers, adaptive_lr=adaptive_lr)
    wandb_logger = WandbLogger(project="Diabetic Retinopathy Arranged")
    trainer = pl.Trainer( max_epochs=25, logger=wandb_logger)
    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(30, expand=True)
    ])
    train_dataset, val_dataset = random_split(FolderDatasetAdaptiveAug(
        "./data",
        transform=train_transforms,
        adaptive_aug=adaptive_aug),
        [0.75, 0.25]
    )
    val_dataset.adaptive_aug = False
    val_dataset.transforms = None
    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=64, shuffle=True),
        DataLoader(val_dataset, batch_size=128))


if __name__ == "__main__":
    train()
