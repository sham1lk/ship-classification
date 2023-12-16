import click
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split, TensorDataset
from dataset import FolderDatasetAdaptiveAug
from model import Classifier
from torchvision.transforms import v2


@click.command()
@click.option("--freeze_layers", default=False)
@click.option("--adaptive_lr", default=False)
@click.option("--adaptive_aug", default=False)
@click.option("--extra_data", default=None)
def train(freeze_layers, adaptive_lr, adaptive_aug, extra_data):
    model = Classifier(freeze=freeze_layers, adaptive_lr=adaptive_lr)
    wandb_logger = WandbLogger(project="Diabetic Retinopathy Arranged")
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
    if extra_data:
        new_data, _ = random_split(val_dataset,
            [0.75, 0.25]
        )
        model = Classifier.load_from_checkpoint(extra_data)
        trainer = pl.Trainer(max_epochs=1, logger=wandb_logger, devices=[0])
        ans = trainer.predict(model, DataLoader(new_data, batch_size=64, shuffle=False))
        ans = torch.cat(ans)
        images = [i[0] for i in list(new_data)]
        del trainer
        trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)
        new_dataset = TensorDataset(torch.tensor(images), torch.tensor(ans))
        trainer.fit(
            model,
            [DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=32),
             DataLoader(new_dataset, batch_size=64, shuffle=True, num_workers=32)],
            DataLoader(val_dataset, batch_size=128)
        )
    else:
        trainer = pl.Trainer(max_epochs=25, logger=wandb_logger)
        trainer.fit(
            model,
            DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=64),
            DataLoader(val_dataset, batch_size=128, num_workers=64))
        trainer.save_checkpoint("model.ckpt")


if __name__ == "__main__":
    train()
