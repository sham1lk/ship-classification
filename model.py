import torch
import pytorch_lightning as pl
from torch.nn import Linear
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


class Classifier(pl.LightningModule):
    def __init__(self, num_classes: int = 10, freeze: bool = False, adaptive_lr: bool = False):
        super().__init__()
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model.classifier[-1] = Linear(in_features=1280, out_features=num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss(y, y_hat) # TODO: add weight to classes
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

