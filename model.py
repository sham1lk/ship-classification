import torch
import pytorch_lightning as pl
import torchmetrics
from torch.nn import Linear
from torchmetrics import F1Score
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


class Classifier(pl.LightningModule):
    def __init__(self, num_classes: int = 5, freeze: bool = False, adaptive_lr: bool = False):
        super().__init__()
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model.classifier[-1] = Linear(in_features=1280, out_features=num_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        self.valid_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.valid_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.adaptive_lr = adaptive_lr
        self.freeze = freeze
        if self.freeze:
            for param in list(self.model.parameters())[:-1]:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.int64)
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y) # TODO: add weight to classes
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.int64)
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)  # TODO: add weight to classes
        self.log("validation_loss", loss)
        self.valid_acc.update(y_hat, y)
        self.valid_f1.update(y_hat, y)

    def on_validation_epoch_end(self):
        self.log('valid_acc_epoch', self.valid_acc.compute())
        self.log('valid_f1_epoch', self.valid_f1.compute())
        self.valid_acc.reset()
        self.valid_f1.reset()

    def on_train_epoch_end(self) -> None:
        if self.freeze:
            for param in list(self.model.parameters())[:-1][::-1]:
                if param.requires_grad:
                    param.requires_grad = True
                    break

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        if self.adaptive_lr:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5) # TODO: Move params to config
            return [optimizer], [scheduler]
        return optimizer

