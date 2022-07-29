import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
import importlib

class LMModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams2 = hparams
        mod_archi_ = importlib.import_module(
            f'src.architectures.{hparams["architecture_name"]}')
        archi_class_ = getattr(mod_archi_, hparams['architecture_name'])
        self.model: torch.nn.Module = archi_class_(hparams['architecture'])

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model.forward(x)
        y = y.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.hparams2['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model.forward(x)
        y = y.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=self.hparams2['batch_size'])
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        self.val_loss = torch.mean(torch.stack(validation_step_outputs))

    def configure_optimizers(self):
        self.optim = torch.optim.AdamW(
            self.parameters(), lr=self.hparams2['optimizer']['learning_rate'])
        self.reduce_lr_on_plateau_scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-4,
            verbose=True
        )
        return [self.optim]

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch > 0:
            if self.reduce_lr_on_plateau_scheduler is not None:
                self.reduce_lr_on_plateau_scheduler.step(self.val_loss, epoch=self.current_epoch)