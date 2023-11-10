import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb

torch.set_float32_matmul_precision('medium')

from .data_module import CIFAR10DataModule
from .module import MyModel

wandb.init(project="RTU-Pytorch")
wandb_logger = WandbLogger()

data_module = CIFAR10DataModule()
model = MyModel()

trainer = Trainer(
    logger=wandb_logger,
    max_epochs=2,
    accelerator='gpu',
    )
trainer.fit(model, data_module)
wandb.finish()
