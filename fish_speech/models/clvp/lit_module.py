import itertools
import math
from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn

class CLVP(L.LightningModule):
    def __init__(
            self,
            
    )