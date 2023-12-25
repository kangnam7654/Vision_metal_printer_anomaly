from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

class Pipeline(pl.LightningModule):
    def __init__(self, model, perceptive_model, criterion):
        super().__init__()
        self.model = model
        self.training_step_counter = 0
        self.perceptive_model = perceptive_model
        self.criterion = criterion
        
    def training_step(self, batch):
        self.training_step_counter += 1
        
    def loop(self, batch):
        frame = batch
        augmented = self.augment(frame)
        
        outs = []
        for augment in augmented:
            out = self.model(augment)
            outs.append(out)
        
        
        
    def augment(self, x):
        return [x, x]