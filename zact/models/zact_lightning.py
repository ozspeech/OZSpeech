import torch
from abc import ABC
import torch.nn as nn
from omegaconf import DictConfig
from zact.data import ZACTDataset
from lightning import LightningModule
from transformers import get_cosine_schedule_with_warmup


class ZACTLightning(LightningModule, ABC):
    def setup_dataset_optimizer(
        self, 
        dataset_cfg: DictConfig, 
        optimizer_cfg: DictConfig
        ):
        self.dataset_cfg = dataset_cfg
        self.optimizer_cfg = optimizer_cfg
        self.dataset = ZACTDataset(dataset_cfg)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=optimizer_cfg['lr'],
            betas=optimizer_cfg["betas"],
            eps=optimizer_cfg["eps"],
            weight_decay=optimizer_cfg["weight_decay"],
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=optimizer_cfg['warmup_steps'],
            num_training_steps=optimizer_cfg['max_steps'],
        )
        
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "total_loss_val",
            },
        }
    
    def get_dataset(self):
        self.dataset.setup()
        train_data=self.dataset.train_dataloader()
        val_data=self.dataset.val_dataloader()
        return train_data, val_data

    def training_step(self, batch, batch_idx):
        (
            x, 
            x_len, 
            y, 
            y_len, 
            durs,
            prompts,
            _,
        ) = batch
        
        losses = self(
            x, 
            x_len, 
            y, 
            y_len, 
            durs,
            prompts,
        )
        
        total_loss, logging_data = 0, {}
        for item in losses:
            if '_loss' in item:
                total_loss += losses[item]
                logging_data[f'{item}_train'] = losses[item]
            else:
                logging_data[f'{item}'] = losses[item]
        logging_data['total_loss_train'] = total_loss
        logging_data['lr'] = self.scheduler.optimizer.param_groups[0]['lr']
        self._logging(logging_data)

        return {"loss": total_loss, "log": losses}
    
    def validation_step(self, batch, batch_idx):
        (
            x, 
            x_len, 
            y, 
            y_len, 
            durs,
            prompts,
            _,
        ) = batch
        
        losses = self(
            x, 
            x_len, 
            y, 
            y_len, 
            durs,
            prompts,
        )
        
        total_loss, logging_data = 0, {}
        for item in losses:
            if '_loss' in item:
                total_loss += losses[item]
                logging_data[f'{item}_val'] = losses[item]
        logging_data['total_loss_val'] = total_loss
        self._logging(logging_data)

        return total_loss
    
    def test_step(self, batch, batch_idx):
        return
    
    def _logging(self, logs: dict):
        for key in logs:
            self.log(
                name=key,
                value=logs[key],
                on_step=True,
                on_epoch=True,
                logger=True,
                batch_size=self.optimizer_cfg['batch_size'],
                sync_dist=True,
            )