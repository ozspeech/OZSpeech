import os
import torch
import argparse
from zact import ZACT
import lightning.pytorch as pl
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint


def train(exp_root, exp_name, devices, batch_size, epochs, ckpt):
    
    if not os.path.exists(os.path.join(exp_root, exp_name)):
        os.mkdir(os.path.join(exp_root, exp_name))

    flow_cfg = OmegaConf.load('configs/flow.yaml')
    codesgen_cfg = OmegaConf.load('configs/generator.yaml')
    style_cfg = OmegaConf.load('configs/style.yaml')
    optimizer_cfg = OmegaConf.load('configs/optimizer.yaml')
    data_config = OmegaConf.load('configs/data.yaml')
    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    flow_cfg['device'] = accelerator
    codesgen_cfg['device'] = accelerator
    style_cfg['device'] = accelerator
    optimizer_cfg['device'] = accelerator
    data_config['device'] = accelerator
    
    optimizer_cfg['epochs'] = epochs
    optimizer_cfg['batch_size'] = batch_size
    data_config['batch_size'] = batch_size

    cfg = OmegaConf.create({
        'flow_matching': flow_cfg,
        'codes_generator': codesgen_cfg
    })
    OmegaConf.save(cfg, os.path.join(os.path.join(exp_root, exp_name), 'config.yaml'))

    model = ZACT(cfg)
    model.setup_dataset_optimizer(data_config, optimizer_cfg)
    train_data, val_data = model.get_dataset()

    checkpoint_callback = ModelCheckpoint(
        monitor='total_loss_val_epoch',
        filename='ckpt-{epoch:02d}-{total_loss_val_epoch:.2f}',
        save_top_k=10,
        mode='min',
        save_last=True,
    )

    trainer = pl.Trainer(
        devices=devices, 
        accelerator=accelerator, 
        max_epochs=epochs,
        enable_checkpointing=True, 
        logger=True,
        log_every_n_steps=1, 
        check_val_every_n_epoch=1,
        default_root_dir=os.path.join(exp_root, exp_name),
        callbacks=[checkpoint_callback],
    )
    
    trainer.fit(
        model=model,
        ckpt_path=ckpt,
        train_dataloaders=train_data,
        val_dataloaders=val_data,
    )


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')  

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', type=str, default=None)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()
    
    exp_root = args.exp_root
    exp_name = args.exp_name
    devices = [int(device) for device in args.devices.split(',')]
    batch_size = args.batch_size
    epochs = args.epochs
    ckpt = args.ckpt
    
    train(exp_root, exp_name, devices, batch_size, epochs, ckpt)