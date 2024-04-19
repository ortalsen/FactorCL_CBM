import hydra
import numpy
import random
from pathlib import Path
import omegaconf
import wandb
import pandas as pd
import pytorch_lightning as pl
import torch
import time
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import EarlyStopping
from lightning.pytorch import seed_everything
import torchvision
from dataset import colored_mnist
from dataset.colored_mnist import*
torch.backends.cudnn.deterministic = True
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

@hydra.main(config_path="configs/", config_name="default")

def main(cfg):

    seed = cfg.train.random_seed
    seed_everything(seed, workers=True)

    run_name = str(cfg.informed_model._target_) + time.strftime("_%Y_%m_%d") + '_1:9_concept_embed'
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=run_name)
    test_metrics_list = []

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)
    hydra.utils.log.info(f"Experiment directory {hydra_dir}")

    for _ in range(cfg.num_eval):
        # The Lightning core, the Trainer
        
        early_stopping_callback = EarlyStopping(
        monitor='concept_encoder/val_loss',  # Metric to monitor for early stopping
        patience=5,           # Number of epochs with no improvement after which training will be stopped
        mode='min',            # Whether to minimize or maximize the monitored metric
        min_delta=0.1
        )
        
        trainer = pl.Trainer(
            default_root_dir=hydra_dir,
            deterministic="warn" if cfg.train.deterministic else False,
            **cfg.train.pl_trainer,
            callbacks=[early_stopping_callback],
            inference_mode=False,
        )
        
        datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
        
        #Train concept encoder
        
        hydra.utils.log.info(f"Instantiating <" f"{cfg.concept_model._target_}>")
        hydra.utils.log.info(f"Training for {cfg.train.pl_trainer.max_epochs} epochs.")
        concept_model: pl.LightningModule = hydra.utils.instantiate(cfg.concept_model)
        trainer.fit(model=concept_model, datamodule=datamodule)
        
        #Train Informed model
        early_stopping_callback = EarlyStopping(
        monitor='informed_encoder/val_loss',  # Metric to monitor for early stopping
        patience=5,           # Number of epochs with no improvement after which training will be stopped
        mode='min',            # Whether to minimize or maximize the monitored metric
        min_delta=0.1
        )
        trainer = pl.Trainer(
            default_root_dir=hydra_dir,
            deterministic="warn" if cfg.train.deterministic else False,
            **cfg.train.informed_trainer,
            # callbacks=[early_stopping_callback],
            inference_mode=False,
        )
        hydra.utils.log.info(f"Instantiating <" f"{cfg.informed_model._target_}>")
        hydra.utils.log.info(f"Training for {cfg.train.pl_trainer.max_epochs} epochs.")
        informed_model: pl.LightningModule = hydra.utils.instantiate(cfg.informed_model, encoder=concept_model) 
        trainer.fit(model=informed_model, datamodule=datamodule)
        
        #Train Informed model + concept Encoder

        # early_stopping_callback = EarlyStopping(
        # monitor='final_model/val_loss',  # Metric to monitor for early stopping
        # patience=5,           # Number of epochs with no improvement after which training will be stopped
        # mode='min',            # Whether to minimize or maximize the monitored metric
        # min_delta=0.1
        # )
        # trainer = pl.Trainer(
        #     default_root_dir=hydra_dir,
        #     deterministic="warn" if cfg.train.deterministic else False,
        #     **cfg.train.final_trainer,
        #     callbacks=[early_stopping_callback],
        #     inference_mode=False,
        # )
        # hydra.utils.log.info(f"Instantiating <" f"{cfg.final_model._target_}>")
        # hydra.utils.log.info(f"Training for {cfg.train.pl_trainer.max_epochs} epochs.")
        # final_model: pl.LightningModule = hydra.utils.instantiate(cfg.final_model, encoder=concept_model, informed_encoder=informed_model) 
        # trainer.fit(model=final_model, datamodule=datamodule)

        
        #Test Final model
        
        hydra.utils.log.info(f"Starting testing!")
        # metrics = trainer.test(model=concept_model, datamodule=datamodule)
        metrics = trainer.test(model=informed_model, datamodule=datamodule)#final_model
        # metrics = trainer.test(model=final_model, datamodule=datamodule)
        test_metrics_list.append(metrics)


    # Calculate mean and confidence intervals for each metric
    metric_names = test_metrics_list[0][0].keys()
    mean_metrics = {}
    std_metrics = {}
    confidence_intervals = {}

    for metric_name in metric_names:
        metric_values = [test_metrics[0][metric_name] for test_metrics in test_metrics_list]
        metric_values = np.array(metric_values)
        mean_metric = np.mean(metric_values)
        std_metric = np.std(metric_values)
        confidence_interval = 1.96 * std_metric / np.sqrt(cfg.num_eval)
        mean_metrics[metric_name] = mean_metric
        confidence_intervals[metric_name] = confidence_interval
        std_metrics[metric_name] = std_metric

    # Print mean and confidence intervals for each metric
    for metric_name, mean_metric in mean_metrics.items():
        print(f"Mean {metric_name}: {mean_metric} \u00B1 {confidence_intervals[metric_name]}")
        wandb.log({metric_name: mean_metric})
        wandb.log({'confidence': confidence_intervals[metric_name]})



if __name__ == "__main__":
    main()
