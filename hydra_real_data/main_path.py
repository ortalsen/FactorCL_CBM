import hydra
import numpy
import random
from pathlib import Path
import omegaconf
import wandb
import numpy as np
import pytorch_lightning as pl
import torch
import time
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import EarlyStopping
from lightning.pytorch import seed_everything
import torchvision

torch.backends.cudnn.deterministic = True
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

@hydra.main(config_path="configs/", config_name="default")

def main(cfg):

    seed = cfg.train.random_seed
    seed_everything(seed, workers=True)

    if cfg.informed_exp:
        
        run_name = str(cfg.informed_model._target_) + '_digital_pathology_' + time.strftime("_%Y_%m_%d") + '_c_logits'
        # print(cfg)
        run = wandb.init(entity=cfg.wandb.entity,
                         project=cfg.wandb.project,
                         config={key: value for key, value in cfg.items() if key != "wandb"},
                         name=run_name)
        test_metrics_list = []

        # Hydra run directory
        hydra_dir = Path(HydraConfig.get().run.dir)
        hydra.utils.log.info(f"Experiment directory {hydra_dir}")

        for _ in range(cfg.num_eval):

            datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data.concept)
            

            #Concept Encoder Model
            hydra.utils.log.info(f"Instantiating <" f"{cfg.concept_model._target_}>")
            hydra.utils.log.info(f"Training for {cfg.train.pl_trainer.max_epochs} epochs.")

            early_stopping_callback = EarlyStopping(
            monitor='concept_encoder/val_loss',  
            patience=3,           
            mode='min',            
            min_delta=0.0
            )
            
            trainer = pl.Trainer(
                default_root_dir=hydra_dir,
                deterministic="warn" if cfg.train.deterministic else False,
                **cfg.train.pl_trainer,
                callbacks=[early_stopping_callback],
                inference_mode=False,
            )
            
            concept_model: pl.LightningModule = hydra.utils.instantiate(cfg.concept_model, cfg_optim=cfg.concept_optim)

            trainer.fit(model=concept_model, datamodule=datamodule)
            del trainer


            datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data.main)
            #Informed Model
            hydra.utils.log.info(f"Instantiating <" f"{cfg.informed_model._target_}>")
            hydra.utils.log.info(f"Training for {cfg.train.pl_trainer.max_epochs} epochs.")

            early_stopping_callback = EarlyStopping(
            monitor='informed_encoder/val_loss', 
            patience=1,         
            mode='min',            
            min_delta=0.0
            )
            trainer = pl.Trainer(
                default_root_dir=hydra_dir,
                deterministic="warn" if cfg.train.deterministic else False,
                **cfg.train.informed_trainer,
                callbacks=[early_stopping_callback],
                inference_mode=False,
            )
            informed_model: pl.LightningModule = hydra.utils.instantiate(cfg.informed_model, encoder=concept_model, cfg_optim=cfg.informed_optim) 

            trainer.fit(model=informed_model, datamodule=datamodule)
            
            
            #Testing
            hydra.utils.log.info(f"Starting testing!")
            metrics = trainer.test(model=informed_model, datamodule=datamodule)
            test_metrics_list.append(metrics)
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
    
    else:
        """
        
         Baseline Experiment Logic
         
        """
        
        run_name = str(cfg.baseline._target_) + '_digital_pathology_' + time.strftime("_%Y_%m_%d") 
        run = wandb.init(entity=cfg.wandb.entity,
                         project=cfg.wandb.project,
                         config={key: value for key, value in cfg.items() if key != "wandb"},
                         name=run_name)
        test_metrics_list = []

        # Hydra run directory
        hydra_dir = Path(HydraConfig.get().run.dir)
        hydra.utils.log.info(f"Experiment directory {hydra_dir}")

        for _ in range(cfg.num_eval):

            datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data.main)

            #Baseline
            hydra.utils.log.info(f"Instantiating <" f"{cfg.baseline._target_}>")
            hydra.utils.log.info(f"Training for {cfg.train.informed_trainer.max_epochs} epochs.")

            early_stopping_callback = EarlyStopping(
            monitor='final_model/val_loss', 
            patience=1,         
            mode='min',            
            min_delta=0.0
            )
            trainer = pl.Trainer(
                default_root_dir=hydra_dir,
                deterministic="warn" if cfg.train.deterministic else False,
                **cfg.train.informed_trainer,
                callbacks=[early_stopping_callback],
                inference_mode=False,
            )
            baseline: pl.LightningModule = hydra.utils.instantiate(cfg.baseline, cfg_optim=cfg.baseline_optim) 

            trainer.fit(model=baseline, datamodule=datamodule)
            
            
            #Testing
            hydra.utils.log.info(f"Starting testing!")
            metrics = trainer.test(model=baseline, datamodule=datamodule)
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
