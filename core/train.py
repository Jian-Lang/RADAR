import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import colorama
import hydra
import icecream
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Back, Fore, Style
from icecream import ic
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.core_utils import (
    BinaryClassificationMetric,
    EarlyStopping,
    calculate_md5,
    copy_config_file,
    get_collator,
    get_dataset,
    get_optimizer,
    get_scheduler,
    is_movable,
    load_model,
    set_seed,
    set_worker_seed,
)
from utils.stats_utils import (
    get_model_params,
)

import wandb


class Pre_Trainer:
    def __init__(self, cfg: DictConfig, config_md5: str):
        self.cfg = cfg

        self.device = "cuda"
        self.task = cfg.task
        if cfg.task == "binary":
            self.evaluator = BinaryClassificationMetric(self.device)
        else:
            raise ValueError("task not supported")
        self.type = cfg.type
        self.model_name = cfg.model
        self.dataset_name = cfg.dataset
        self.batch_size = cfg.batch_size
        self.num_epoch = cfg.num_epoch
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.save_path = Path("model") / f"{self.model_name}-{self.dataset_name}" / f"{config_md5}"
        self.save_path.mkdir(parents=True, exist_ok=True)

        if cfg.type == "default":
            self.dataset_range = ["default"]
        else:
            raise ValueError("experiment type not supported")

        self.collator = get_collator(cfg.model, cfg.dataset, **cfg.data)

    def _reset(self, cfg, fold, type):
        cpu_count = os.cpu_count()
        train_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split="train", **cfg.data)
        test_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split="test", **cfg.data)
        if cfg.task == "binary":
            valid_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split="valid", **cfg.data)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            collate_fn=self.collator,
            num_workers=min(cpu_count, cfg.batch_size // 2),
            shuffle=True,
            generator=self.generator,
            worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed),
            pin_memory=False,
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            collate_fn=self.collator,
            num_workers=min(cpu_count, cfg.batch_size // 2),
            shuffle=False,
            generator=self.generator,
            worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed),
            pin_memory=False,
        )
        if cfg.task == "binary":
            self.valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=cfg.batch_size,
                collate_fn=self.collator,
                num_workers=min(cpu_count, cfg.batch_size // 2),
                shuffle=False,
                generator=self.generator,
                worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed),
                pin_memory=False,
            )

        steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
        self.model = load_model(cfg.model, **dict(cfg.para))
        self.model.to(self.device)
        # self.model = torch.compile(self.model)
        self.optimizer = get_optimizer(self.model, **dict(cfg.opt))
        self.scheduler = get_scheduler(self.optimizer, steps_per_epoch=steps_per_epoch, **dict(cfg.sche))
        self.earlystopping = EarlyStopping(patience=cfg.patience, path=self.save_path / "ckpt.pt")

    def run(self):
        acc_list, f1_list, prec_list, rec_list = [], [], [], []
        for fold in self.dataset_range:
            self._reset(self.cfg, fold, self.type)
            logger.info(f"Current fold: {fold}")
            for epoch in range(self.num_epoch):
                logger.info(f"Current Epoch: {epoch}")
                self._train(epoch=epoch)
                self._valid(split="valid", epoch=epoch, use_earlystop=True)
                if self.earlystopping.early_stop:
                    logger.info(f"{Fore.GREEN}Early stopping at epoch {epoch}")
                    break
                self._valid(split="test", epoch=epoch)
            logger.info(f"{Fore.RED}Best of Acc in fold {fold}:")
            self.model.load_state_dict(torch.load(self.save_path / "ckpt.pt", weights_only=False))
            best_metrics = self._valid(split="test", epoch=epoch, final=True)
            acc_list.append(best_metrics["acc"])
            f1_list.append(best_metrics["macro_f1"])
            prec_list.append(best_metrics["macro_prec"])
            rec_list.append(best_metrics["macro_rec"])

        logger.info(
            f"Best of Acc in all fold: {np.mean(acc_list)}, Best F1: {np.mean(f1_list)}, Best Precision: {np.mean(prec_list)}, Best Recall: {np.mean(rec_list)}"
        )
        torch.save(self.model.state_dict(), self.save_path / "ckpt.pt")
        shutil.copy(self.save_path / "ckpt.pt", self.save_path.parent / "ckpt.pt")
        logger.info(f"Model saved at {self.save_path / 'ckpt.pt'}")
        wandb.log(
            {
                "final/acc": np.mean(acc_list),
                "final/f1": np.mean(f1_list),
                "final/prec": np.mean(prec_list),
                "final/rec": np.mean(rec_list),
            }
        )

    def _train(self, epoch: int):
        loss_list = []
        loss_pre_list = []
        self.model.train()
        pbar = tqdm(self.train_dataloader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}")
        epoch_start_time = time.time()
        for batch in pbar:
            _ = batch.pop("vids")
            inputs = {
                key: value.to(self.device) if is_movable(value) else value for key, value in batch.items()
            }
            labels = inputs.pop("labels")

            output = self.model(**inputs)
            logits = output["logits"] if isinstance(output, dict) else output

            if hasattr(self.model, "cal_loss"):
                match self.model.name:
                    case "FANVM":
                        loss, loss_pred = self.model.cal_loss(
                            **output, label=labels, label_event=inputs["label_event"]
                        )
                    case _:
                        loss, loss_pred = self.model.cal_loss(**output, label=labels)
            else:
                loss = loss_pred = F.cross_entropy(logits, labels)

            _, preds = torch.max(logits, 1)
            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
            loss_pre_list.append(loss_pred.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        if self.cfg.get("exp.eff", None) is not None:
            total_params, trainable_params = get_model_params(self.model)
            max_memory_allocated = torch.cuda.max_memory_allocated()
            epoch_time = time.time() - epoch_start_time
            wandb.log(
                {
                    "tot_params": total_params,
                    "train_params": trainable_params,
                    "max_gpu_memory": max_memory_allocated / 1024 / 1024,
                    "epoch_time": epoch_time,
                },
                step=epoch,
            )
        metrics = self.evaluator.compute()
        # print
        logger.info(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list)}")
        wandb.log(
            {
                "train/loss": np.mean(loss_list),
                "train/loss_pred": np.mean(loss_pre_list),
                "train/acc": metrics["acc"],
                "train/f1": metrics["macro_f1"],
            },
            step=epoch,
        )

        logger.info(
            f"{Fore.BLUE}Train: Acc: {metrics['acc']:.5f}, Macro F1: {metrics['macro_f1']:.5f}, Macro Prec: {metrics['macro_prec']:.5f}, Macro Rec: {metrics['macro_rec']:.5f}"
        )

    def _valid(self, split: str, epoch: int, use_earlystop=False, final=False):
        loss_list = []
        self.model.eval()
        if split == "valid" and final:
            raise ValueError("print_wrong only support test split")
        if split == "valid":
            dataloader = self.valid_dataloader
            split_name = "Valid"
            fcolor = Fore.YELLOW
        elif split == "test":
            dataloader = self.test_dataloader
            split_name = "Test"
            fcolor = Fore.RED
        else:
            raise ValueError("split not supported")
        for batch in tqdm(dataloader, bar_format=f"{fcolor}{{l_bar}}{{bar}}{{r_bar}}"):
            vids = batch.pop("vids")
            inputs = {
                key: value.to(self.device) if is_movable(value) else value for key, value in batch.items()
            }
            labels = inputs.pop("labels")

            with torch.no_grad():
                output = self.model(**inputs)
                logits = output["logits"] if isinstance(output, dict) else output
                loss = F.cross_entropy(logits, labels)

            _, preds = torch.max(logits, 1)
            if final:
                wrong_indices = (preds != labels).nonzero(as_tuple=True)[0]
                for idx in wrong_indices:
                    vid = vids[idx]
                    logger.debug(
                        f"{Fore.RED}True label: {labels[idx].item()}, Predicted label: {preds[idx].item()} for video {vid}"
                    )
            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
        metrics = self.evaluator.compute()

        logger.info(f"{fcolor}{split_name}: Loss: {np.mean(loss_list):.5f}")
        logger.info(
            f"{fcolor}{split_name}: Acc: {metrics['acc']:.5f}, Macro F1: {metrics['macro_f1']:.5f}, Macro Prec: {metrics['macro_prec']:.5f}, Macro Rec: {metrics['macro_rec']:.5f}"
        )
        wandb.log(
            {
                f"{split}/acc": metrics["acc"],
                f"{split}/f1": metrics["macro_f1"],
            },
            step=epoch,
        )
        if use_earlystop:
            if self.task == "binary":
                self.earlystopping(metrics["macro_f1"], self.model)
            elif self.task == "ternary":
                self.earlystopping(metrics["acc"], self.model)
            else:
                raise ValueError("task not supported")
        return metrics


@hydra.main(version_base=None, config_path="config", config_name="")
def main(cfg: DictConfig):
    config_str = OmegaConf.to_yaml(cfg)
    config_md5 = calculate_md5(config_str)[:6]

    run = wandb.init(
        project="RADAR-train",
        name=config_md5,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled",
    )
    logger.remove()
    log_path = Path(f"log/{datetime.now().strftime('%m%d-%H%M%S')}") / config_md5
    logger.add(log_path / "log.log", retention="1 days", level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    logger.info(OmegaConf.to_yaml(cfg))
    pd.set_option("future.no_silent_downcasting", True)
    colorama.init()
    icecream.install()
    set_seed(cfg.seed)

    trainer = Pre_Trainer(cfg, config_md5)
    trainer.run()


if __name__ == "__main__":
    copy_config_file()
    main()
