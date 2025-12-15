import json
import math
import os
import sys
import time
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import colorama
import hydra
import icecream
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from colorama import Back, Fore, Style
from icecream import ic
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.core_adapt_utils import (
    copy_adapt_config_file,
    load_adapt_model,
)
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
from utils.select_method import get_selection_method


class Adaptor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = "cuda"
        self.tgt_dataset = get_dataset(cfg.src_model, cfg.tgt_dataset, **cfg.tgt_data)
        self.collator = get_collator(cfg.src_model, cfg.tgt_dataset, **cfg.tgt_data)
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.exp = cfg.get("exp", None)
        if isinstance(self.exp, str) and "event" in self.exp:
            drop_last = True
        else:
            drop_last = False

        self.tgt_dataloader = DataLoader(
            self.tgt_dataset,
            batch_size=cfg.adapt.batch_size,
            collate_fn=self.collator,
            num_workers=min(os.cpu_count(), cfg.adapt.batch_size // 2),
            shuffle=True,
            generator=self.generator,
            worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed),
            pin_memory=True,
            drop_last=drop_last,
        )
        match cfg.task:
            case "binary":
                self.evaluator = BinaryClassificationMetric(self.device)
            case _:
                raise ValueError("task not supported")
        src_model = self._load_src_model()

        self.adapt_method = load_adapt_model(
            cfg.adapt_method,
            src_model=src_model,
            # selection_method=self.selection_method,
            # para=cfg.para,
            cfg=cfg,
        )
        self.selection_method = get_selection_method(
            cfg.select_method, cfg=cfg, adaptat_method=self.adapt_method
        )
        # self.adapt_method.to(self.device)

    def _load_src_model(self):
        src_ckpt = self.cfg.get("src_ckpt", f"model/{self.cfg.src_model}-{self.cfg.src_dataset}/ckpt.pt")
        src_cfg = OmegaConf.load(f"core/config/{self.cfg.src_model}_{self.cfg.src_dataset}.yaml")
        src_model_cfg = src_cfg.get("para", {})
        src_model = load_model(self.cfg.src_model, **src_model_cfg)
        src_model.load_state_dict(torch.load(src_ckpt, weights_only=False))
        src_model.to(self.device)
        return src_model

    def __offline_adapt(self):
        if not hasattr(self.adapt_method, "offline_adapt"):
            return
        else:
            raise NotImplementedError("Offline adaptation is not implemented for now")

    def run(self):
        self.__offline_adapt()
        previous_batches = []
        for batch in tqdm(self.tgt_dataloader, desc="Adapting", total=len(self.tgt_dataloader)):
            batch = {
                key: value.to(self.device) if is_movable(value) else value for key, value in batch.items()
            }
            if self.cfg.adapt_method == "RADAR" or self.cfg.adapt_method == "SelfTrain":
                previous_batches.append(batch)
            adapt_result = self.adapt_method.adapt(
                batch=batch,
                nbsteps=self.cfg.adapt.steps,
                selection_method=self.selection_method,
                previous_batches=previous_batches,
            )
            # rewrite label last batch in previous_batches using adapt_result["pred"]
            if self.cfg.adapt_method == "RADAR":
                previous_batches[-1]["p_label"] = torch.argmax(
                    torch.softmax(adapt_result["pred"], dim=1), dim=1
                ).detach()
            # batch = {key: value.to("cpu") if is_movable(value) else value for key, value in batch.items()}
            self.evaluator.update(torch.argmax(adapt_result["pred"], dim=1), batch["labels"])

        metric = self.evaluator.compute()
        wandb.log(data=metric)
        logger.info(f"Metric: {metric}")
        logger.info(
            f"{metric['acc'] * 100:.2f} & {metric['macro_f1'] * 100:.2f} & {metric['macro_rec'] * 100:.2f}"
        )


@hydra.main(version_base=None, config_path="adapt_config", config_name="")
def main(cfg: DictConfig):
    config_str = OmegaConf.to_yaml(cfg)
    config_md5 = calculate_md5(config_str)[:6]

    run = wandb.init(
        project="RADAR",
        name=config_md5,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online",
    )
    # run.log_code("core", exclude_fn=lambda path: "__pycache__" in path)
    logger.remove()
    log_path = Path(f"log/{datetime.now().strftime('%m%d-%H%M%S')}") / config_md5
    logger.add(log_path / "log.log", retention="1 days", level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    logger.info(OmegaConf.to_yaml(cfg))
    colorama.init()
    icecream.install()
    set_seed(cfg.seed)

    adaptor = Adaptor(cfg)
    adaptor.run()


if __name__ == "__main__":
    copy_adapt_config_file()
    main()
