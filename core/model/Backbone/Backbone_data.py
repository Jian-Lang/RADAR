import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from ..Base.base_data import (
    Base_Dataset,
    FakeSV_Dataset,
    FakeTT_Dataset,
    FMNV_Dataset,
    FVC_Dataset,
    TRUEE_Dataset,
)


class Backbone_Dataset(Base_Dataset):
    def __init__(self, fold: int, split: str, task: str, **kargs):
        super().__init__()
        fea_path = self.data_path / "model/Backbone"

        self.data = self._get_data(fold, split, task)

        self.text_fea = torch.load(fea_path / "fea_text.pt", weights_only=True)
        self.frame_fea = torch.load(fea_path / "fea_vision.pt", weights_only=True)
        self.audio_fea = torch.load(fea_path / "fea_audio.pt", weights_only=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        label = item["label"]
        vid = item["vid"]

        text_fea = self.text_fea[vid]
        vision_fea = self.frame_fea[vid].mean(0)
        audio_fea = self.audio_fea[vid]

        return {
            "vid": vid,
            "label": torch.tensor(label),
            "text_fea": text_fea,
            "vision_fea": vision_fea,
            "audio_fea": audio_fea,
        }


class Backbone_Collator:
    def __init__(self, **kargs):
        pass

    def __call__(self, batch):
        vids = [item["vid"] for item in batch]
        labels = [item["label"] for item in batch]
        text_fea = [item["text_fea"] for item in batch]
        vision_fea = [item["vision_fea"] for item in batch]
        audio_fea = [item["audio_fea"] for item in batch]

        return {
            "vids": vids,
            "labels": torch.stack(labels),
            "text_fea": torch.stack(text_fea),
            "vision_fea": torch.stack(vision_fea),
            "audio_fea": torch.stack(audio_fea),
        }


class FakeSV_Backbone_Dataset(Backbone_Dataset, FakeSV_Dataset):
    def __init__(self, fold: int, split: str, task: str, **kargs):
        super().__init__(fold=fold, split=split, task=task, **kargs)


class FakeSV_Backbone_Collator(Backbone_Collator):
    pass


class FakeTT_Backbone_Dataset(Backbone_Dataset, FakeTT_Dataset):
    def __init__(self, fold: int, split: str, task: str, **kargs):
        super().__init__(fold=fold, split=split, task=task, **kargs)


class FakeTT_Backbone_Collator(Backbone_Collator):
    pass


class FVC_Backbone_Dataset(Backbone_Dataset, FVC_Dataset):
    def __init__(self, fold: int, split: str, task: str, **kargs):
        super().__init__(fold=fold, split=split, task=task, **kargs)


class FVC_Backbone_Collator(Backbone_Collator):
    pass
