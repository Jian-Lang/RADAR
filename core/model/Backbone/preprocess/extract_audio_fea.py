import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    ChineseCLIPModel,
    CLIPModel,
    DebertaModel,
    DistilBertModel,
    ModernBertModel,
    Siglip2ForImageClassification,
)

from core.utils.extractor_utils import HFTextFeatureExtractor, HFVisionFeatureExtractor

config = [
    ["FakeSV", "google-bert/bert-base-multilingual-cased", "mean"],
    ["FakeTT", "google-bert/bert-base-multilingual-cased", "mean"],
    ["FVC", "google-bert/bert-base-multilingual-cased", "mean"],
]

save_name = "fea_audio"
dataset_dir_base = "data/"


class MyTextDataset(Dataset):
    def __init__(self, dataset_dir):
        self.data_df = pd.read_json(os.path.join(dataset_dir, "data.jsonl"), lines=True, dtype={"vid": "str"})

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        vid = self.data_df.loc[index, "vid"]

        title = self.data_df.loc[index, "title"]
        trans = self.data_df.loc[index, "transcript"]
        ocr = self.data_df.loc[index, "ocr"]

        texts = [title, trans, ocr]
        return vid, texts


def collate_fn(batch):
    vids, texts = zip(*batch)

    titles = [text[0] for text in texts]
    trans = [text[1] for text in texts]
    ocr = [text[2] for text in texts]

    return vids, titles, trans, ocr


for cfg in config:
    dataset_name, model_id, select_method = cfg
    max_length = 512
    print(f"Processing dataset: {dataset_name}")

    dataset_dir = os.path.join(dataset_dir_base, dataset_name)
    output_file = os.path.join(dataset_dir, "model", f"Backbone/{save_name}.pt")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Loading model: {model_id}")
    extractor = HFTextFeatureExtractor(model_id, device="cuda" if torch.cuda.is_available() else "cpu")

    dataset = MyTextDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=8)

    features = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding texts for {dataset_name}"):
            vids, titles, trans, ocr = batch

            inputs_trans = extractor.preprocess(
                trans, padding="max_length", return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)

            cls_trans = extractor.extract(**inputs_trans, select_method=select_method)

            cls_combined = cls_trans
            cls_combined = cls_combined.cpu()

            for i, vid in enumerate(vids):
                features[vid] = cls_combined[i]

    print(f"Saving features to {output_file}")
    torch.save(features, output_file)
    print(f"Finished processing dataset: {dataset_name}\n")
