import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from regex import F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel

from core.utils.extractor_utils import HFVisionFeatureExtractor

config = [
    ["FakeSV", "google/vit-base-patch16-224", "mean"],
    ["FakeTT", "google/vit-base-patch16-224", "mean"],
    ["FVC", "google/vit-base-patch16-224", "mean"],
]
save_name = "fea_vision"

dataset_dir_base = "data"
frames_path = "frames_16"


class MyDataset(Dataset):
    def __init__(self, dataset_dir):
        data_df = pd.read_json(os.path.join(dataset_dir, "data.jsonl"), lines=True, dtype={"vid": "str"})
        self.vids = data_df["vid"].tolist()

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]
        frames = []
        for i in range(16):
            frame_path = os.path.join(dataset_dir, frames_path, f"{vid}", f"frame_{i:03d}.jpg")
            if os.path.exists(frame_path):
                frame = Image.open(frame_path).convert("RGB")
                frames.append(frame)
            else:
                frames.append(Image.new("RGB", (224, 224), color="black"))
        return vid, frames


def collate_fn(batch):
    vids, all_frames = zip(*batch)
    all_frames = [frame for frames in all_frames for frame in frames]
    processed_frames = extractor.preprocess(all_frames, padding=True, return_tensors="pt")
    return vids, processed_frames


for cfg in config:
    dataset_name, model_id, select_method = cfg
    print(f"Processing dataset: {dataset_name}")

    dataset_dir = os.path.join(dataset_dir_base, dataset_name)
    output_file = os.path.join(dataset_dir, "model", f"Backbone/{save_name}.pt")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Loading model: {model_id}")
    extractor = HFVisionFeatureExtractor(model_id, device="cuda" if torch.cuda.is_available() else "cpu")

    dataset = MyDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=8)

    features = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for vids, processed_frames in tqdm(dataloader, desc=f"Extracting features for {dataset_name}"):
            bs = len(vids)
            inputs = {k: v.to(device) for k, v in processed_frames.items()}
            cls_features = extractor.extract(**inputs, select_method=select_method)

            cls_features = cls_features.view(bs, 16, -1)  # Shape: (batch_size, 16, hidden_dim)

            for i, vid in enumerate(vids):
                features[vid] = cls_features[i].cpu()

    print(f"Saving features to {output_file}")
    torch.save(features, output_file)
    print(f"Finished processing dataset: {dataset_name}\n")
