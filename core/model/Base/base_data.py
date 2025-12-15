from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset


class Base_Dataset(Dataset):
    def __init__(self, **kargs):
        super().__init__()
        self.data_path = Path("data")

    def _get_data(self, fold: int, split: str, task: str = "binary"):
        raise NotImplementedError

    def _get_event_split_data(self, split: str):
        """
        Split data by events with ratio 70:15:15 (train:val:test) using seed 2025
        """
        data = self._get_complete_data()

        # Get unique events
        unique_events = data["event"].unique()

        # Set random seed for reproducibility
        np.random.seed(2025)

        # Shuffle events
        shuffled_events = np.random.permutation(unique_events)

        # Calculate split indices
        n_events = len(shuffled_events)
        train_end = int(0.7 * n_events)
        val_end = int(0.85 * n_events)  # 0.7 + 0.15 = 0.85

        train_events = shuffled_events[:train_end]
        val_events = shuffled_events[train_end:val_end]
        test_events = shuffled_events[val_end:]

        if split == "train":
            data = data[data["event"].isin(train_events)]
        elif split == "valid":
            data = data[data["event"].isin(val_events)]
        elif split == "test":
            data = data[data["event"].isin(test_events)]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        return data

    def _get_event_wise_data(self):
        """
        Split data by event sequence order - events are ordered chronologically
        without any train/val/test split
        """
        data = self._get_complete_data()

        # Sort data by event to maintain chronological order
        data_sorted = data.sort_values("event").reset_index(drop=True)

        return data_sorted

    def _get_random_split_data(self, split: str):
        """
        Randomly split data with ratio 70:15:15 (train:val:test) using seed 2025
        """
        data = self._get_complete_data()

        # Set random seed for reproducibility
        np.random.seed(2025)

        # Get all indices and shuffle them
        indices = np.arange(len(data))
        shuffled_indices = np.random.permutation(indices)

        # Calculate split indices
        n_samples = len(shuffled_indices)
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)

        # Split indices
        if split == "train":
            selected_indices = shuffled_indices[:train_end]
        elif split == "valid":
            selected_indices = shuffled_indices[train_end:val_end]
        elif split == "test":
            selected_indices = shuffled_indices[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}")

        # Filter data by selected indices
        filtered_data = data.iloc[selected_indices].reset_index(drop=True)
        return filtered_data


class FakeSV_Dataset(Base_Dataset):
    def __init__(self, **kargs):
        super(FakeSV_Dataset, self).__init__()
        self.data_path = Path("data/FakeSV")

    def _get_complete_data(self):
        data_complete = pd.read_json(
            "./data/FakeSV/data_complete.jsonl", orient="records", dtype=False, lines=True
        )
        replace_values = {"辟谣": 2, "假": 1, "真": 0}
        data_complete["label"] = data_complete["annotation"].replace(replace_values)
        data_complete = data_complete[data_complete["label"] != 2]
        data_complete["event"], _ = pd.factorize(data_complete["keywords"])
        data_complete["vid"] = data_complete["video_id"]
        return data_complete

    def _get_data(self, fold, split, task="binary"):
        if fold == "default":
            data = self._get_event_split_data(split)
        elif fold == "all":
            data = self._get_complete_data()
        elif fold == "event_wise":
            data = self._get_event_wise_data()
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data


class FakeTT_Dataset(Base_Dataset):
    def __init__(self, **kargs):
        super(FakeTT_Dataset, self).__init__()
        self.data_path = Path("data/FakeTT")

    def _get_complete_data(self):
        data = pd.read_json(
            "data/FakeTT/data_complete.jsonl", orient="records", lines=True, dtype={"video_id": "str"}
        )
        replace_values = {"fake": 1, "real": 0}
        data["label"] = data["annotation"].replace(replace_values)
        data["event"], _ = pd.factorize(data["event"])
        data["vid"] = data["video_id"]
        data["title"] = data["description"]
        # set type of video_id to str
        return data

    def _get_data(self, fold, split, task="binary"):
        if fold == "default":
            data = self._get_event_split_data(split)
        elif fold == "all":
            data = self._get_complete_data()
        elif fold == "event_wise":
            data = self._get_event_wise_data()
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data


class FVC_Dataset(Base_Dataset):
    def __init__(self, **kargs):
        super(FVC_Dataset, self).__init__()
        self.data_path = Path("data/FVC")

    def _get_complete_data(self):
        data = pd.read_json(
            "data/FVC/data_complete.jsonl", orient="records", lines=True, dtype={"vid": "str"}
        )
        data = data[data["label"].isin(["fake", "real"])]
        replace_values = {"fake": 1, "real": 0}
        data["label"] = data["label"].replace(replace_values)
        data["event"], _ = pd.factorize(data["event_id"])
        data["video_id"] = data["vid"]
        return data

    def _get_data(self, fold, split, task="binary"):
        if fold == "default":
            data = self._get_event_split_data(split)
        elif fold == "all":
            data = self._get_complete_data()
        elif fold == "event_wise":
            data = self._get_event_wise_data()
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data
