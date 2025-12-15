from typing import List

from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer


class TextFeatureExtractor:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def preprocess(self, texts: List[str], **kwargs):
        raise NotImplementedError

    def extract(self, **inputs):
        raise NotImplementedError

    def delete(self):
        del self.model
        del self.tokenizer


class VisionFeatureExtractor:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def preprocess(self, images: List[Image.Image], **kwargs):
        raise NotImplementedError

    def extract(self, **inputs):
        raise NotImplementedError

    def delete(self):
        del self.model
        del self.processor


class HFTextFeatureExtractor(TextFeatureExtractor):
    def __init__(self, model_id: str, device: str = "cuda"):
        super().__init__(model_id)
        self.model = AutoModel.from_pretrained(model_id, device_map=device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if hasattr(self.model, "text_model"):
            self.model = self.model.text_model

    def preprocess(self, texts: List[str], **kwargs):
        return self.tokenizer(texts, **kwargs)

    def extract(
        self,
        select_method: str = "cls",
        **inputs,
    ):
        outputs = self.model(**inputs)
        if select_method == "cls":
            return outputs.last_hidden_state[:, 0, :]
        elif select_method == "mean":
            return outputs.last_hidden_state.mean(dim=1)
        elif select_method == "pooler_output":
            if hasattr(outputs, "pooler_output"):
                return outputs.pooler_output
            else:
                # print available keys
                print(f"Available keys: {outputs.keys()}")
                raise ValueError(f"Invalid select_method: {select_method}")
        else:
            raise ValueError(f"Invalid select_method: {select_method}")


class HFVisionFeatureExtractor(VisionFeatureExtractor):
    def __init__(self, model_id: str, device: str = "cuda"):
        super().__init__(model_id)
        self.model = AutoModel.from_pretrained(model_id, device_map=device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        if hasattr(self.model, "vision_model"):
            self.model = self.model.vision_model

    def preprocess(self, images: List[Image.Image], **kwargs):
        return self.processor(images=images, **kwargs)

    def extract(
        self,
        select_method: str = "cls",
        **inputs,
    ):
        outputs = self.model(**inputs)
        if select_method == "cls":
            return outputs.last_hidden_state[:, 0, :]
        elif select_method == "mean":
            return outputs.last_hidden_state.mean(dim=1)
        elif select_method == "pooler_output":
            return outputs.pooler_output
        else:
            raise ValueError(f"Invalid select_method: {select_method}")
