# -*- coding: utf-8 -*-
from typing import Any, Dict, List

Batch = List


class BaseSelection(object):
    def __init__(self, meta_conf, model_adaptation_method):
        self.device = "cuda"
        self.meta_conf = meta_conf
        self.model = model_adaptation_method.copy_model()
        self.model.to(self.device)

        self.initialize()

    def initialize(self):
        pass

    def clean_up(self):
        pass

    def save_state(self):
        pass

    def select_state(
        self,
        current_batch: Batch,
        previous_batches: List[Batch],
    ) -> Dict[str, Any]:
        pass


class LastIterate(BaseSelection):
    """Naively return the model generated from the last iterate of adaptation."""

    def __init__(self, meta_conf, model_adaptation_method):
        super().__init__(meta_conf, model_adaptation_method)

    def initialize(self):
        if hasattr(self.model, "ssh"):
            self.model.ssh.eval()
            self.model.main_model.eval()
        else:
            self.model.eval()

        self.optimal_state = None

    def clean_up(self):
        self.optimal_state = None

    def save_state(self, state, current_batch):
        self.optimal_state = state

    def select_state(self) -> Dict[str, Any]:
        """return the optimal state and sync the model defined in the model selection method."""
        return self.optimal_state

    @property
    def name(self):
        return "last_iterate"


def get_selection_method(selection_method_name: str, cfg, adaptat_method):
    match selection_method_name:
        case "last_iterate":
            return LastIterate(cfg, adaptat_method)
        case _:
            raise ValueError(f"Selection method {selection_method_name} not implemented")
