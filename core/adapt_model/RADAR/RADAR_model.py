import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from utils.adapt_utils import softmax_entropy
from utils.advance_utils import fork_rng_with_seed
from utils.core_adapt_utils import compute_grad_norm, print_param_frozen
from utils.select_method import BaseSelection

from core.adapt_model.Base.base_model import BaseAdaptModel


class RADAR(BaseAdaptModel):
    def __init__(self, src_model: nn.Module, cfg: DictConfig, **kargs):
        super(RADAR, self).__init__(src_model, cfg, **kargs)
        self.cfg = cfg
        self.fishers = None  # Initialize fishers attribute
        self.num_retrieve = cfg.para.num_retrieve
        self.retrieval_entropy_threshold = cfg.para.entropy_threshold
        self.align_loss_weight = cfg.para.align_loss_weight

        self.memory_size = cfg.para.get("memory_size", 768)

    def _init_src_model(self, model: nn.Module):
        model.train()
        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.requires_grad_(True)
                # bn module always uses batch statistics, in both training and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
            if self.cfg.para.unfreeze_encoder:
                parts = n.split(".")
                if len(parts) > 2 and parts[1] == "ffn" and parts[2] == "3":
                    m.requires_grad_(True)
        return model

    def _initialize_trainable_parameters(self):
        """
        Collect the affine scale + shift parameters from norm layers.

        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        self._adapt_module_names = []
        adapt_params = []
        adapt_param_names = []

        for n, m in self.src_model.named_modules():
            if isinstance(
                m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)
            ):  # only bn is used in the paper.
                self._adapt_module_names.append(n)
                for name_parameter, parameter in m.named_parameters():
                    if name_parameter in ["weight", "bias"]:
                        adapt_params.append(parameter)
                        adapt_param_names.append(f"{n}.{name_parameter}")
            if self.cfg.para.unfreeze_encoder:
                parts = n.split(".")
                if len(parts) > 2 and parts[1] == "ffn" and parts[2] == "3":
                    self._adapt_module_names.append(n)
                    for name_parameter, parameter in m.named_parameters():
                        if name_parameter in ["weight", "bias"]:
                            adapt_params.append(parameter)
                            adapt_param_names.append(f"{n}.{name_parameter}")
        # remove duplicate
        self._adapt_module_names = list(set(self._adapt_module_names))
        adapt_params = list(set(adapt_params))
        adapt_param_names = list(set(adapt_param_names))
        assert len(self._adapt_module_names) > 0, "RADAR needs some adaptable model parameters."
        return adapt_params, adapt_param_names

    def align_loss_prototype(self, src_fea, tgt_fea_list, tgt_entropy_list=None):
        """
        Compute prototype-based alignment loss using cosine similarity.

        For each batch element b:
            • Compute prototype c = weighted_mean(tgt_fea_list[b], weights)
            • weights derived from entropy: low entropy → high weight
            • Minimize: loss = 1 - cos(v, c)

        Args:
            src_fea: Tensor [B, D] - source features
            tgt_fea_list: List of B tensors, each [K, D] - retrieved target features
            tgt_entropy_list: Optional list of B tensors, each [K] - entropy values for weighting

        Returns:
            Scalar tensor: Mean cosine distance loss across batch
        """
        B, D = src_fea.shape
        total_loss = 0.0

        for b in range(B):
            # Get target samples [K, D]
            tgt_samples = tgt_fea_list[b].detach()  # Stop gradient
            K = tgt_samples.shape[0]

            # Compute weights from entropy
            if tgt_entropy_list is not None and tgt_entropy_list[b] is not None:
                entropy = tgt_entropy_list[b]  # [K]
                # Convert entropy to weights: lower entropy → higher weight
                # Use softmax with negative entropy for numerical stability
                weights = F.softmax(-entropy, dim=0)  # [K]
            else:
                # Uniform weights if no entropy provided
                weights = torch.ones(K, device=tgt_samples.device) / K  # [K]

            # Compute weighted prototype centroid
            # c = sum(w_i * v_i) where sum(w_i) = 1
            prototype = (weights.unsqueeze(1) * tgt_samples).sum(dim=0)  # [D]

            # Get source sample
            src_sample = src_fea[b]  # [D]

            # Compute cosine similarity
            cos_sim = F.cosine_similarity(src_sample.unsqueeze(0), prototype.unsqueeze(0), dim=1)

            # Loss = 1 - cosine_similarity (range [0, 2])
            loss = 1.0 - cos_sim

            total_loss += loss

        # Average over batch
        return total_loss / B

    def one_adapt_step(self, batch, retrieved_dict, step):
        # NOTE: batch: B * {vid, text_fea, vision_fea, label}
        # NOTE: retrieved_items: List[Dict] where each dict contains batch {vid, label, text_fea, vision_fea, similarity}
        # NOTE: entropy_threshold: Only items with entropy < threshold will be selected for adaptation
        selected_retrieved_dict = {}
        ratio_selected = []
        # Process each batch item's retrieved candidates
        align_idx = []
        vid2idx = {}
        idx2vid = {}
        with torch.no_grad():
            with fork_rng_with_seed(self.cfg.seed):
                outputs = self.src_model(**batch)
        logits = outputs["logits"]

        for i, (query_vid, item_retrieved) in enumerate(retrieved_dict.items()):
            # Create a proper batch format for model inference
            retrieved_batch = {
                "vids": item_retrieved["vid"],
                "labels": item_retrieved["label"],
                "text_fea": item_retrieved["text_fea"],
                "vision_fea": item_retrieved["vision_fea"],
                "audio_fea": item_retrieved["audio_fea"],
            }
            vid2idx[query_vid] = i
            idx2vid[i] = query_vid
            # forward for retrieved items
            with torch.no_grad():
                with fork_rng_with_seed(self.cfg.seed):
                    outputs = self.src_model(**retrieved_batch)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
            entropy = softmax_entropy(logits)  # [num_retrieve]

            # Select items with entropy below threshold
            low_entropy_mask = entropy < self.retrieval_entropy_threshold
            low_entropy_indices = torch.nonzero(low_entropy_mask, as_tuple=False).squeeze(-1)
            if len(low_entropy_indices) == 0:
                continue

            # Skip this batch item if no samples meet the entropy threshold
            ratio_selected.append(len(low_entropy_indices) / len(item_retrieved["vid"]))

            # Store low entropy items for potential use
            low_entropy_batch = {
                "vids": [item_retrieved["vid"][i] for i in low_entropy_indices],
                "labels": item_retrieved["label"][low_entropy_indices],
                "similarity": item_retrieved["similarity"][low_entropy_indices],
                "probs": probs[low_entropy_indices],
                "logits": logits[low_entropy_indices],
                "text_fea": outputs["text_fea"][low_entropy_indices].detach(),
                "vision_fea": outputs["vision_fea"][low_entropy_indices].detach(),
                "audio_fea": outputs["audio_fea"][low_entropy_indices].detach(),
                "p_label": item_retrieved["p_label"][low_entropy_indices],
                "entropy": entropy[low_entropy_indices],
            }
            selected_retrieved_dict[query_vid] = low_entropy_batch
            align_idx.append(i)

        wandb.log({"ratio_selected": sum(ratio_selected) / len(ratio_selected)}, commit=False)
        # Check if we have any selected samples to work with
        if len(selected_retrieved_dict) == 0:
            # No samples meet the entropy threshold, skip adaptation for this step
            with fork_rng_with_seed(self.cfg.seed):
                outputs = self.src_model(**batch)
            logits = outputs["logits"]

            return {
                "optimizer": copy.deepcopy(self.optimizer).state_dict(),
                "loss": 0.0,  # No adaptation loss
                "grads": {},  # No gradients computed
                "pred": logits,
                # "skipped": True,  # Flag to indicate adaptation was skipped
            }

        # Stack features for alignment loss computation
        text_fea_align_retrieved = [item["text_fea"] for item in selected_retrieved_dict.values()]
        vision_fea_align_retrieved = [item["vision_fea"] for item in selected_retrieved_dict.values()]
        audio_fea_align_retrieved = [item["audio_fea"] for item in selected_retrieved_dict.values()]
        text_entropy_align_retrieved = [item["entropy"] for item in selected_retrieved_dict.values()]
        vision_entropy_align_retrieved = [item["entropy"] for item in selected_retrieved_dict.values()]
        audio_entropy_align_retrieved = [item["entropy"] for item in selected_retrieved_dict.values()]

        # Compute similarity-weighted pseudo labels
        pseudo_labels_dict = dict()
        for i, (query_vid, retrieved_batch) in enumerate(selected_retrieved_dict.items()):
            # Get similarities and probabilities for this batch
            similarities = retrieved_batch["similarity"]  # [k]
            logits = retrieved_batch["logits"]  # [k, num_classes]
            probs = retrieved_batch["probs"]  # [k, num_classes]
            preds = torch.argmax(probs, dim=1)  # [k]
            labels = retrieved_batch["labels"].to(torch.float32)  # [k]
            p_labels = retrieved_batch["p_label"]  # [k]
            # replace p_labels with preds if p_labels is -1
            p_labels = torch.where(p_labels == -1, preds, p_labels)

            # Use weighted_probs aggregation
            temperature = 0.1
            weights = F.softmax(similarities / temperature, dim=0)  # [k]
            labels = (weights.unsqueeze(1) * probs).detach().sum(dim=0)  # [num_classes]
            pseudo_labels_dict[query_vid] = labels

        # for backward
        with fork_rng_with_seed(self.cfg.seed):
            outputs = self.src_model(**batch)
        logits = outputs["logits"]
        entropy_loss = softmax_entropy(logits).mean(0)

        # selectively merge self_label and pseudo_labels using the entropy of y_hat_selected
        self_labels = copy.deepcopy(logits.detach())
        entropy = softmax_entropy(logits)
        # entropy_threshold = self.self_label_entropy_threshold
        # Create self_label_dict with vid as key and label as value for samples that pass the mask filter
        self_label_dict = {batch["vids"][i]: self_labels[i] for i in range(len(batch["vids"]))}
        # merge self_label_dict and pseudo_labels_dict
        for vid, label in self_label_dict.items():
            if vid in pseudo_labels_dict:
                pseudo_labels_dict[vid] = (pseudo_labels_dict[vid] + label) / 2

        # make logits and labels using vid from pseudo_labels_dict and label from pseudo_labels_dict
        self_logits = torch.stack([outputs["logits"][vid2idx[vid]] for vid in pseudo_labels_dict.keys()])
        pseudo_labels = torch.stack([pseudo_labels_dict[vid] for vid in pseudo_labels_dict.keys()])
        ground_true_labels = torch.stack([batch["labels"][vid2idx[vid]] for vid in pseudo_labels_dict.keys()])
        self_labels = torch.stack([outputs["probs"][vid2idx[vid]] for vid in pseudo_labels_dict.keys()])

        self_acc = accuracy_score(
            torch.argmax(self_labels, dim=1).cpu().numpy(), ground_true_labels.cpu().numpy()
        )
        pseudo_acc = accuracy_score(
            torch.argmax(pseudo_labels, dim=1).cpu().numpy(), ground_true_labels.cpu().numpy()
        )
        # ic check whether self_logits have grads
        pred_loss = F.cross_entropy(
            self_logits,
            torch.argmax(torch.softmax(pseudo_labels.detach(), dim=1), dim=1),
            label_smoothing=0.1,
        )
        # Only compute losses for samples that have valid selected batches
        if len(align_idx) < logits.shape[0]:
            # Some samples were skipped, only use the first len(selected_batch) samples for loss computation
            text_fea_align_self = outputs["text_fea"][align_idx]
            vision_fea_align_self = outputs["vision_fea"][align_idx]
            audio_fea_align_self = outputs["audio_fea"][align_idx]
        else:
            text_fea_align_self = outputs["text_fea"]
            vision_fea_align_self = outputs["vision_fea"]
            audio_fea_align_self = outputs["audio_fea"]
        # calculate acc of pseudo_labels

        # calculate acc of self_label
        text_align_loss = self.align_loss_prototype(
            text_fea_align_self, text_fea_align_retrieved, text_entropy_align_retrieved
        )
        vision_align_loss = self.align_loss_prototype(
            vision_fea_align_self, vision_fea_align_retrieved, vision_entropy_align_retrieved
        )
        audio_align_loss = self.align_loss_prototype(
            audio_fea_align_self, audio_fea_align_retrieved, audio_entropy_align_retrieved
        )
        # conduct a log to align_loss
        align_loss = (text_align_loss + vision_align_loss + audio_align_loss) / 3
        align_loss = align_loss * self.align_loss_weight
        loss = align_loss + pred_loss + entropy_loss

        acc = accuracy_score(torch.argmax(logits, dim=1).cpu().numpy(), batch["labels"].cpu().numpy())
        wandb.log(
            {
                "pred_loss": pred_loss,
                "align_loss": align_loss,
                "step": step,
                "loss": loss,
                "runtime/acc": acc,
                "pseudo/acc": pseudo_acc,
                "self/acc": self_acc,
                "text_align/loss": text_align_loss,
                "vision_align/loss": vision_align_loss,
                "audio_align/loss": audio_align_loss,
            },
            commit=False,
        )

        loss.backward()
        grads = dict(
            (name, param.grad.clone().detach())
            for name, param in self.src_model.named_parameters()
            if param.grad is not None
        )
        grad_norm = compute_grad_norm(grads)
        wandb.log({"grad_norm": grad_norm})
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {
            "optimizer": copy.deepcopy(self.optimizer).state_dict(),
            "loss": loss.item(),
            "grads": grads,
            "pred": logits,
            # "skipped": False,  # Normal adaptation was performed
        }

    def retrieve(
        self, src_batch, available_batch_list, n=5, text_weight=0.33, vision_weight=0.33, audio_weight=0.33
    ):
        """
        Retrieve the most similar n items from available_batch_list for each item in src_batch.

        Args:
            src_batch: Source batch containing items to query
            available_batch_list: List of available batches to search from
            n: Number of most similar items to retrieve for each source item
            text_weight: Weight for text feature similarity
            vision_weight: Weight for vision feature similarity
            audio_weight: Weight for audio feature similarity

        Returns:
            List of retrieved similar items for each source item
        """

        # Extract features from source batch
        src_text_fea = src_batch["text_fea"]  # [src_batch_size, text_dim]
        src_vision_fea = src_batch["vision_fea"]  # [src_batch_size, vision_dim]
        src_audio_fea = src_batch["audio_fea"]  # [src_batch_size, audio_dim]
        src_batch_size = src_text_fea.shape[0]

        # Collect all available features
        all_text_fea = []
        all_vision_fea = []
        all_audio_fea = []
        all_vids = []
        all_labels = []
        all_p_labels = []
        batch_indices = []  # Track which batch each item comes from

        for batch_idx, batch in enumerate(available_batch_list):
            batch_size = batch["text_fea"].shape[0]
            all_text_fea.append(batch["text_fea"])
            all_vision_fea.append(batch["vision_fea"])
            all_audio_fea.append(batch["audio_fea"])
            all_vids.extend(batch["vids"])
            all_labels.append(batch["labels"])
            if "p_label" in batch:
                all_p_labels.extend(batch["p_label"])
            else:
                all_p_labels.extend(torch.full_like(batch["labels"], -1, dtype=torch.int64))
            batch_indices.extend([batch_idx] * batch_size)

        memory_size = self.memory_size
        all_vids = all_vids[-memory_size:]
        all_p_labels = all_p_labels[-memory_size:]
        batch_indices = batch_indices[-memory_size:]

        if not all_text_fea:
            return [[] for _ in range(src_batch_size)]

        # Stack all features
        all_text_fea = torch.cat(all_text_fea, dim=0)  # [total_available, text_dim]
        all_vision_fea = torch.cat(all_vision_fea, dim=0)  # [total_available, vision_dim]
        all_audio_fea = torch.cat(all_audio_fea, dim=0)  # [total_available, audio_dim]
        all_labels = torch.cat(all_labels, dim=0)  # [total_available]

        all_text_fea = all_text_fea[-memory_size:]
        all_vision_fea = all_vision_fea[-memory_size:]
        all_audio_fea = all_audio_fea[-memory_size:]
        all_labels = all_labels[-memory_size:]

        # Normalize features for cosine similarity
        src_text_fea_norm = F.normalize(src_text_fea, p=2, dim=1)
        src_vision_fea_norm = F.normalize(src_vision_fea, p=2, dim=1)
        src_audio_fea_norm = F.normalize(src_audio_fea, p=2, dim=1)
        all_text_fea_norm = F.normalize(all_text_fea, p=2, dim=1)
        all_vision_fea_norm = F.normalize(all_vision_fea, p=2, dim=1)
        all_audio_fea_norm = F.normalize(all_audio_fea, p=2, dim=1)

        # Compute similarities efficiently using matrix multiplication
        # Text similarity: [src_batch_size, total_available]
        text_sim = torch.mm(src_text_fea_norm, all_text_fea_norm.t())

        # Vision similarity: [src_batch_size, total_available]
        vision_sim = torch.mm(src_vision_fea_norm, all_vision_fea_norm.t())

        # Audio similarity: [src_batch_size, total_available]
        audio_sim = torch.mm(src_audio_fea_norm, all_audio_fea_norm.t())

        # Weighted combination of similarities
        combined_sim = text_weight * text_sim + vision_weight * vision_sim + audio_weight * audio_sim

        # Get top n similar items for each source item
        topk_values, topk_indices = torch.topk(combined_sim, k=min(n, combined_sim.shape[1]), dim=1)

        # Organize results
        retrieved_items = {}
        for src_idx in range(src_batch_size):
            similar_items = []
            for rank in range(topk_indices.shape[1]):
                retrieved_idx = topk_indices[src_idx, rank].item()
                similarity_score = topk_values[src_idx, rank].item()

                similar_items.append(
                    {
                        "vid": all_vids[retrieved_idx],
                        "label": all_labels[retrieved_idx],  # Keep as tensor, don't convert to scalar
                        "text_fea": all_text_fea[retrieved_idx],
                        "vision_fea": all_vision_fea[retrieved_idx],
                        "audio_fea": all_audio_fea[retrieved_idx],
                        "similarity": torch.tensor(
                            similarity_score, device=all_text_fea.device
                        ),  # Convert to tensor for consistency
                        "p_label": all_p_labels[retrieved_idx],
                        # "batch_idx": batch_indices[retrieved_idx],
                        # "item_idx": retrieved_idx,
                    }
                )
            similar_items = {
                "vid": [item["vid"] for item in similar_items],
                "label": torch.stack([item["label"] for item in similar_items]),
                "text_fea": torch.stack([item["text_fea"] for item in similar_items]),
                "vision_fea": torch.stack([item["vision_fea"] for item in similar_items]),
                "audio_fea": torch.stack([item["audio_fea"] for item in similar_items]),
                "similarity": torch.stack([item["similarity"] for item in similar_items]),
                "p_label": torch.stack([item["p_label"] for item in similar_items]),
            }
            retrieved_items[src_batch["vids"][src_idx]] = similar_items
            # logger debug the query vid and retrieved vids
            # logger.debug(f"Query vid: {src_batch['vids'][src_idx]}, Retrieved vids: {similar_items['vid']}")
            # logger.debug(
            #     f"Query label: {src_batch['labels'][src_idx]}, Retrieved labels: {similar_items['label']}, Right: {src_batch['labels'][src_idx].item() == max(similar_items['label'].tolist(), key=similar_items['label'].tolist().count)}"
            # )
            # logger query vid, retrieved vids, and similarity
            logger.debug(
                f"Query vid: {src_batch['vids'][src_idx]}, Retrieved vids: {similar_items['vid']}, Similarity: {similar_items['similarity']}"
            )
        return retrieved_items

    def adapt(self, batch, nbsteps: int, selection_method: BaseSelection, previous_batches: list):
        selection_method.initialize()
        batch_idx = len(previous_batches)

        retrieved_dict = self.retrieve(batch, previous_batches, self.num_retrieve)
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(batch, retrieved_dict, batch_idx)

            selection_method.save_state(
                {
                    "model": copy.deepcopy(self.src_model).state_dict(),
                    "step": step,
                    "lr": self.cfg.opt.lr,
                    **adaptation_result,
                },
                current_batch=batch,
            )

        optimal_state = selection_method.select_state()
        self.src_model.load_state_dict(optimal_state["model"])
        selection_method.clean_up()
        return optimal_state
