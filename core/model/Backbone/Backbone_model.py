import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 1. Feature-specific FFN ----------
class FeatureFFN(nn.Module):
    """
    A general two-layer feedforward network with internal BatchNorm.
    Default hidden layer = in_dim, you can adjust freely.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden_dim or in_dim
        self.ffn = nn.Sequential(
            nn.LazyLinear(hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),  # Add BN again for convenient TTA statistics update
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: (B, D)  Single modal single sample features
        Returns: (B, out_dim)
        """
        return self.ffn(x)


# ---------- 2. Transformer Aggregation ----------
class ModalityAggregator(nn.Module):
    """
    Concatenate text/image features into sequences, use Transformer Encoder for interactive aggregation.
    """

    def __init__(
        self, emb_dim: int, num_layers: int = 1, num_heads: int = 4, ff_dim: int = 2048, dropout: float = 0
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,  # Convenient for (B, T D) input
            norm_first=False,  # Keep consistent with modern implementation
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_seq, return_attn=False):
        """
        x_seq: (B, T=2, D)  Currently T=2=[text, vision], can be extended to multimodal
        Returns: (B, D) Global pooled representation
        """
        if return_attn:
            attention_weights = []
            h = x_seq

            for layer in self.encoder.layers:
                h_new, attn_weights = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=False)
                h_new = layer.dropout1(h_new)
                h = layer.norm1(h + h_new)

                h_new = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
                h_new = layer.dropout2(h_new)
                h = layer.norm2(h + h_new)

                attention_weights.append(attn_weights)

            final_attention = attention_weights[-1]  # (B, num_heads, T, T)
            final_attention = final_attention.mean(dim=1)  # (B, T, T)

            pooled = h.mean(dim=1)  # (B, D)
            return pooled, final_attention
        else:
            h = self.encoder(x_seq)  # (B, T, D)
            # Simple CLS pooling; can also use mean/max
            pooled = h.mean(dim=1)  # Use mean pooling across all tokens
            return pooled


# ---------- 3. MLP Classifier ----------
class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = None, dropout=0.1):
        super().__init__()
        hidden = hidden or in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            # nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            # nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.mlp(x)


# ---------- 4. Overall Model ----------
class Backbone(nn.Module):
    """
    Input:
        text_fea:  (B, D_t)  Exported from CLIP text tower
        vision_fea:(B, D_v)  Exported from CLIP image tower
    """

    def __init__(
        self,
        text_dim: int = 128,
        vision_dim: int = 128,
        audio_dim: int = 128,
        emb_dim: int = 128,
        hidden_dim_ffn: int = 512,
        num_classes: int = 2,
        agg_layers: int = 2,
        agg_heads: int = 8,
        dropout: float = 0.1,
        bn_momentum: float = 0.0,
        learning_rate_src: float = 1e-4,
        **kwargs,
    ):
        super().__init__()
        self.txt_ffn = FeatureFFN(text_dim, emb_dim, hidden_dim_ffn, dropout)
        self.vis_ffn = FeatureFFN(vision_dim, emb_dim, hidden_dim_ffn, dropout)
        self.audio_ffn = FeatureFFN(audio_dim, emb_dim, hidden_dim_ffn, dropout)
        self.aggregator = ModalityAggregator(
            emb_dim, num_layers=agg_layers, num_heads=agg_heads, dropout=dropout
        )
        self.classifier = ClassifierHead(emb_dim, num_classes, dropout=dropout)

    def forward(self, **inputs):
        """
        Maintains original forward interface and output format
        """
        text_fea = inputs["text_fea"]
        vision_fea = inputs["vision_fea"]
        audio_fea = inputs["audio_fea"]
        adapt = inputs.get("adapt", None)

        # Handle vision feature averaging if needed
        if vision_fea.dim() == 3:  # (B, T, D)
            vision_fea = vision_fea.mean(1)  # (B, D)

        txt = self.txt_ffn(text_fea)  # (B, E)
        vis = self.vis_ffn(vision_fea)  # (B, E)
        audio = self.audio_ffn(audio_fea)  # (B, E)
        txt_copy = txt.clone()
        vis_copy = vis.clone()
        audio_copy = audio.clone()
        # Concatenate into sequence [txt, vis, audio]
        seq = torch.stack([txt, audio, vis], dim=1)  # (B, 3, E)

        if adapt == "ABPEM":
            fused, attn_matrix = self.aggregator(seq, return_attn=True)  # (B, E)
        else:
            fused = self.aggregator(seq)  # (B, E)
            attn_matrix = None

        logits = self.classifier(fused)  # (B, C)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        return {
            "logits": logits,
            "probs": probs,
            "pred": pred,
            "text_fea": txt_copy,
            "vision_fea": vis_copy,
            "audio_fea": audio_copy,
            "attn_matrix": attn_matrix,
        }
