import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers import TransformerLayer
from labml_nn.utils import clone_module_list


class PatchEmbeddings(Module):
    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        bs, c, h, w = x.shape
        x = x.permute(2, 3, 0, 1)
        x = x.view(h * w, bs, c)

        return x


class LearnedPositionalEmbeddings(Module):
    def __init__(self, d_model: int, max_len: int = 5_000):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]]
        return x + pe


class ClassificationHead(Module):

    def __init__(self, d_model: int, n_hidden: int, n_classes: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, n_hidden)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x: torch.Tensor):
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return x


class VisionTransformer(Module):
    def __init__(self, transformer_layer: TransformerLayer, n_layers: int,
                 patch_emb: PatchEmbeddings, pos_emb: LearnedPositionalEmbeddings,
                 classification: ClassificationHead):
        super().__init__()
        # Patch embeddings
        self.patch_emb = patch_emb
        self.pos_emb = pos_emb
        # Classification head
        self.classification = classification
        self.transformer_layers = clone_module_list(transformer_layer, n_layers)
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, transformer_layer.size), requires_grad=True)
        self.ln = nn.LayerNorm([transformer_layer.size])

    def forward(self, x: torch.Tensor):
        x = self.patch_emb(x)
        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x])
        x = self.pos_emb(x)
        for layer in self.transformer_layers:
            x = layer(x=x, mask=None)
        x = x[0]
        x = self.ln(x)
        x = self.classification(x)
        return x