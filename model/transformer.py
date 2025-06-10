import math
import typing

import torch
import torch.nn as nn
from torch import Tensor

from block import TransformerBlock


class Transformer(nn.Module):

    token_embedding: nn.Embedding

    positional_embedding: nn.Embedding

    dropout: nn.Dropout

    blocks: nn.ModuleList

    final_layer_norm: nn.LayerNorm

    language_modelling_head: nn.Linear

    def __init__(
        self,
        number_of_layers: int,
        vocab_size: int,
        number_of_heads: int,
        embedding_dimension: int,
        block_size: int,
        dropout_probability: float,
        device: typing.Optional[torch.device] = None
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dimension,
            device=device
        )

        self.positional_embedding = nn.Embedding(
            block_size,
            embedding_dimension,
            device=device
        )

        self.dropout = nn.Dropout(dropout_probability)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                number_of_heads,
                embedding_dimension,
                block_size,
                dropout_probability,
                device
            )
            for _ in range(number_of_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(
            embedding_dimension,
            bias=False,
            device=device
        )

        self.language_modelling_head = nn.Linear(
            embedding_dimension,
            vocab_size,
            bias=False,
            device=device
        )

        # Tie together the weights of the token embedding table and the language
        # modelling head
        self.token_embedding.weight = self.language_modelling_head.weight

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
