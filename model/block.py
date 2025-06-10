import math
import typing

import torch
import torch.nn as nn
from torch import Tensor

from .attn import SelfAttention
from .ffwd import FeedForward


class TransformerBlock(nn.Module):
    '''
    A single transformer block consisting of a multi-head self-attention as the
    communication mechanism and a feedforward neural network as the computation
    mechanism. Each of these layers is preceded by layer normalization and
    followed by a residual connection.
    '''

    layer_norm_1: nn.LayerNorm
    '''
    First layer normalization, applied before self-attention.
    '''

    attention: SelfAttention
    '''
    Multi-head causal self-attention.
    '''

    layer_norm_2: nn.LayerNorm
    '''
    Second layer normalization, applied after self-attention and before
    feed-forward computation.
    '''

    feed_forward: FeedForward
    '''
    Position-wise feedforward neural network (multi-layer perceptron).
    '''

    def __init__(
        self,
        number_of_heads: int,
        embedding_dimension: int,
        block_size: int,
        dropout_probability: float,
        device: typing.Optional[torch.device] = None
    ):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            embedding_dimension,
            bias=False,
            device=device
        )

        self.attention = SelfAttention(
            number_of_heads,
            embedding_dimension,
            block_size,
            dropout_probability,
            device
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_dimension,
            bias=False,
            device=device
        )

        self.feed_forward = FeedForward(
            embedding_dimension,
            dropout_probability,
            device
        )

    def forward(self, x: Tensor) -> Tensor:
        # Apply layer normalization before self-attention
        first_normalization = self.layer_norm_1(x)

        # Apply self-attention and include a residual connection
        x = x + self.attention(first_normalization)

        # Apply layer normalization before feedforward neural network
        second_normalization = self.layer_norm_2(x)

        # Apply feedforward neural network with a residual connection
        x = x + self.feed_forward(second_normalization)

        return x
