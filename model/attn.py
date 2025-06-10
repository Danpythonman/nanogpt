import math
import typing

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from .exception import InconsistentDimensionalityException


class SelfAttention(nn.Module):
    '''
    Multi-head self-attention to be used in a transformer.
    '''

    embedding_dimension: int
    '''
    The input and output dimensionality of the self-attention layer.
    '''

    number_of_heads: int
    '''
    Number of heads used in self-attention. Each head processes a portion of the
    embedding dimension.
    '''

    attention_weights: nn.Linear
    '''
    The linear layer that projects the input into concatenated query, key, and
    value vectors. Shape is (B, T, embedding_dimension * 3).
    '''

    linear_projection: nn.Linear
    '''
    The linear layer that projects the output of the attention heads back to the
    original embedding dimension.
    '''

    attention_dropout: nn.Dropout
    '''
    Dropout layer applied after softmax.
    '''

    residual_dropout: nn.Dropout
    '''
    Dropout layer applied to the final output of self-attention before adding to
    the residual path.
    '''

    bias: Tensor
    '''
    The lower-triangular causal mask to be applied when calculating attention
    scores. It prevents attention from being applied to future tokens.
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

        if embedding_dimension % number_of_heads != 0:
            raise InconsistentDimensionalityException(
                '`embedding_dimension mod number_of_heads` must be 0. '
                f'Currently it is {embedding_dimension} mod {number_of_heads} '
                f'= {embedding_dimension % number_of_heads}'
            )

        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads

        self.attention_weights = nn.Linear(
            embedding_dimension,
            embedding_dimension * 3,
            bias=False,
            device=device
        )

        self.linear_projection = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False,
            device=device
        )

        self.attention_dropout = nn.Dropout(dropout_probability)

        self.residual_dropout = nn.Dropout(dropout_probability)

        self.register_buffer(
            'bias',
            torch.tril(
                torch.ones(block_size, block_size, device=device)
            ).view(1, 1, block_size, block_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        # B - batch size
        # T - sequence length (time dimension)
        # C - embedding dimension (channel length)
        B, T, C = x.shape

        # Apply self attention weights to `x`
        concatenated_attentions: Tensor = self.attention_weights(x)

        # Split the concatenated attentions into query, key, and value tensors
        q, k, v = concatenated_attentions.split(self.embedding_dimension, dim=2)

        # Calculate head size as the channel length divided by the number of
        # heads
        head_size = C // self.number_of_heads

        q = q.view(
            B,
            T,
            self.number_of_heads,
            head_size
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.number_of_heads,
            head_size
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.number_of_heads,
            head_size
        ).transpose(1, 2)

        # At this point, `q`, `k`, and `v` are of shape (B, nh, T, hs), where nh
        # is the number of heads and hs is the head size

        # Calculate attention weights by:
        #   1. Computing the dot product of query and key vectors (raw attention
        #      scores)
        #   2. Scaling down by the square root of head size to stabilize
        #      gradients
        #   3. Masking out future tokens via a lower-triangular causal mask
        #   4. Applying softmax to normalize attention weights
        #   5. Applying dropout
        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        attention = attention.masked_fill(
            self.bias[:, :, :T, :T] == 0,
            float('-inf')
        )
        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)

        # At this point `attention` has shape (B, nh, T, T)

        # Apply attention weights to the value vectors, which gives us the
        # context layer. `y` is a tensor of shape (B, T, nh, hs)
        y: Tensor = attention @ v

        # Reassemble all head outputs side-by-side. `y` becomes a tensor of
        # shape (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Apply linear projection and dropout
        y = self.linear_projection(y)
        y = self.residual_dropout(y)

        return y
