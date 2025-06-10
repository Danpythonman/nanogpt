import typing

import torch
import torch.nn as nn
from torch import Tensor


class FeedForward(nn.Module):
    '''
    Multi-layer perceptron to be used as the feed-forward computation section of
    transformer blocks. Should be applied after the self-attention communication
    section of the blocks.
    '''

    linear_full: nn.Linear
    '''
    The first linear layer of the feed-forward network. This will expand the
    embedding dimension by a factor of 4.
    '''

    gelu: nn.GELU
    '''
    The Gaussian error linear units layer of the feed-forward network. This
    applies a non-linearity before contracting back to the original embedding
    dimensionality.
    '''

    linear_projection: nn.Linear
    '''
    The last linear layer of the feed-forward network. This will contract the
    embedding dimension by a factor of 4, returning to the original embedding
    dimensionality.
    '''

    dropout: nn.Dropout
    '''
    The dropout layer of the feed-forward network to combat overfitting during
    training.
    '''

    def __init__(
        self,
        embedding_dimension: int,
        dropout_probability: float,
        device: typing.Optional[torch.device] = None
    ):
        super().__init__()

        self.linear_full = nn.Linear(
            embedding_dimension,
            embedding_dimension * 4,
            bias=False,
            device=device
        )

        self.gelu = nn.GELU()

        self.linear_projection = nn.Linear(
            embedding_dimension * 4,
            embedding_dimension,
            bias=False,
            device=device
        )

        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_full(x)
        x = self.gelu(x)
        x = self.linear_projection(x)
        x = self.dropout(x)
        return x
