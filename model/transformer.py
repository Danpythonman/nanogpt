import typing

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from .block import TransformerBlock
from .exception import InconsistentDimensionalityException


class Transformer(nn.Module):
    '''
    Transformer architecture to train on text and predict new text.
    '''

    block_size: int
    '''
    The size of a block that the transformer sees. Also called "context length".
    '''

    token_embedding: nn.Embedding
    '''
    Embedding layer for tokens. Encodes each token as a vector.
    '''

    positional_embedding: nn.Embedding
    '''
    Embedding layer for positions. Encodes each position as a vector.
    '''

    dropout: nn.Dropout
    '''
    Dropout layer to prevent overfitting during training.
    '''

    blocks: nn.ModuleList
    '''
    Stack of transformer blocks, each composed of:
      1. Multi-head self-attention (communication)
      2. Feedforward MLP (computation)
    Each block is preceded by layer normalization and followed by a residual
    connection.
    '''

    final_layer_norm: nn.LayerNorm
    '''
    Layer normalization to be applied after the transformer blocks.
    '''

    language_modelling_head: nn.Linear
    '''
    Final layer to project model outputs to vocabulary logits.
    '''

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

        self.block_size = block_size

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

    def forward(
        self,
        idx: Tensor,
        targets: typing.Optional[Tensor] = None
    ) -> typing.Tuple[Tensor, typing.Optional[Tensor]]:
        B, T = idx.shape

        if T > self.block_size:
            raise InconsistentDimensionalityException(
                f'Cannot forward sequence of length {T} because block size is '
                f'only of length {self.block_size}'
            )

        # Token embeddings of shape (B, T, n_embd)
        token_emb = self.token_embedding(idx)

        # Vector of position indices of shape (T,)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Positional embeddings of shape (T, n_embd)
        position_emb = self.positional_embedding(pos)

        # Add token embeddings and positional embeddings and apply dropout
        x = self.dropout(token_emb + position_emb)

        # Apply each of the transformer blocks to the input
        for block in self.blocks:
            x = block(x)

        # Apply the final layer normalization
        x = self.final_layer_norm(x)

        if targets is not None:
            # Training
            #
            # Since we have targets, we can calculate the loss.
            logits: Tensor = self.language_modelling_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference
            #
            # Note that doing `x[:, -1, :]` would result in the second dimension
            # being dropped. When we do `x[:, [-1], :]`, we get the same value
            # (everything from the last time entry) while keeping the time
            # dimension intact (so the size becomes (B, 1, vocab_size)).
            logits: Tensor = self.language_modelling_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: typing.Optional[int] = None
    ) -> Tensor:
        '''
        Take a sequence of indices `idx` of shape (B, T) and complete the
        sequence `max_new_tokens` times, feeding the predictions back into the
        model each time. The output is of shape (B, T + max_new_tokens).
        '''

        for _ in range(max_new_tokens):
            # Truncate context to the last `block_size` tokens if needed
            if idx.shape[1] <= self.block_size:
                idx_cond = idx
            else:
                idx_cond = idx[:, -self.block_size:]

            # Apply the model to `idx`
            logits, _ = self(idx_cond)

            # Cast logits to tensor to preserve type hints
            logits = typing.cast(Tensor, logits)

            # Only get the logits of the last token
            logits = logits[:, -1, :]

            # Scale logits by temperature
            logits /= temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Apply softmax to the logits to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the probability distribution to get the prediction of
            # the next index
            idx_next = torch.multinomial(probs, num_samples=1)

            # Concatenate the context with the prediction
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
