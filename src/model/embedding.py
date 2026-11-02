
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Transforma IDs de tokens em vetores numéricos.

    Entrada:
        Tensor Long [batch_size, sequence_length]

    Saída:
        Tensor Float32 [batch_size, sequence_length, embedding_dim]
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        embedding_dim: int = 256,
        padding_idx: int = 0,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size deve ser positivo.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim deve ser positivo.")

        if not 0 <= padding_idx < vocab_size:
            raise ValueError("padding_idx fora dos limites do vocabulário.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        # Inicialização dos pesos em FP32.
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=embedding_dim ** -0.5,
        )

        # Garante que o vetor de padding permaneça zerado.
        with torch.no_grad():
            self.embedding.weight[padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            raise TypeError("Os IDs de entrada devem usar torch.long.")

        if input_ids.ndim != 2:
            raise ValueError(
                "A entrada deve possuir formato [batch_size, sequence_length]."
            )

        return self.embedding(input_ids)
