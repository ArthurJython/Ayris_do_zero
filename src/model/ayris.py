
import math

import torch
import torch.nn as nn

from src.model.embedding import TokenEmbedding
from src.model.transformer_block import TransformerBlock


class AYRISModel(nn.Module):
    """
    Arquitetura principal da AYRIS.

    Decoder-only Transformer autoregressivo.

    Entrada:
        input_ids: [batch_size, sequence_length]

    Saída:
        logits: [batch_size, sequence_length, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 11,
        mlp_dim: int = 1024,
        max_seq_len: int = 512,
        padding_idx: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size deve ser positivo.")

        if d_model <= 0:
            raise ValueError("d_model deve ser positivo.")

        if num_layers <= 0:
            raise ValueError("num_layers deve ser positivo.")

        if max_seq_len <= 0:
            raise ValueError("max_seq_len deve ser positivo.")

        if not 0 <= padding_idx < vocab_size:
            raise ValueError("padding_idx inválido.")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx

        # 1. Representação dos tokens.
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )

        # 2. Empilhamento dos blocos Transformer.
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 3. Normalização final.
        self.final_norm = nn.LayerNorm(d_model)

        # 4. Projeção para o vocabulário.
        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        # 5. Inicialização dos pesos.
        self.apply(self._initialize_module)

        # Escalonamento das projeções residuais.
        residual_std = 0.02 / math.sqrt(2 * num_layers)

        for block in self.blocks:
            nn.init.normal_(
                block.attention.out_proj.weight,
                mean=0.0,
                std=residual_std,
            )

            nn.init.normal_(
                block.mlp[2].weight,
                mean=0.0,
                std=residual_std,
            )

        # 6. Compartilhamento dos pesos do embedding
        # com a projeção de saída.
        self.lm_head.weight = self.token_embedding.embedding.weight

        # O vetor de padding permanece zerado.
        with torch.no_grad():
            self.token_embedding.embedding.weight[
                padding_idx
            ].zero_()

    @staticmethod
    def _initialize_module(module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        if input_ids.dtype != torch.long:
            raise TypeError("input_ids deve usar torch.long.")

        if input_ids.ndim != 2:
            raise ValueError(
                "Entrada deve ter formato [batch, sequence]."
            )

        batch_size, seq_len = input_ids.shape

        if batch_size <= 0 or seq_len <= 0:
            raise ValueError(
                "Batch e sequência devem ser positivos."
            )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequência excede o contexto máximo "
                f"de {self.max_seq_len} tokens."
            )

        # IDs -> vetores.
        x = self.token_embedding(input_ids)

        # Processamento pelos blocos Transformer.
        for block in self.blocks:
            x = block(x)

        # Normalização final.
        x = self.final_norm(x)

        # Vetores -> logits do vocabulário.
        logits = self.lm_head(x)

        # Impede que o modelo preveja PAD como próximo token.
        pad_mask = (
            torch.arange(
                self.vocab_size,
                device=logits.device,
            ) == self.padding_idx
        )

        logits = logits.masked_fill(
            pad_mask.view(1, 1, -1),
            torch.finfo(logits.dtype).min,
        )

        return logits

    def count_parameters(self) -> int:
        """
        Conta parâmetros únicos, evitando contar duas
        vezes os pesos compartilhados.
        """

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )
