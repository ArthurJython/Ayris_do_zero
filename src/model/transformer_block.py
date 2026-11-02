
import torch
import torch.nn as nn

from src.model.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    """
    Transformer Decoder Block com arquitetura Pre-LayerNorm.

    Estrutura:
        LayerNorm -> Causal Attention -> Residual
        LayerNorm -> MLP -> Residual

    Entrada e saída:
        [batch_size, sequence_length, d_model]
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        mlp_dim: int = 1024,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model deve ser positivo.")

        if mlp_dim <= 0:
            raise ValueError("mlp_dim deve ser positivo.")

        if norm_eps <= 0:
            raise ValueError("norm_eps deve ser positivo.")

        self.d_model = d_model

        # Primeiro bloco: normalização e atenção causal.
        self.ln1 = nn.LayerNorm(
            d_model,
            eps=norm_eps,
        )

        self.attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Segundo bloco: normalização e rede feed-forward.
        self.ln2 = nn.LayerNorm(
            d_model,
            eps=norm_eps,
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "Entrada deve ter formato [batch, sequence, d_model]."
            )

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Esperado d_model={self.d_model}, "
                f"recebido {x.shape[-1]}."
            )

        if x.shape[0] <= 0 or x.shape[1] <= 0:
            raise ValueError(
                "Batch e sequência devem ser positivos."
            )

        if x.dtype != torch.float32:
            raise TypeError("A entrada deve usar torch.float32.")

        # Atenção com conexão residual.
        x = x + self.attention(self.ln1(x))

        # MLP com conexão residual.
        x = x + self.mlp(self.ln2(x))

        return x
