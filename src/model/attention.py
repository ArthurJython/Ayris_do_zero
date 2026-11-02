
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """
    Atenção própria causal com múltiplas cabeças e RoPE.

    Entrada:
        [batch_size, sequence_length, d_model]

    Saída:
        [batch_size, sequence_length, d_model]
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model deve ser positivo.")

        if num_heads <= 0:
            raise ValueError("num_heads deve ser positivo.")

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model deve ser divisível por num_heads."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout deve estar entre 0 e 1.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim deve ser par para utilizar RoPE."
            )

        # Projeções de Query, Key e Value.
        self.q_proj = nn.Linear(
            d_model, d_model, bias=bias
        )

        self.k_proj = nn.Linear(
            d_model, d_model, bias=bias
        )

        self.v_proj = nn.Linear(
            d_model, d_model, bias=bias
        )

        # Projeção após a união das cabeças.
        self.out_proj = nn.Linear(
            d_model, d_model, bias=bias
        )

        # Codificação posicional rotacional.
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            base=rope_base,
        )

        self.resid_dropout = nn.Dropout(dropout)

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
                "Batch e comprimento da sequência devem ser positivos."
            )

        if x.dtype != torch.float32:
            raise TypeError("A entrada deve usar torch.float32.")

        batch_size, seq_len, _ = x.shape

        # Projeções lineares.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, D] -> [B, H, T, head_dim]
        def split_heads(tensor):
            return tensor.view(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Aplicação da RoPE em Query e Key.
        q, k = self.rope(q, k)

        # Atenção causal.
        # Cada posição só pode acessar posições anteriores
        # e a própria posição.
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # [B, H, T, head_dim] -> [B, T, D]
        attention = attention.transpose(1, 2).contiguous()

        attention = attention.view(
            batch_size,
            seq_len,
            self.d_model,
        )

        # Projeção final.
        output = self.out_proj(attention)

        return self.resid_dropout(output)
