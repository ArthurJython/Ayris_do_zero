
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).

    Aplica rotações às representações Q (Query) e K (Key).

    Formato de entrada:
        [batch, heads, sequence_length, head_dim]

    Formato de saída:
        Mesmo formato da entrada.

    Implementação com pares intercalados:
        (x0, x1), (x2, x3), ...
    """

    def __init__(
        self,
        head_dim: int = 64,
        base: float = 10000.0,
    ):
        super().__init__()

        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim deve ser positivo e par.")

        if base <= 0:
            raise ValueError("base deve ser positiva.")

        self.head_dim = head_dim
        self.base = base

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    @staticmethod
    def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]

        return torch.stack(
            (-odd, even),
            dim=-1,
        ).flatten(-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if q.ndim != 4 or k.ndim != 4:
            raise ValueError(
                "Q e K devem ter formato [batch, heads, sequence, head_dim]."
            )

        if q.shape[-1] != self.head_dim:
            raise ValueError("Dimensão de Q incompatível com head_dim.")

        if k.shape[-1] != self.head_dim:
            raise ValueError("Dimensão de K incompatível com head_dim.")

        if q.shape[0] != k.shape[0]:
            raise ValueError("Q e K devem possuir o mesmo batch.")

        if q.shape[-2] != k.shape[-2]:
            raise ValueError("Q e K devem possuir o mesmo comprimento.")

        if q.device != k.device or q.dtype != k.dtype:
            raise ValueError("Q e K devem possuir o mesmo dispositivo e dtype.")

        if not isinstance(position_offset, int) or position_offset < 0:
            raise ValueError("position_offset deve ser um inteiro não negativo.")

        seq_len = q.shape[-2]

        positions = torch.arange(
            position_offset,
            position_offset + seq_len,
            device=q.device,
            dtype=torch.float32,
        )

        inv_freq = self.inv_freq.to(device=q.device)

        frequencies = torch.outer(positions, inv_freq)

        cos = frequencies.cos().repeat_interleave(2, dim=-1)
        sin = frequencies.sin().repeat_interleave(2, dim=-1)

        cos = cos[None, None, :, :].to(dtype=q.dtype)
        sin = sin[None, None, :, :].to(dtype=q.dtype)

        q_rotated = q * cos + self.rotate_pairs(q) * sin
        k_rotated = k * cos + self.rotate_pairs(k) * sin

        return q_rotated, k_rotated
