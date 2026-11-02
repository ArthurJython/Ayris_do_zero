
import json
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


class TokenWindowDataset(Dataset):
    """
    Dataset autoregressivo para a AYRIS.

    Cada item retorna:

        input_ids:  [block_size]
        target_ids: [block_size]

    Os alvos representam os próximos tokens.

    As janelas nunca atravessam os limites dos fragmentos.
    """

    def __init__(
        self,
        binary_path: str | Path,
        index_path: str | Path,
        block_size: int = 512,
        stride: int = 512,
        vocab_size: int = 4096,
    ):
        self.binary_path = Path(binary_path)
        self.index_path = Path(index_path)

        if block_size <= 0:
            raise ValueError("block_size deve ser positivo.")

        if stride <= 0 or stride > block_size:
            raise ValueError(
                "stride deve ser positivo e menor ou igual a block_size."
            )

        if vocab_size <= 0 or vocab_size > 65536:
            raise ValueError("vocab_size incompatível com uint16.")

        if not self.binary_path.exists():
            raise FileNotFoundError(self.binary_path)

        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)

        if self.binary_path.stat().st_size % 2 != 0:
            raise ValueError("Tamanho do arquivo binário inválido.")

        self.block_size = block_size
        self.stride = stride
        self.vocab_size = vocab_size

        self.total_tokens = self.binary_path.stat().st_size // 2

        self.tokens = np.memmap(
            self.binary_path,
            dtype=np.dtype("<u2"),
            mode="r",
        )

        self.windows = []

        self._load_index()

    def _load_index(self):
        expected_offset = 0

        with self.index_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON inválido na linha {line_number}."
                    ) from exc

                offset = record.get("offset")
                length = record.get("length")

                if (
                    isinstance(offset, bool)
                    or not isinstance(offset, int)
                    or isinstance(length, bool)
                    or not isinstance(length, int)
                    or offset < 0
                    or length <= 0
                ):
                    raise ValueError(
                        f"Offset ou comprimento inválido na linha {line_number}."
                    )

                if offset != expected_offset:
                    raise ValueError(
                        f"Descontinuidade no índice: linha {line_number}."
                    )

                if offset + length > self.total_tokens:
                    raise ValueError(
                        f"Sequência ultrapassa o arquivo na linha {line_number}."
                    )

                if length < self.block_size + 1:
                    raise ValueError(
                        f"Fragmento curto demais na linha {line_number}."
                    )

                # Último início possível de uma janela completa.
                last_start = length - self.block_size - 1

                starts = list(
                    range(
                        0,
                        last_start + 1,
                        self.stride,
                    )
                )

                # Inclui a parte final do fragmento.
                # Pode haver sobreposição com a janela anterior,
                # mas nenhum token é descartado como alvo.
                if starts[-1] != last_start:
                    starts.append(last_start)

                for local_start in starts:
                    self.windows.append(
                        (offset + local_start,)
                    )

                expected_offset += length

        if expected_offset != self.total_tokens:
            raise ValueError(
                "O índice não cobre todo o arquivo binário."
            )

        if not self.windows:
            raise ValueError("Nenhuma janela de treinamento foi criada.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):

        if index < 0 or index >= len(self.windows):
            raise IndexError(index)

        start = self.windows[index][0]
        end = start + self.block_size + 1

        # Copia os dados para evitar problemas com o memmap
        # somente de leitura.
        window = np.array(
            self.tokens[start:end],
            dtype=np.int64,
            copy=True,
        )

        if window.size != self.block_size + 1:
            raise RuntimeError("Janela incompleta.")

        if window.max() >= self.vocab_size:
            raise ValueError("ID de token fora do vocabulário.")

        # Entrada: tokens 0 até 511.
        # Alvo: tokens 1 até 512.
        input_ids = torch.from_numpy(window[:-1])
        target_ids = torch.from_numpy(window[1:])

        return input_ids, target_ids


def create_dataloader(
    data_dir: str | Path,
    split: str,
    batch_size: int = 1,
    block_size: int = 512,
    stride: int = 512,
    shuffle: bool | None = None,
    num_workers: int = 0,
):
    if split not in {"train", "validation", "test"}:
        raise ValueError("Divisão inválida.")

    if batch_size <= 0:
        raise ValueError("batch_size deve ser positivo.")

    if num_workers < 0:
        raise ValueError("num_workers não pode ser negativo.")

    data_dir = Path(data_dir)

    dataset = TokenWindowDataset(
        binary_path=data_dir / f"{split}.bin",
        index_path=data_dir / f"{split}.index.jsonl",
        block_size=block_size,
        stride=stride,
    )

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
