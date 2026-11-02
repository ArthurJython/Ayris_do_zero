
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ChatConversationDataset(Dataset):
    """
    Dataset conversacional da AYRIS.

    Cada item representa uma conversa completa.

    Retorna:
        input_ids:  [sequence_length]
        target_ids: [sequence_length]
        loss_mask:  [sequence_length]

    A máscara indica quais tokens-alvo devem participar da loss.
    """

    def __init__(
        self,
        binary_path: str | Path,
        mask_path: str | Path,
        index_path: str | Path,
        max_seq_len: int = 512,
        vocab_size: int = 4096,
    ):
        self.binary_path = Path(binary_path)
        self.mask_path = Path(mask_path)
        self.index_path = Path(index_path)

        if max_seq_len <= 0:
            raise ValueError("max_seq_len deve ser positivo.")

        if vocab_size <= 0 or vocab_size > 65536:
            raise ValueError("vocab_size incompatível com uint16.")

        for path in (self.binary_path, self.mask_path, self.index_path):
            if not path.exists():
                raise FileNotFoundError(path)

        if self.binary_path.stat().st_size % 2 != 0:
            raise ValueError("Arquivo de tokens possui tamanho inválido.")

        token_count = self.binary_path.stat().st_size // 2

        if self.mask_path.stat().st_size != token_count:
            raise ValueError("Arquivo de máscara não corresponde aos tokens.")

        self.tokens = np.memmap(
            self.binary_path,
            dtype=np.dtype("<u2"),
            mode="r",
        )

        self.masks = np.memmap(
            self.mask_path,
            dtype=np.uint8,
            mode="r",
        )

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.fragments = []

        expected_offset = 0

        with self.index_path.open("r", encoding="utf-8") as file:
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
                    or offset != expected_offset
                    or length < 2
                ):
                    raise ValueError(
                        f"Offset ou comprimento inválido na linha {line_number}."
                    )

                if offset + length > token_count:
                    raise ValueError(
                        f"Fragmento ultrapassa o arquivo na linha {line_number}."
                    )

                if length - 1 > max_seq_len:
                    raise ValueError(
                        f"Conversa {line_number} excede o contexto de "
                        f"{max_seq_len} tokens."
                    )

                self.fragments.append((offset, length))
                expected_offset += length

        if expected_offset != token_count:
            raise ValueError("O índice não cobre todo o arquivo de tokens.")

        if not self.fragments:
            raise ValueError("Nenhuma conversa encontrada.")

        self._validate_fragments()

    def _validate_fragments(self):
        for number, (offset, length) in enumerate(self.fragments, start=1):
            ids = np.asarray(
                self.tokens[offset:offset + length],
                dtype=np.int64,
            )
            mask = np.asarray(
                self.masks[offset:offset + length],
                dtype=np.uint8,
            )

            if ids[0] != 2 or ids[-1] != 3:
                raise ValueError(
                    f"Conversa {number} não possui BOS/EOS válidos."
                )

            if ids.max() >= self.vocab_size:
                raise ValueError(
                    f"Conversa {number} possui token fora do vocabulário."
                )

            if np.any(mask > 1):
                raise ValueError(
                    f"Conversa {number} possui máscara inválida."
                )

            if not np.any(mask[1:]):
                raise ValueError(
                    f"Conversa {number} não possui tokens-alvo para treinamento."
                )

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.fragments):
            raise IndexError(index)

        offset, length = self.fragments[index]

        ids = np.array(
            self.tokens[offset:offset + length],
            dtype=np.int64,
            copy=True,
        )

        mask = np.array(
            self.masks[offset:offset + length],
            dtype=np.uint8,
            copy=True,
        )

        input_ids = torch.from_numpy(ids[:-1])
        target_ids = torch.from_numpy(ids[1:])
        loss_mask = torch.from_numpy(mask[1:].astype(np.bool_))

        return input_ids, target_ids, loss_mask
