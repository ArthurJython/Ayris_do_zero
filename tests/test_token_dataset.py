
import json
import tempfile
import unittest

from pathlib import Path

import numpy as np
import torch

from src.data.dataset import (
    TokenWindowDataset,
    create_dataloader,
)


class TestTokenWindowDataset(unittest.TestCase):

    def setUp(self):

        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

        # Dois fragmentos separados.
        first = np.arange(10, 20, dtype="<u2")
        second = np.arange(30, 40, dtype="<u2")

        tokens = np.concatenate([first, second])

        self.binary_path = self.directory / "train.bin"
        tokens.tofile(self.binary_path)

        self.index_path = self.directory / "train.index.jsonl"

        records = [
            {
                "offset": 0,
                "length": 10,
                "filename": "document_a.pdf",
            },
            {
                "offset": 10,
                "length": 10,
                "filename": "document_b.pdf",
            },
        ]

        with self.index_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            for record in records:
                file.write(json.dumps(record) + "\n")

        self.dataset = TokenWindowDataset(
            binary_path=self.binary_path,
            index_path=self.index_path,
            block_size=4,
            stride=4,
            vocab_size=4096,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_window_count(self):
        # Cada fragmento gera três janelas.
        self.assertEqual(len(self.dataset), 6)

    def test_input_target_shift(self):
        x, y = self.dataset[0]

        self.assertEqual(x.tolist(), [10, 11, 12, 13])
        self.assertEqual(y.tolist(), [11, 12, 13, 14])

    def test_fragments_never_mix(self):

        # Índices 0, 1 e 2 pertencem ao primeiro fragmento.
        # Índices 3, 4 e 5 pertencem ao segundo.
        x, y = self.dataset[3]

        self.assertTrue(torch.all(x >= 30).item())
        self.assertTrue(torch.all(y >= 30).item())

        self.assertTrue(torch.all(x < 40).item())
        self.assertTrue(torch.all(y < 40).item())

    def test_last_token_is_preserved(self):

        # Última janela do primeiro fragmento.
        _, y = self.dataset[2]

        self.assertEqual(y[-1].item(), 19)

    def test_tensor_types(self):
        x, y = self.dataset[0]

        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)

    def test_dataloader(self):

        loader = create_dataloader(
            data_dir=self.directory,
            split="train",
            batch_size=2,
            block_size=4,
            stride=4,
            shuffle=False,
        )

        x, y = next(iter(loader))

        self.assertEqual(x.shape, (2, 4))
        self.assertEqual(y.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
