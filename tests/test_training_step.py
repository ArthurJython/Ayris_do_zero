
import unittest

import torch

from src.model.ayris import AYRISModel
from src.training.step import train_step


class TestTrainingStep(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

        self.model = AYRISModel(
            vocab_size=32,
            d_model=16,
            num_heads=2,
            num_layers=1,
            mlp_dim=32,
            max_seq_len=8,
            dropout=0.0,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=0.1,
        )

        self.input_ids = torch.tensor([
            [2, 7, 8, 9],
            [2, 10, 11, 12],
        ])

        self.target_ids = torch.tensor([
            [7, 8, 9, 13],
            [10, 11, 12, 14],
        ])

    def test_finite_loss(self):
        result = train_step(
            self.model,
            self.optimizer,
            self.input_ids,
            self.target_ids,
        )

        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))

    def test_finite_gradient_norm(self):
        result = train_step(
            self.model,
            self.optimizer,
            self.input_ids,
            self.target_ids,
        )

        self.assertTrue(
            torch.isfinite(torch.tensor(result["grad_norm"]))
        )

    def test_parameters_updated(self):
        before = (
            self.model.token_embedding.embedding.weight
            .detach()
            .clone()
        )

        train_step(
            self.model,
            self.optimizer,
            self.input_ids,
            self.target_ids,
        )

        after = self.model.token_embedding.embedding.weight.detach()

        self.assertFalse(torch.equal(before, after))

    def test_learning_rate(self):
        result = train_step(
            self.model,
            self.optimizer,
            self.input_ids,
            self.target_ids,
        )

        self.assertEqual(result["learning_rate"], 3e-4)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            train_step(
                self.model,
                self.optimizer,
                self.input_ids,
                self.target_ids[:, :-1],
            )

    def test_padding_target_rejected(self):
        targets = self.target_ids.clone()
        targets[0, 0] = 0

        with self.assertRaises(ValueError):
            train_step(
                self.model,
                self.optimizer,
                self.input_ids,
                targets,
            )


if __name__ == "__main__":
    unittest.main()
