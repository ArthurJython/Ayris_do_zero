
import unittest
import torch

from src.model.embedding import TokenEmbedding


class TestTokenEmbedding(unittest.TestCase):

    def setUp(self):
        self.layer = TokenEmbedding(
            vocab_size=4096,
            embedding_dim=256,
            padding_idx=0,
        )

    def test_output_shape(self):
        ids = torch.tensor([
            [2, 10, 15, 6],
            [5, 20, 30, 0],
        ])

        output = self.layer(ids)

        self.assertEqual(output.shape, (2, 4, 256))

    def test_output_dtype(self):
        ids = torch.tensor([[2, 10, 15]])

        output = self.layer(ids)

        self.assertEqual(output.dtype, torch.float32)

    def test_padding_is_zero(self):
        ids = torch.tensor([[2, 10, 0]])

        output = self.layer(ids)

        self.assertTrue(
            torch.all(output[0, 2] == 0).item()
        )

    def test_padding_gradient_is_zero(self):
        ids = torch.tensor([[2, 10, 0]])

        output = self.layer(ids)
        output.sum().backward()

        padding_gradient = self.layer.embedding.weight.grad[0]

        self.assertTrue(
            torch.all(padding_gradient == 0).item()
        )

    def test_parameter_count(self):
        total = sum(
            parameter.numel()
            for parameter in self.layer.parameters()
        )

        self.assertEqual(total, 4096 * 256)

    def test_invalid_dtype(self):
        ids = torch.tensor([[2, 10]], dtype=torch.float32)

        with self.assertRaises(TypeError):
            self.layer(ids)


if __name__ == "__main__":
    unittest.main()
