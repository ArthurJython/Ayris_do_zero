
import unittest
import torch

from src.model.attention import CausalSelfAttention


class TestCausalSelfAttention(unittest.TestCase):

    def setUp(self):
        self.layer = CausalSelfAttention(
            d_model=256,
            num_heads=4,
            dropout=0.0,
        )
        self.layer.eval()

    def test_output_shape(self):
        x = torch.randn(2, 16, 256)

        output = self.layer(x)

        self.assertEqual(output.shape, (2, 16, 256))

    def test_output_dtype(self):
        x = torch.randn(2, 16, 256)

        output = self.layer(x)

        self.assertEqual(output.dtype, torch.float32)

    def test_causal_mask(self):
        """
        Alterações nos tokens futuros não podem modificar
        as saídas das posições anteriores.
        """

        x1 = torch.randn(1, 8, 256)
        x2 = x1.clone()

        # Modifica somente as posições futuras.
        x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :]) * 100

        output1 = self.layer(x1)
        output2 = self.layer(x2)

        # Posições 0, 1, 2 e 3 não podem ser afetadas.
        torch.testing.assert_close(
            output1[:, :4, :],
            output2[:, :4, :],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_gradient(self):
        x = torch.randn(
            2, 8, 256,
            requires_grad=True,
        )

        output = self.layer(x)

        loss = output.square().mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_parameter_count(self):
        total = sum(
            parameter.numel()
            for parameter in self.layer.parameters()
        )

        # Quatro projeções lineares de 256 -> 256,
        # todas com bias.
        expected = 4 * (256 * 256 + 256)

        self.assertEqual(total, expected)

    def test_invalid_model_dimension(self):
        with self.assertRaises(ValueError):
            CausalSelfAttention(
                d_model=255,
                num_heads=4,
            )

    def test_invalid_head_dimension(self):
        with self.assertRaises(ValueError):
            CausalSelfAttention(
                d_model=33,
                num_heads=3,
            )

    def test_invalid_input_shape(self):
        x = torch.randn(2, 256)

        with self.assertRaises(ValueError):
            self.layer(x)

    def test_invalid_input_dimension(self):
        x = torch.randn(2, 8, 128)

        with self.assertRaises(ValueError):
            self.layer(x)

    def test_invalid_dtype(self):
        x = torch.randn(2, 8, 256, dtype=torch.float64)

        with self.assertRaises(TypeError):
            self.layer(x)


if __name__ == "__main__":
    unittest.main()
