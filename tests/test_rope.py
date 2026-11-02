
import unittest
import torch

from src.model.rope import RotaryEmbedding


class TestRotaryEmbedding(unittest.TestCase):

    def setUp(self):
        self.rope = RotaryEmbedding(head_dim=64)

    def test_output_shape(self):
        q = torch.randn(2, 4, 16, 64)
        k = torch.randn(2, 4, 16, 64)

        q_out, k_out = self.rope(q, k)

        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)

    def test_output_dtype(self):
        q = torch.randn(1, 4, 8, 64)
        k = torch.randn(1, 4, 8, 64)

        q_out, k_out = self.rope(q, k)

        self.assertEqual(q_out.dtype, torch.float32)
        self.assertEqual(k_out.dtype, torch.float32)

    def test_position_zero_is_unchanged(self):
        q = torch.randn(1, 4, 8, 64)
        k = torch.randn(1, 4, 8, 64)

        q_out, k_out = self.rope(q, k)

        torch.testing.assert_close(q_out[:, :, 0, :], q[:, :, 0, :])
        torch.testing.assert_close(k_out[:, :, 0, :], k[:, :, 0, :])

    def test_rotation_preserves_norm(self):
        q = torch.randn(2, 4, 16, 64)
        k = torch.randn(2, 4, 16, 64)

        q_out, k_out = self.rope(q, k)

        torch.testing.assert_close(
            torch.linalg.vector_norm(q_out, dim=-1),
            torch.linalg.vector_norm(q, dim=-1),
        )

        torch.testing.assert_close(
            torch.linalg.vector_norm(k_out, dim=-1),
            torch.linalg.vector_norm(k, dim=-1),
        )

    def test_position_offset(self):
        q = torch.randn(1, 4, 8, 64)
        k = torch.randn(1, 4, 8, 64)

        q_out_0, _ = self.rope(q, k, position_offset=0)
        q_out_1, _ = self.rope(q, k, position_offset=1)

        self.assertFalse(
            torch.allclose(q_out_0, q_out_1)
        )

    def test_gradient(self):
        q = torch.randn(1, 4, 8, 64, requires_grad=True)
        k = torch.randn(1, 4, 8, 64, requires_grad=True)

        q_out, k_out = self.rope(q, k)

        loss = q_out.sum() + k_out.sum()
        loss.backward()

        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)

        self.assertTrue(torch.isfinite(q.grad).all().item())
        self.assertTrue(torch.isfinite(k.grad).all().item())

    def test_invalid_dimension(self):
        with self.assertRaises(ValueError):
            RotaryEmbedding(head_dim=63)

    def test_invalid_input_shape(self):
        q = torch.randn(4, 64)
        k = torch.randn(4, 64)

        with self.assertRaises(ValueError):
            self.rope(q, k)


if __name__ == "__main__":
    unittest.main()
