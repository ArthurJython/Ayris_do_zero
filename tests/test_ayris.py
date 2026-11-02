
import unittest

import torch
import torch.nn.functional as F

from src.model.ayris import AYRISModel


class TestAYRISModel(unittest.TestCase):

    def setUp(self):
        # Modelo menor para acelerar os testes funcionais.
        self.model = AYRISModel(
            vocab_size=4096,
            d_model=256,
            num_heads=4,
            num_layers=2,
            mlp_dim=1024,
            max_seq_len=512,
            dropout=0.0,
        )

        self.model.eval()

    def test_output_shape(self):
        ids = torch.randint(7, 4096, (2, 8))

        output = self.model(ids)

        self.assertEqual(output.shape, (2, 8, 4096))

    def test_output_dtype(self):
        ids = torch.randint(7, 4096, (2, 8))

        output = self.model(ids)

        self.assertEqual(output.dtype, torch.float32)

    def test_weight_tying(self):
        self.assertIs(
            self.model.lm_head.weight,
            self.model.token_embedding.embedding.weight,
        )

    def test_padding_embedding_is_zero(self):
        padding_vector = (
            self.model.token_embedding.embedding.weight[0]
        )

        self.assertTrue(
            torch.all(padding_vector == 0).item()
        )

    def test_padding_logit_is_suppressed(self):
        ids = torch.randint(7, 4096, (1, 4))

        output = self.model(ids)

        expected = torch.finfo(torch.float32).min

        self.assertTrue(
            torch.all(output[..., 0] == expected).item()
        )

    def test_causal_property(self):
        ids1 = torch.randint(7, 4096, (1, 8))
        ids2 = ids1.clone()

        ids2[:, 4:] = torch.randint(7, 4096, (1, 4))

        output1 = self.model(ids1)
        output2 = self.model(ids2)

        # Tokens futuros não alteram logits anteriores.
        torch.testing.assert_close(
            output1[:, :4, :],
            output2[:, :4, :],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_gradient(self):
        ids = torch.randint(7, 4096, (2, 8))
        targets = torch.randint(7, 4096, (2, 8))

        output = self.model(ids)

        loss = F.cross_entropy(
            output.reshape(-1, 4096),
            targets.reshape(-1),
        )

        loss.backward()

        embedding_grad = (
            self.model.token_embedding.embedding.weight.grad
        )

        self.assertIsNotNone(embedding_grad)

        self.assertTrue(
            torch.isfinite(embedding_grad).all().item()
        )

        # PAD não recebe gradiente.
        self.assertTrue(
            torch.all(embedding_grad[0] == 0).item()
        )

    def test_context_limit(self):
        ids = torch.randint(7, 4096, (1, 513))

        with self.assertRaises(ValueError):
            self.model(ids)

    def test_invalid_input_shape(self):
        ids = torch.randint(7, 4096, (8,))

        with self.assertRaises(ValueError):
            self.model(ids)

    def test_invalid_dtype(self):
        ids = torch.randn(2, 8)

        with self.assertRaises(TypeError):
            self.model(ids)

    def test_empty_sequence(self):
        ids = torch.empty((1, 0), dtype=torch.long)

        with self.assertRaises(ValueError):
            self.model(ids)

    def test_default_parameter_count(self):
        model = AYRISModel()

        expected = 9_736_448

        self.assertEqual(
            model.count_parameters(),
            expected,
        )

    def test_default_number_of_blocks(self):
        model = AYRISModel()

        self.assertEqual(len(model.blocks), 11)


if __name__ == "__main__":
    unittest.main()
