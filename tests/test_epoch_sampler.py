import unittest

import torch

from src.training.trainer import EpochBatchSampler, format_duration


class TestEpochBatchSampler(unittest.TestCase):
    def test_epoch_uses_every_index_once(self):
        generator = torch.Generator().manual_seed(123)
        sampler = EpochBatchSampler(12, 3, generator)
        all_indices = []
        for _ in range(4):
            all_indices.extend(sampler.next_indices())
        self.assertEqual(len(all_indices), 12)
        self.assertEqual(sorted(all_indices), list(range(12)))

    def test_new_epoch_reshuffles(self):
        generator = torch.Generator().manual_seed(123)
        sampler = EpochBatchSampler(12, 3, generator)
        first_epoch = []
        for _ in range(4):
            first_epoch.extend(sampler.next_indices())
        second_epoch = []
        for _ in range(4):
            second_epoch.extend(sampler.next_indices())
        self.assertEqual(sorted(second_epoch), list(range(12)))
        self.assertNotEqual(first_epoch, second_epoch)
        self.assertEqual(sampler.epoch, 2)

    def test_resume_preserves_order_and_position(self):
        generator_a = torch.Generator().manual_seed(123)
        sampler_a = EpochBatchSampler(12, 3, generator_a)
        first_batch = sampler_a.next_indices()
        state = sampler_a.state_dict()
        expected_next = sampler_a.next_indices()

        generator_b = torch.Generator().manual_seed(999)
        sampler_b = EpochBatchSampler(12, 3, generator_b, state=state)
        actual_next = sampler_b.next_indices()

        self.assertEqual(len(first_batch), 3)
        self.assertEqual(actual_next, expected_next)

    def test_final_partial_batch_does_not_cross_epoch(self):
        generator = torch.Generator().manual_seed(123)
        sampler = EpochBatchSampler(10, 3, generator)
        batches = [sampler.next_indices() for _ in range(4)]
        self.assertEqual([len(batch) for batch in batches], [3, 3, 3, 1])
        self.assertEqual(sampler.epoch, 1)
        next_batch = sampler.next_indices()
        self.assertEqual(len(next_batch), 3)
        self.assertEqual(sampler.epoch, 2)

    def test_invalid_arguments(self):
        generator = torch.Generator().manual_seed(1)
        with self.assertRaises(ValueError):
            EpochBatchSampler(0, 3, generator)
        with self.assertRaises(ValueError):
            EpochBatchSampler(10, 0, generator)


class TestFormatDuration(unittest.TestCase):
    def test_duration_formats(self):
        self.assertEqual(format_duration(9), "9s")
        self.assertEqual(format_duration(75), "1m 15s")
        self.assertEqual(format_duration(3661), "1h 1m 1s")


if __name__ == "__main__":
    unittest.main()
