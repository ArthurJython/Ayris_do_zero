
import argparse
import math
import random

from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.dataset import TokenWindowDataset
from src.model.ayris import AYRISModel


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/tokenized"


def evaluate(
    checkpoint_path,
    split,
    max_batches=128,
    threads=4,
    seed=42,
):

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    torch.set_num_threads(threads)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = AYRISModel(**checkpoint["model_config"])

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    dataset = TokenWindowDataset(
        binary_path=DATA_DIR / f"{split}.bin",
        index_path=DATA_DIR / f"{split}.index.jsonl",
        block_size=model.max_seq_len,
        stride=512,
        vocab_size=model.vocab_size,
    )

    total_windows = len(dataset)

    if max_batches == 0 or max_batches >= total_windows:

        indexes = list(range(total_windows))

    else:

        rng = random.Random(seed)

        indexes = sorted(
            rng.sample(
                range(total_windows),
                max_batches,
            )
        )

    total_loss = 0.0
    total_tokens = 0

    with torch.inference_mode():

        for position, index in enumerate(indexes, start=1):

            input_ids, target_ids = dataset[index]

            input_ids = input_ids.unsqueeze(0)
            target_ids = target_ids.unsqueeze(0)

            logits = model(input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                target_ids.reshape(-1),
                reduction="sum",
            )

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Loss não finita na janela {index}."
                )

            total_loss += loss.item()
            total_tokens += target_ids.numel()

            if position % 32 == 0:
                print(
                    f"Avaliadas: {position}/{len(indexes)}",
                    flush=True,
                )

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)

    print("\n" + "=" * 60)
    print("AVALIAÇÃO AYRIS")
    print("=" * 60)

    print("Checkpoint:", checkpoint_path)
    print("Passo do checkpoint:", checkpoint["step"])
    print("Divisão:", split)
    print("Janelas disponíveis:", total_windows)
    print("Janelas avaliadas:", len(indexes))
    print("Tokens avaliados:", total_tokens)

    print(f"\nLoss: {average_loss:.6f}")
    print(f"Perplexity: {perplexity:.4f}")

    return average_loss, perplexity


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=128,
        help="Use 0 para avaliar todas as janelas.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.max_batches < 0:
        parser.error("--max-batches não pode ser negativo.")

    if args.threads <= 0:
        parser.error("--threads deve ser positivo.")

    evaluate(
        checkpoint_path=args.checkpoint,
        split=args.split,
        max_batches=args.max_batches,
        threads=args.threads,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
