
import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.model.ayris import AYRISModel


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT = (
    ROOT / "checkpoints/training/fase_03_5000/ayris_best.pt"
)

TOKENIZER_PATH = (
    ROOT / "checkpoints/tokenizer/tokenizer.json"
)

DEVICE = torch.device("cpu")


def load_model(checkpoint_path):

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(TOKENIZER_PATH)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = AYRISModel(**checkpoint["model_config"])

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    print("Checkpoint:", checkpoint_path.name)
    print("Passo:", checkpoint["step"])
    print("Parâmetros:", f"{model.count_parameters():,}")

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=80,
    temperature=0.7,
    top_k=40,
):

    if not prompt.strip():
        raise ValueError("O prompt não pode estar vazio.")

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens deve ser positivo.")

    if temperature < 0:
        raise ValueError("temperature não pode ser negativa.")

    if temperature > 0 and top_k <= 0:
        raise ValueError("top_k deve ser positivo.")

    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")

    if bos_id is None or eos_id is None:
        raise ValueError("BOS ou EOS não encontrado.")

    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    ).ids

    input_ids = [bos_id] + prompt_ids
    generated_ids = []

    # Impede geração de tokens estruturais.
    forbidden_tokens = [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<SYS>",
        "<USR>",
        "<AST>",
    ]

    forbidden_ids = [
        token_id
        for token in forbidden_tokens
        if (token_id := tokenizer.token_to_id(token)) is not None
    ]

    with torch.inference_mode():

        for _ in range(max_new_tokens):

            context = input_ids[-model.max_seq_len:]

            tensor = torch.tensor(
                [context],
                dtype=torch.long,
                device=DEVICE,
            )

            logits = model(tensor)

            next_logits = logits[0, -1, :].clone()

            if forbidden_ids:
                next_logits[forbidden_ids] = float("-inf")

            if temperature == 0:
                next_token = torch.argmax(next_logits)

            else:

                next_logits /= temperature

                k = min(top_k, next_logits.numel())

                values, _ = torch.topk(next_logits, k)

                threshold = values[-1]

                next_logits[next_logits < threshold] = float("-inf")

                probabilities = torch.softmax(
                    next_logits,
                    dim=-1,
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                ).squeeze(0)

            token_id = next_token.item()

            if token_id == eos_id:
                break

            generated_ids.append(token_id)
            input_ids.append(token_id)

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


def main():

    parser = argparse.ArgumentParser(
        description="Geração de texto AYRIS."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--prompt",
        default="A capital do Brasil é",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=18,
    )

    args = parser.parse_args()

    if args.threads <= 0:
        parser.error("--threads deve ser positivo.")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.checkpoint)

    print("\n" + "=" * 60)
    print("GERAÇÃO — AYRIS")
    print("=" * 60)

    print("\nPrompt:", args.prompt)

    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\nResposta:")
    print(args.prompt + response)


if __name__ == "__main__":
    main()
