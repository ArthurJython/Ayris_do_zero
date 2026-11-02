
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.data.chat_dataset import ChatConversationDataset
from src.model.ayris import AYRISModel
from src.training.chat_step import train_chat_step, masked_cross_entropy


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/chat/tokenized"
CHECKPOINT_ROOT = ROOT / "checkpoints/training"

DEVICE = torch.device("cpu")

MODEL_CONFIG = {
    "vocab_size": 4096,
    "d_model": 256,
    "num_heads": 4,
    "num_layers": 11,
    "mlp_dim": 1024,
    "max_seq_len": 512,
    "padding_idx": 0,
    "dropout": 0.0,
}

SEED = 18
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
MAX_GRAD_NORM = 1.0
MIN_LR_RATIO = 0.1


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_data():
    manifest_path = DATA_DIR / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("vocab_size") != MODEL_CONFIG["vocab_size"]:
        raise ValueError("Vocabulário incompatível.")

    if manifest.get("context_length") != MODEL_CONFIG["max_seq_len"]:
        raise ValueError("Contexto incompatível.")

    tokenizer_path = ROOT / "checkpoints/tokenizer/tokenizer.json"

    if sha256_file(tokenizer_path) != manifest.get("tokenizer_sha256"):
        raise ValueError("Tokenizer diferente do utilizado na preparação.")

    files = {}

    for split in ("train", "validation", "test"):
        paths = [
            DATA_DIR / f"{split}.bin",
            DATA_DIR / f"{split}.mask.bin",
            DATA_DIR / f"{split}.index.jsonl",
        ]

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)

            files[str(path.relative_to(ROOT))] = sha256_file(path)

    files["manifest.json"] = sha256_file(manifest_path)

    return files


def load_dataset(split):
    return ChatConversationDataset(
        binary_path=DATA_DIR / f"{split}.bin",
        mask_path=DATA_DIR / f"{split}.mask.bin",
        index_path=DATA_DIR / f"{split}.index.jsonl",
        max_seq_len=MODEL_CONFIG["max_seq_len"],
        vocab_size=MODEL_CONFIG["vocab_size"],
    )


class ChatSampler:
    """Embaralha conversas sem reposição e permite retomada."""

    def __init__(self, dataset_size, seed, state=None):
        self.dataset_size = dataset_size
        self.generator = torch.Generator(device="cpu")

        if state is None:
            self.generator.manual_seed(seed)
            self.epoch = 1
            self.position = 0
            self.order = torch.randperm(
                dataset_size,
                generator=self.generator,
            )
        else:
            if state["dataset_size"] != dataset_size:
                raise ValueError("Tamanho do dataset mudou.")

            self.generator.set_state(state["generator_state"])
            self.epoch = int(state["epoch"])
            self.position = int(state["position"])
            self.order = state["order"].clone().long()

    def next_index(self):
        if self.position >= self.dataset_size:
            self.epoch += 1
            self.position = 0
            self.order = torch.randperm(
                self.dataset_size,
                generator=self.generator,
            )

        index = int(self.order[self.position])
        self.position += 1
        return index

    def state_dict(self):
        return {
            "dataset_size": self.dataset_size,
            "epoch": self.epoch,
            "position": self.position,
            "order": self.order.clone(),
            "generator_state": self.generator.get_state(),
        }


def build_optimizer(model, learning_rate):
    decay = []
    no_decay = []

    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)

    groups = []

    if decay:
        groups.append({
            "params": decay,
            "weight_decay": WEIGHT_DECAY,
        })

    if no_decay:
        groups.append({
            "params": no_decay,
            "weight_decay": 0.0,
        })

    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=BETAS,
        eps=ADAM_EPS,
    )


def build_scheduler(optimizer, total_steps, warmup_steps):
    if not 0 < warmup_steps < total_steps:
        raise ValueError("Warmup precisa ser menor que steps.")

    def schedule(step):
        if step < warmup_steps:
            return max(
                (step + 1) / warmup_steps,
                1.0 / warmup_steps,
            )

        progress = (step - warmup_steps) / max(
            1,
            total_steps - warmup_steps,
        )
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=schedule,
    )


def evaluate(model, dataset):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_response_tokens = 0

    try:
        with torch.inference_mode():
            for index in range(len(dataset)):
                inputs, targets, mask = dataset[index]

                inputs = inputs.unsqueeze(0).to(DEVICE)
                targets = targets.unsqueeze(0).to(DEVICE)
                mask = mask.unsqueeze(0).to(DEVICE)

                logits = model(inputs)

                loss, token_count = masked_cross_entropy(
                    logits,
                    targets,
                    mask,
                )

                total_loss += loss.item() * token_count
                total_response_tokens += token_count

    finally:
        model.train(was_training)

    if total_response_tokens == 0:
        raise RuntimeError("Validação sem tokens de resposta.")

    mean_loss = total_loss / total_response_tokens
    perplexity = math.exp(mean_loss) if mean_loss < 700 else float("inf")

    return mean_loss, perplexity, total_response_tokens


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    sampler,
    step,
    best_loss,
    config,
    fingerprint,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format": "ayris_chat_v1",
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "sampler_state_dict": sampler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "model_config": MODEL_CONFIG,
        "training_config": config,
        "data_fingerprint": fingerprint,
        "best_validation_loss": best_loss,
        "saved_at": utc_now(),
    }

    temporary = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary.open("wb") as file:
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary, path)

    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def restore_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    config,
    fingerprint,
    train_dataset_size,
):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )

    if checkpoint.get("format") != "ayris_chat_v1":
        raise ValueError("Formato de checkpoint incompatível.")

    if checkpoint["model_config"] != MODEL_CONFIG:
        raise ValueError("Arquitetura incompatível.")

    if checkpoint["training_config"] != config:
        raise ValueError("Configuração diferente da execução original.")

    if checkpoint["data_fingerprint"] != fingerprint:
        raise ValueError("Dataset mudou desde o checkpoint.")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    sampler = ChatSampler(
        train_dataset_size,
        seed=SEED,
        state=checkpoint["sampler_state_dict"],
    )

    torch.set_rng_state(checkpoint["torch_rng_state"])
    random.setstate(checkpoint["python_rng_state"])

    return (
        int(checkpoint["step"]),
        float(checkpoint["best_validation_loss"]),
        sampler,
    )


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.steps <= 0:
        raise ValueError("steps precisa ser positivo.")

    if not 0 < args.warmup_steps < args.steps:
        raise ValueError("Warmup precisa ser menor que steps.")

    if args.learning_rate <= 0:
        raise ValueError("Learning rate inválido.")

    if args.eval_interval <= 0 or args.save_interval <= 0:
        raise ValueError("Intervalos precisam ser positivos.")

    fingerprint = validate_data()

    config = {
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "threads": args.threads,
        "seed": args.seed,
        "batch_size": 1,
        "loss": "assistant_tokens_only",
    }

    run_dir = CHECKPOINT_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    latest_path = run_dir / "ayris_latest.pt"
    best_path = run_dir / "ayris_best.pt"

    if not args.resume and any(run_dir.iterdir()):
        raise FileExistsError(
            f"{run_dir} já contém arquivos. Use outro run-name."
        )

    train_dataset = load_dataset("train")
    validation_dataset = load_dataset("validation")

    model = AYRISModel(**MODEL_CONFIG).to(DEVICE)
    optimizer = build_optimizer(model, args.learning_rate)
    scheduler = build_scheduler(
        optimizer,
        args.steps,
        args.warmup_steps,
    )

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(latest_path)

        start_step, best_loss, sampler = restore_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            config,
            fingerprint,
            len(train_dataset),
        )

        if start_step >= args.steps:
            raise ValueError("Checkpoint já atingiu o total de passos.")

        print(f"Retomando no passo {start_step}.")

    else:
        source_path = args.init_from.expanduser().resolve()

        if not source_path.exists():
            raise FileNotFoundError(source_path)

        source = torch.load(
            source_path,
            map_location=DEVICE,
            weights_only=False,
        )

        if source.get("model_config") != MODEL_CONFIG:
            raise ValueError("Checkpoint de origem incompatível.")

        model.load_state_dict(source["model_state_dict"])

        print(f"Pesos carregados: {source_path}")
        print(f"Passo de origem: {source.get('step', 'desconhecido')}")

        sampler = ChatSampler(
            len(train_dataset),
            args.seed + 1,
        )

        start_step = 0

        best_loss, ppl, count = evaluate(
            model,
            validation_dataset,
        )

        print("\nAvaliação inicial:")
        print(f"Validation Loss: {best_loss:.6f}")
        print(f"Perplexity: {ppl:.3f}")
        print(f"Tokens avaliados: {count}")

        save_checkpoint(
            latest_path, model, optimizer, scheduler,
            sampler, 0, best_loss, config, fingerprint,
        )

        save_checkpoint(
            best_path, model, optimizer, scheduler,
            sampler, 0, best_loss, config, fingerprint,
        )

    metrics_path = run_dir / "metrics.jsonl"
    metrics_mode = "a" if args.resume else "w"

    log_file = metrics_path.open(metrics_mode, encoding="utf-8")

    def log(record):
        record["timestamp_utc"] = utc_now()
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()

    log({
        "event": "run_started" if not args.resume else "run_resumed",
        "step": start_step,
        "target_steps": args.steps,
    })

    session_start = time.monotonic()
    completed_step = start_step
    losses = []

    print("\n" + "=" * 60)
    print("FINE-TUNING CONVERSACIONAL — AYRIS")
    print("=" * 60)
    print(f"Conversas de treino: {len(train_dataset)}")
    print(f"Conversas de validação: {len(validation_dataset)}")
    print(f"Passos planejados: {args.steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Threads CPU: {args.threads}")

    try:
        for step in range(start_step + 1, args.steps + 1):
            index = sampler.next_index()

            inputs, targets, mask = train_dataset[index]

            metrics = train_chat_step(
                model,
                optimizer,
                inputs.unsqueeze(0).to(DEVICE),
                targets.unsqueeze(0).to(DEVICE),
                mask.unsqueeze(0).to(DEVICE),
                max_grad_norm=MAX_GRAD_NORM,
            )

            scheduler.step()
            completed_step = step
            losses.append(metrics["loss"])

            if step % 5 == 0 or step == args.steps:
                mean_loss = sum(losses) / len(losses)
                losses.clear()

                elapsed = time.monotonic() - session_start

                print(
                    f"Passo {step}/{args.steps} | "
                    f"Loss média: {mean_loss:.6f} | "
                    f"Gradiente: {metrics['grad_norm_before_clip']:.4f} | "
                    f"LR: {metrics['learning_rate']:.8f} | "
                    f"Época: {sampler.epoch}"
                )

                log({
                    "event": "training",
                    "step": step,
                    "epoch": sampler.epoch,
                    "loss_mean": mean_loss,
                    "gradient_before_clip": metrics["grad_norm_before_clip"],
                    "gradient_after_clip": metrics["grad_norm_after_clip"],
                    "learning_rate": metrics["learning_rate"],
                    "elapsed_seconds": elapsed,
                })

            if step % args.eval_interval == 0 or step == args.steps:
                val_loss, val_ppl, val_tokens = evaluate(
                    model,
                    validation_dataset,
                )

                improved = val_loss < best_loss - 1e-6

                if improved:
                    best_loss = val_loss

                    save_checkpoint(
                        best_path, model, optimizer, scheduler,
                        sampler, step, best_loss, config, fingerprint,
                    )

                print("\n--- VALIDAÇÃO ---")
                print(f"Passo: {step}")
                print(f"Validation Loss: {val_loss:.6f}")
                print(f"Perplexity: {val_ppl:.3f}")
                print(f"Tokens: {val_tokens}")
                print(f"Novo melhor: {'sim' if improved else 'não'}")
                print()

                log({
                    "event": "validation",
                    "step": step,
                    "validation_loss": val_loss,
                    "perplexity": val_ppl,
                    "response_tokens": val_tokens,
                    "best_validation_loss": best_loss,
                })

            if step % args.save_interval == 0 or step == args.steps:
                save_checkpoint(
                    latest_path, model, optimizer, scheduler,
                    sampler, step, best_loss, config, fingerprint,
                )

                print(f"Checkpoint recente salvo: passo {step}")

    except KeyboardInterrupt:
        print("\nInterrupção solicitada.")

    finally:
        if completed_step > start_step:
            save_checkpoint(
                latest_path, model, optimizer, scheduler,
                sampler, completed_step, best_loss, config, fingerprint,
            )
            print(f"Último estado salvo: passo {completed_step}")

        log({
            "event": "run_stopped",
            "step": completed_step,
            "best_validation_loss": best_loss,
        })

        log_file.close()

    print("\nFine-tuning encerrado.")
    print(f"Último checkpoint: {latest_path}")
    print(f"Melhor checkpoint: {best_path}")
    print(f"Melhor validation loss: {best_loss:.6f}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tuning conversacional da AYRIS."
    )

    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-name", type=str, default="chat_finetune_v1")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init-from", type=Path)
    group.add_argument("--resume", action="store_true")

    return parser


def main():
    args = build_parser().parse_args()

    if (
        not args.run_name
        or Path(args.run_name).name != args.run_name
        or args.run_name in {".", ".."}
    ):
        raise ValueError("run-name inválido.")

    train(args)


if __name__ == "__main__":
    main()
