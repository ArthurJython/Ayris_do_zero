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
import torch.nn.functional as F

from src.data.dataset import TokenWindowDataset
from src.model.ayris import AYRISModel
from src.training.step import train_step


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/tokenized"
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

# Fase 4: valores padrão. Todos podem ser sobrescritos pela CLI.
DEFAULT_STEPS = 100_000
DEFAULT_BATCH_SIZE = 3
BLOCK_SIZE = 512
STRIDE = 512

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8

WARMUP_STEPS = 1_000
MIN_LR_RATIO = 0.1
MAX_GRAD_NORM = 1.0

EVAL_BATCHES = 128
EVAL_INTERVAL = 5_000
SAVE_INTERVAL = 1_000
LOG_INTERVAL = 250

OVERFIT_PATIENCE = 2
OVERFIT_RELATIVE_THRESHOLD = 0.01  # 1% acima do melhor val loss

SEED = 42
ANSI_RED = "\033[91m"
ANSI_RESET = "\033[0m"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_data():
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("vocab_size") != MODEL_CONFIG["vocab_size"]:
        raise ValueError("Vocabulário do dataset incompatível com o modelo.")
    if manifest.get("context_length") != BLOCK_SIZE:
        raise ValueError("Contexto do dataset incompatível.")

    tokenizer_path = ROOT / "checkpoints/tokenizer/tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)

    tokenizer_hash = sha256_file(tokenizer_path)
    if tokenizer_hash != manifest.get("tokenizer_sha256"):
        raise ValueError("Tokenizer mudou depois da tokenização do corpus.")

    files = [
        DATA_DIR / "train.bin",
        DATA_DIR / "train.index.jsonl",
        DATA_DIR / "validation.bin",
        DATA_DIR / "validation.index.jsonl",
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Arquivos do dataset ausentes: " + ", ".join(missing))

    return {
        "manifest": sha256_file(manifest_path),
        "tokenizer": tokenizer_hash,
        "files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in files
        },
    }


class EpochBatchSampler:
    """Embaralha cada época sem reposição e preserva posição para resume."""

    def __init__(self, dataset_size, batch_size, generator, state=None):
        if dataset_size <= 0:
            raise ValueError("dataset_size deve ser positivo.")
        if batch_size <= 0:
            raise ValueError("batch_size deve ser positivo.")

        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.generator = generator

        if state is None:
            self.epoch = 1
            self.position = 0
            self.order = torch.randperm(
                self.dataset_size,
                generator=self.generator,
            )
        else:
            if int(state["dataset_size"]) != self.dataset_size:
                raise ValueError("Tamanho do dataset mudou desde o checkpoint.")
            if int(state["batch_size"]) != self.batch_size:
                raise ValueError("Batch size mudou desde o checkpoint.")
            self.epoch = int(state["epoch"])
            self.position = int(state["position"])
            self.order = state["order"].clone().long()
            self.generator.set_state(state["generator_state"])
            if self.order.numel() != self.dataset_size:
                raise ValueError("Ordem da época salva está corrompida.")
            if not 0 <= self.position <= self.dataset_size:
                raise ValueError("Posição da época salva é inválida.")

    def next_indices(self):
        if self.position >= self.dataset_size:
            self.epoch += 1
            self.position = 0
            self.order = torch.randperm(
                self.dataset_size,
                generator=self.generator,
            )

        end = min(self.position + self.batch_size, self.dataset_size)
        indexes = self.order[self.position:end].tolist()
        self.position = end
        return indexes

    def state_dict(self):
        return {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "position": self.position,
            "order": self.order.clone(),
            "generator_state": self.generator.get_state(),
        }


def build_training_config(args):
    return {
        "total_steps": args.steps,
        "batch_size": args.batch_size,
        "block_size": BLOCK_SIZE,
        "stride": STRIDE,
        "learning_rate": args.learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "betas": list(BETAS),
        "adam_eps": ADAM_EPS,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": MIN_LR_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "eval_batches": args.eval_batches,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "overfit_patience": OVERFIT_PATIENCE,
        "overfit_relative_threshold": OVERFIT_RELATIVE_THRESHOLD,
        "seed": args.seed,
        "sampling": "epoch_shuffle_without_replacement",
    }


def build_optimizer(model, learning_rate):
    decay_parameters = []
    no_decay_parameters = []

    # named_parameters remove duplicatas de objetos compartilhados (weight tying).
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    groups = []
    if decay_parameters:
        groups.append({"params": decay_parameters, "weight_decay": WEIGHT_DECAY})
    if no_decay_parameters:
        groups.append({"params": no_decay_parameters, "weight_decay": 0.0})

    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=BETAS,
        eps=ADAM_EPS,
    )


def build_scheduler(optimizer, total_steps, warmup_steps):
    if not 0 < warmup_steps < total_steps:
        raise ValueError("warmup_steps deve ser maior que zero e menor que steps.")

    def schedule(step):
        if step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0 / warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def load_dataset(split):
    return TokenWindowDataset(
        binary_path=DATA_DIR / f"{split}.bin",
        index_path=DATA_DIR / f"{split}.index.jsonl",
        block_size=BLOCK_SIZE,
        stride=STRIDE,
        vocab_size=MODEL_CONFIG["vocab_size"],
    )


def get_training_batch(dataset, sampler):
    indexes = sampler.next_indices()
    samples = [dataset[index] for index in indexes]
    input_ids = torch.stack([sample[0] for sample in samples]).to(DEVICE)
    target_ids = torch.stack([sample[1] for sample in samples]).to(DEVICE)
    return input_ids, target_ids


def fixed_validation_indices(dataset, count, seed):
    sample_count = min(count, len(dataset))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    # Amostragem fixa sem reposição. Mesmas janelas entre checkpoints.
    return torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()


def evaluate(model, dataset, indexes):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    try:
        with torch.inference_mode():
            for index in indexes:
                input_ids, target_ids = dataset[index]
                input_ids = input_ids.unsqueeze(0).to(DEVICE)
                target_ids = target_ids.unsqueeze(0).to(DEVICE)
                logits = model(input_ids)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, model.vocab_size),
                    target_ids.reshape(-1),
                    reduction="sum",
                )
                if not torch.isfinite(loss_sum).item():
                    raise FloatingPointError(f"Loss de validação não finita na janela {index}.")
                total_loss += loss_sum.item()
                total_tokens += target_ids.numel()
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise RuntimeError("Avaliação sem tokens.")

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss) if mean_loss < 700 else float("inf")
    return mean_loss, perplexity, total_tokens


def save_stage_snapshot(
    path,
    model,
    step,
    best_val_loss,
    training_config,
    tokens_processed,
    elapsed_seconds,
):
    """Salva somente os pesos FP32 e metadados para inspecionar a evolução.

    Este arquivo NÃO substitui ayris_latest.pt: não contém estado do AdamW,
    scheduler, RNG ou sampler e, portanto, não serve para --resume.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "snapshot_format": "ayris_stage_weights_v1",
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "model_config": dict(MODEL_CONFIG),
        "training_config": dict(training_config),
        "best_val_loss": float(best_val_loss),
        "tokens_processed": int(tokens_processed),
        "elapsed_seconds": float(elapsed_seconds),
        "saved_at": utc_now(),
        "weights_only": True,
        "resume_supported": False,
    }

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("wb") as file:
            torch.save(snapshot, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def archive_stage_snapshot(
    run_dir,
    model,
    step,
    best_val_loss,
    training_config,
    tokens_processed,
    elapsed_seconds,
):
    """Cria um snapshot histórico sem deixar uma falha de arquivo parar o treino."""
    path = run_dir / "stages" / f"ayris_step_{step:06d}.pt"
    try:
        save_stage_snapshot(
            path,
            model,
            step,
            best_val_loss,
            training_config,
            tokens_processed,
            elapsed_seconds,
        )
        print(f"Estágio histórico salvo: {path}", flush=True)
    except Exception as error:
        print(
            f"[AVISO] Não foi possível arquivar o estágio {step:,}: {error!r}. "
            "O checkpoint completo ayris_latest.pt continua sendo o arquivo de retomada.",
            file=sys.stderr,
            flush=True,
        )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    sampler,
    step,
    best_val_loss,
    bad_validation_evals,
    training_config,
    data_fingerprint,
    tokens_processed,
    elapsed_seconds,
    started_at,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError(f"Parâmetro não finito: {name}")

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "sampler_state_dict": sampler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "best_val_loss": best_val_loss,
        "bad_validation_evals": bad_validation_evals,
        "model_config": MODEL_CONFIG,
        "training_config": training_config,
        "data_fingerprint": data_fingerprint,
        "tokens_processed": tokens_processed,
        "elapsed_seconds": elapsed_seconds,
        "started_at": started_at,
        "saved_at": utc_now(),
    }

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("wb") as file:
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def restore_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    sampler_generator,
    training_config,
    data_fingerprint,
):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

    if checkpoint.get("model_config") != MODEL_CONFIG:
        raise ValueError("Arquitetura incompatível com o checkpoint.")
    if checkpoint.get("training_config") != training_config:
        raise ValueError(
            "Configuração incompatível. Para resume, mantenha steps, batch, LR, "
            "warmup e intervalos iguais aos da execução original."
        )
    if checkpoint.get("data_fingerprint") != data_fingerprint:
        raise ValueError("Dataset ou tokenizer mudou desde o checkpoint.")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    sampler = EpochBatchSampler(
        dataset_size=checkpoint["sampler_state_dict"]["dataset_size"],
        batch_size=checkpoint["sampler_state_dict"]["batch_size"],
        generator=sampler_generator,
        state=checkpoint["sampler_state_dict"],
    )

    torch.set_rng_state(checkpoint["torch_rng_state"])
    random.setstate(checkpoint["python_rng_state"])

    return (
        int(checkpoint["step"]),
        float(checkpoint["best_val_loss"]),
        int(checkpoint.get("bad_validation_evals", 0)),
        int(checkpoint.get("tokens_processed", 0)),
        float(checkpoint.get("elapsed_seconds", 0.0)),
        checkpoint.get("started_at", utc_now()),
        sampler,
    )


def write_metric(file, record):
    record["timestamp_utc"] = utc_now()
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
    file.flush()


def print_red_alert(message):
    # ANSI 91 = vermelho claro em terminais compatíveis.
    print(f"{ANSI_RED}[ALERTA] POSSÍVEL OVERFITTING: {message}{ANSI_RESET}", flush=True)


def train(args):
    if args.steps <= 0:
        raise ValueError("steps deve ser positivo.")
    if args.batch_size <= 0:
        raise ValueError("batch-size deve ser positivo.")
    if args.threads <= 0:
        raise ValueError("threads deve ser positivo.")
    if args.warmup_steps <= 0 or args.warmup_steps >= args.steps:
        raise ValueError("warmup-steps deve ser maior que zero e menor que steps.")
    if args.eval_batches <= 0 or args.eval_interval <= 0 or args.save_interval <= 0 or args.log_interval <= 0:
        raise ValueError("Intervalos e quantidade de avaliação devem ser positivos.")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate deve ser positivo.")
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        raise ValueError("run-name inválido.")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(args.seed + 1)

    data_fingerprint = validate_data()
    training_config = build_training_config(args)

    run_dir = CHECKPOINT_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "ayris_latest.pt"
    best_path = run_dir / "ayris_best.pt"
    metrics_path = run_dir / "metrics.jsonl"

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(latest_path)
    elif any(run_dir.iterdir()):
        raise FileExistsError(
            f"{run_dir} já contém arquivos. Use --resume ou outro --run-name."
        )

    print("=" * 68)
    print("TREINAMENTO DA AYRIS — FASE 4")
    print("=" * 68)
    print(f"Passos planejados: {args.steps:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Contexto: {BLOCK_SIZE}")
    print(f"Tokens por atualização: {args.batch_size * BLOCK_SIZE:,}")
    print(f"Learning Rate máximo: {args.learning_rate:.8f}")
    print(f"Warmup: {args.warmup_steps:,}")
    print(f"Validação: {args.eval_batches} janelas a cada {args.eval_interval:,} passos")
    print(f"Checkpoint: a cada {args.save_interval:,} passos")
    print(f"Dispositivo: {DEVICE}")
    print(f"Threads CPU: {args.threads}")

    train_dataset = load_dataset("train")
    validation_dataset = load_dataset("validation")
    validation_indexes = fixed_validation_indices(
        validation_dataset,
        args.eval_batches,
        args.seed + 2,
    )

    print(f"Janelas de treinamento: {len(train_dataset):,}")
    print(f"Janelas de validação: {len(validation_dataset):,}")
    print(f"Janelas fixas de avaliação: {len(validation_indexes):,}")
    print(f"Atualizações por época: {math.ceil(len(train_dataset) / args.batch_size):,}")

    model = AYRISModel(**MODEL_CONFIG).to(DEVICE)
    optimizer = build_optimizer(model, args.learning_rate)
    scheduler = build_scheduler(optimizer, args.steps, args.warmup_steps)

    print(f"Parâmetros: {model.count_parameters():,}")

    start_step = 0
    best_val_loss = float("inf")
    bad_validation_evals = 0
    tokens_processed = 0
    elapsed_before_resume = 0.0
    started_at = utc_now()

    if args.resume:
        (
            start_step,
            best_val_loss,
            bad_validation_evals,
            tokens_processed,
            elapsed_before_resume,
            started_at,
            sampler,
        ) = restore_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            sampler_generator,
            training_config,
            data_fingerprint,
        )
        print(f"\nCheckpoint recuperado no passo {start_step:,}.")
        print(f"Tempo acumulado anterior: {format_duration(elapsed_before_resume)}")
        if start_step >= args.steps:
            raise ValueError("O checkpoint já atingiu o total de passos solicitado.")

        # Arquiva o estado recuperado para que o ponto de retomada também fique visível.
        archive_stage_snapshot(
            run_dir, model, start_step, best_val_loss,
            training_config, tokens_processed, elapsed_before_resume,
        )
    else:
        if args.init_from is not None:
            source_path = Path(args.init_from).expanduser().resolve()
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            source = torch.load(source_path, map_location=DEVICE, weights_only=False)
            if source.get("model_config") != MODEL_CONFIG:
                raise ValueError("Checkpoint de origem possui arquitetura incompatível.")
            model.load_state_dict(source["model_state_dict"])
            print(f"\nPesos iniciais carregados de: {source_path}")
            print(f"Passo de origem: {source.get('step', 'desconhecido')}")
            print("AdamW e scheduler serão inicializados para a fase 4.")

        sampler = EpochBatchSampler(
            len(train_dataset),
            args.batch_size,
            sampler_generator,
        )

        initial_loss, initial_ppl, initial_tokens = evaluate(
            model,
            validation_dataset,
            validation_indexes,
        )
        best_val_loss = initial_loss
        print("\nAvaliação inicial:")
        print(f"Validation Loss: {initial_loss:.6f}")
        print(f"Perplexity: {initial_ppl:.3f}")
        print(f"Tokens de validação: {initial_tokens:,}")

        save_checkpoint(
            best_path, model, optimizer, scheduler, sampler,
            0, best_val_loss, bad_validation_evals,
            training_config, data_fingerprint,
            tokens_processed, elapsed_before_resume, started_at,
        )
        save_checkpoint(
            latest_path, model, optimizer, scheduler, sampler,
            0, best_val_loss, bad_validation_evals,
            training_config, data_fingerprint,
            tokens_processed, elapsed_before_resume, started_at,
        )
        # Estágio zero: pesos recebidos da fase anterior (ou inicialização aleatória).
        archive_stage_snapshot(
            run_dir, model, 0, best_val_loss,
            training_config, tokens_processed, elapsed_before_resume,
        )

    session_start = time.monotonic()
    last_saved_step = start_step
    completed_step = start_step
    safe_to_save = True
    log_losses = []
    eval_losses = []
    last_log_time = time.monotonic()

    metrics_mode = "a" if args.resume else "w"
    try:
        with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
            write_metric(metrics_file, {
                "event": "run_started" if not args.resume else "run_resumed",
                "step": start_step,
                "target_steps": args.steps,
                "batch_size": args.batch_size,
                "tokens_processed": tokens_processed,
                "elapsed_seconds_before_session": elapsed_before_resume,
                "started_at": started_at,
            })

            for step in range(start_step + 1, args.steps + 1):
                safe_to_save = False
                input_ids, target_ids = get_training_batch(train_dataset, sampler)
                batch_tokens = target_ids.numel()

                metrics = train_step(
                    model,
                    optimizer,
                    input_ids,
                    target_ids,
                    max_grad_norm=MAX_GRAD_NORM,
                )
                scheduler.step()

                tokens_processed += batch_tokens
                completed_step = step
                safe_to_save = True
                log_losses.append(metrics["loss"])
                eval_losses.append(metrics["loss"])

                if step % args.log_interval == 0 or step == args.steps:
                    now = time.monotonic()
                    session_elapsed = now - session_start
                    total_elapsed = elapsed_before_resume + session_elapsed
                    interval_elapsed = max(now - last_log_time, 1e-9)
                    steps_in_session = step - start_step
                    steps_per_second = steps_in_session / max(session_elapsed, 1e-9)
                    tokens_per_second = (tokens_processed) / max(total_elapsed, 1e-9)
                    remaining_steps = args.steps - step
                    eta_seconds = remaining_steps / max(steps_per_second, 1e-9)
                    mean_train_loss = sum(log_losses) / len(log_losses)
                    log_losses.clear()

                    print(
                        f"Passo {step:,}/{args.steps:,} | "
                        f"Train Loss: {mean_train_loss:.5f} | "
                        f"Gradiente: {metrics['grad_norm_before_clip']:.5f} -> "
                        f"{metrics['grad_norm_after_clip']:.5f} | "
                        f"LR usado: {metrics['learning_rate']:.8f}"
                    )
                    print(
                        f"Tempo acumulado: {format_duration(total_elapsed)} | "
                        f"Velocidade média: {steps_per_second:.3f} passos/s | "
                        f"{tokens_per_second:.1f} tokens/s | "
                        f"ETA aproximado: {format_duration(eta_seconds)}"
                    )
                    print(f"Época atual: {sampler.epoch} | Posição: {sampler.position:,}/{len(train_dataset):,}")

                    write_metric(metrics_file, {
                        "event": "training",
                        "step": step,
                        "epoch": sampler.epoch,
                        "epoch_position": sampler.position,
                        "train_loss_mean": mean_train_loss,
                        "gradient_before_clip": metrics["grad_norm_before_clip"],
                        "gradient_after_clip": metrics["grad_norm_after_clip"],
                        "learning_rate_used": metrics["learning_rate"],
                        "tokens_processed": tokens_processed,
                        "elapsed_seconds": total_elapsed,
                        "elapsed_human": format_duration(total_elapsed),
                        "steps_per_second": steps_per_second,
                        "tokens_per_second": tokens_per_second,
                        "eta_seconds": eta_seconds,
                    })
                    last_log_time = now

                # Alerta e seleção de checkpoint são avaliados em conjunto.
                should_evaluate = (
                    step % args.eval_interval == 0
                    or step == args.steps
                )
                if should_evaluate:
                    train_loss_interval = (
                        sum(eval_losses) / len(eval_losses)
                        if eval_losses else float("nan")
                    )
                    eval_losses.clear()

                    val_loss, val_ppl, val_token_count = evaluate(
                        model,
                        validation_dataset,
                        validation_indexes,
                    )

                    generalization_gap = val_loss - train_loss_interval
                    threshold = best_val_loss * (1.0 + OVERFIT_RELATIVE_THRESHOLD)

                    improved = val_loss < best_val_loss - 1e-6
                    if improved:
                        best_val_loss = val_loss
                        bad_validation_evals = 0
                    elif val_loss > threshold:
                        bad_validation_evals += 1
                    else:
                        bad_validation_evals = max(0, bad_validation_evals - 1)

                    now = time.monotonic()
                    total_elapsed = elapsed_before_resume + (now - session_start)

                    print("\n" + "=" * 60)
                    print(f"VALIDAÇÃO | Passo {step:,}/{args.steps:,}")
                    print(f"Train Loss média desde a última avaliação: {train_loss_interval:.6f}")
                    print(f"Validation Loss: {val_loss:.6f}")
                    print(f"Perplexity: {val_ppl:.3f}")
                    print(f"Diferença Val - Train: {generalization_gap:.6f}")
                    print(f"Janelas avaliadas: {len(validation_indexes):,} | Tokens: {val_token_count:,}")
                    print(f"Tempo acumulado: {format_duration(total_elapsed)}")

                    warning = bad_validation_evals >= OVERFIT_PATIENCE
                    if warning:
                        print_red_alert(
                            f"Validation Loss ficou mais de {OVERFIT_RELATIVE_THRESHOLD:.0%} acima "
                            f"do melhor resultado por {bad_validation_evals} avaliações consecutivas. "
                            "Verifique a tendência antes de continuar; isto é um indício, não um diagnóstico."
                        )
                    elif bad_validation_evals > 0:
                        print(
                            f"Atenção: validação acima do melhor em {bad_validation_evals}/"
                            f"{OVERFIT_PATIENCE} avaliações necessárias para o alerta."
                        )

                    write_metric(metrics_file, {
                        "event": "validation",
                        "step": step,
                        "train_loss_interval": train_loss_interval,
                        "validation_loss": val_loss,
                        "perplexity": val_ppl,
                        "generalization_gap": generalization_gap,
                        "best_validation_loss": best_val_loss,
                        "bad_validation_evals": bad_validation_evals,
                        "possible_overfitting": warning,
                        "validation_windows": len(validation_indexes),
                        "validation_tokens": val_token_count,
                        "tokens_processed": tokens_processed,
                        "elapsed_seconds": total_elapsed,
                        "elapsed_human": format_duration(total_elapsed),
                    })

                    if improved:
                        save_checkpoint(
                            best_path, model, optimizer, scheduler, sampler,
                            step, best_val_loss, bad_validation_evals,
                            training_config, data_fingerprint,
                            tokens_processed, total_elapsed, started_at,
                        )
                        print("Novo melhor checkpoint salvo.")

                    print("=" * 60 + "\n")

                if step % args.save_interval == 0 or step == args.steps:
                    now = time.monotonic()
                    total_elapsed = elapsed_before_resume + (now - session_start)
                    save_checkpoint(
                        latest_path, model, optimizer, scheduler, sampler,
                        step, best_val_loss, bad_validation_evals,
                        training_config, data_fingerprint,
                        tokens_processed, total_elapsed, started_at,
                    )
                    # Snapshot somente dos pesos em cada checkpoint periódico.
                    archive_stage_snapshot(
                        run_dir, model, step, best_val_loss,
                        training_config, tokens_processed, total_elapsed,
                    )
                    last_saved_step = step
                    print(f"Checkpoint recente salvo: {step:,}")

            # Finalização: registra duração real e taxa média.
            session_elapsed = time.monotonic() - session_start
            total_elapsed = elapsed_before_resume + session_elapsed
            avg_steps_per_second = (completed_step - start_step) / max(session_elapsed, 1e-9)
            avg_tokens_per_second = tokens_processed / max(total_elapsed, 1e-9)

            write_metric(metrics_file, {
                "event": "run_finished",
                "step": completed_step,
                "tokens_processed": tokens_processed,
                "elapsed_seconds": total_elapsed,
                "elapsed_human": format_duration(total_elapsed),
                "session_elapsed_seconds": session_elapsed,
                "average_steps_per_second": avg_steps_per_second,
                "average_tokens_per_second": avg_tokens_per_second,
                "best_validation_loss": best_val_loss,
                "finished_at": utc_now(),
            })

    except KeyboardInterrupt:
        print("\nInterrupção solicitada pelo usuário.")
        if safe_to_save and completed_step > last_saved_step:
            total_elapsed = elapsed_before_resume + (time.monotonic() - session_start)
            save_checkpoint(
                latest_path, model, optimizer, scheduler, sampler,
                completed_step, best_val_loss, bad_validation_evals,
                training_config, data_fingerprint,
                tokens_processed, total_elapsed, started_at,
            )
            archive_stage_snapshot(
                run_dir, model, completed_step, best_val_loss,
                training_config, tokens_processed, total_elapsed,
            )
            last_saved_step = completed_step
            print(f"Checkpoint salvo no passo {completed_step:,}.")
        else:
            print(f"Mantido último checkpoint seguro: passo {last_saved_step:,}.")
        print("Retome com --resume e a mesma configuração.")
        return

    except Exception:
        print("\nERRO durante o treinamento.", file=sys.stderr)
        if safe_to_save and completed_step > last_saved_step:
            try:
                total_elapsed = elapsed_before_resume + (time.monotonic() - session_start)
                save_checkpoint(
                    latest_path, model, optimizer, scheduler, sampler,
                    completed_step, best_val_loss, bad_validation_evals,
                    training_config, data_fingerprint,
                    tokens_processed, total_elapsed, started_at,
                )
                archive_stage_snapshot(
                    run_dir, model, completed_step, best_val_loss,
                    training_config, tokens_processed, total_elapsed,
                )
                print(f"Último estado seguro salvo no passo {completed_step:,}.")
            except Exception as save_error:
                print(f"Falha ao salvar estado seguro: {save_error!r}", file=sys.stderr)
        raise

    print("=" * 68)
    print("FASE DE TREINAMENTO CONCLUÍDA")
    print("=" * 68)
    print(f"Passos concluídos: {completed_step:,}")
    print(f"Tokens processados: {tokens_processed:,}")
    print(f"Melhor Validation Loss: {best_val_loss:.6f}")
    print(f"Tempo total: {format_duration(total_elapsed)} ({total_elapsed:.1f} segundos)")
    print(f"Velocidade média: {avg_steps_per_second:.3f} passos/s")
    print(f"Velocidade média: {avg_tokens_per_second:.1f} tokens/s")
    print(f"Checkpoint recente: {latest_path}")
    print(f"Melhor checkpoint: {best_path}")
    print(f"Histórico: {metrics_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Treinamento da AYRIS por épocas, com monitoramento e checkpoints."
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--run-name", type=str, default="fase_04_100k")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--seed", type=int, default=SEED)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init-from", type=Path, help="Inicializa pesos de um checkpoint; reinicia AdamW e scheduler.")
    group.add_argument("--resume", action="store_true", help="Retoma checkpoint da própria execução.")
    group.add_argument("--from-scratch", action="store_true", help="Inicializa pesos aleatórios.")
    return parser


def main():
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
