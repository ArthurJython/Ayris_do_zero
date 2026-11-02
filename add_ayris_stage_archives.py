#!/usr/bin/env python3
"""Patch the AYRIS trainer to archive model-only snapshots at each save interval.

Run from the AYRIS repository root, after safely stopping the current trainer:
    python3 add_ayris_stage_archives.py

The live Python process does not reload source files. Stop with Ctrl+C, apply this
patch, then resume the same run with --resume.
"""
from pathlib import Path
import sys

TARGET = Path("src/training/trainer.py")

HELPERS = r'''def save_stage_snapshot(
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


'''

REPLACEMENTS = [
    (
        'def save_checkpoint(\n',
        HELPERS + 'def save_checkpoint(\n',
        'inserir funções de snapshot',
    ),
    (
        '''        print(f"Tempo acumulado anterior: {format_duration(elapsed_before_resume)}")
        if start_step >= args.steps:
            raise ValueError("O checkpoint já atingiu o total de passos solicitado.")
''',
        '''        print(f"Tempo acumulado anterior: {format_duration(elapsed_before_resume)}")
        if start_step >= args.steps:
            raise ValueError("O checkpoint já atingiu o total de passos solicitado.")

        # Arquiva o estado recuperado para que o ponto de retomada também fique visível.
        archive_stage_snapshot(
            run_dir, model, start_step, best_val_loss,
            training_config, tokens_processed, elapsed_before_resume,
        )
''',
        'arquivar estado ao retomar',
    ),
    (
        '''        save_checkpoint(
            latest_path, model, optimizer, scheduler, sampler,
            0, best_val_loss, bad_validation_evals,
            training_config, data_fingerprint,
            tokens_processed, elapsed_before_resume, started_at,
        )
''',
        '''        save_checkpoint(
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
''',
        'arquivar estágio inicial',
    ),
    (
        '''                    last_saved_step = step
                    print(f"Checkpoint recente salvo: {step:,}")
''',
        '''                    # Snapshot somente dos pesos em cada checkpoint periódico.
                    archive_stage_snapshot(
                        run_dir, model, step, best_val_loss,
                        training_config, tokens_processed, total_elapsed,
                    )
                    last_saved_step = step
                    print(f"Checkpoint recente salvo: {step:,}")
''',
        'arquivar cada checkpoint periódico',
    ),
    (
        '''            last_saved_step = completed_step
            print(f"Checkpoint salvo no passo {completed_step:,}.")
''',
        '''            archive_stage_snapshot(
                run_dir, model, completed_step, best_val_loss,
                training_config, tokens_processed, total_elapsed,
            )
            last_saved_step = completed_step
            print(f"Checkpoint salvo no passo {completed_step:,}.")
''',
        'arquivar estado salvo por Ctrl+C',
    ),
    (
        '''                print(f"Último estado seguro salvo no passo {completed_step:,}.")
''',
        '''                archive_stage_snapshot(
                    run_dir, model, completed_step, best_val_loss,
                    training_config, tokens_processed, total_elapsed,
                )
                print(f"Último estado seguro salvo no passo {completed_step:,}.")
''',
        'arquivar estado salvo após erro',
    ),
]


def main():
    if not TARGET.is_file():
        print(
            f"Não encontrei {TARGET}. Execute este script na raiz do repositório AYRIS.",
            file=sys.stderr,
        )
        return 1

    source = TARGET.read_text(encoding="utf-8")
    if 'def save_stage_snapshot(' in source or 'snapshot_format": "ayris_stage_weights_v1' in source:
        print("O treinador já parece conter o sistema de snapshots. Nenhuma alteração feita.")
        return 0

    updated = source
    for old, new, description in REPLACEMENTS:
        count = updated.count(old)
        if count != 1:
            print(
                f"Não apliquei alterações: trecho para '{description}' encontrado {count} vezes; "
                "esperava exatamente 1. O arquivo original foi mantido.",
                file=sys.stderr,
            )
            return 2
        updated = updated.replace(old, new, 1)

    backup = TARGET.with_suffix(TARGET.suffix + ".before_stage_archives.bak")
    if backup.exists():
        print(
            f"Backup já existe: {backup}. Renomeie-o ou confira manualmente antes de continuar.",
            file=sys.stderr,
        )
        return 3

    backup.write_text(source, encoding="utf-8")
    TARGET.write_text(updated, encoding="utf-8")
    print(f"Treinador atualizado: {TARGET}")
    print(f"Backup original: {backup}")
    print("Snapshots históricos serão gravados em checkpoints/training/<run-name>/stages/.")
    print("Cada snapshot contém pesos + metadados; para --resume continue usando ayris_latest.pt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
