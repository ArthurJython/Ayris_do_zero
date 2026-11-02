#!/usr/bin/env bash
set -Eeuo pipefail

# AYRIS — lançador do fine-tuning conversacional.
# Salve este arquivo na raiz do projeto AYRIS e execute com:
#   ./treinar_ayris_chat.sh novo
#   ./treinar_ayris_chat.sh retomar

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
TRAINER_MODULE="src.training.chat_trainer"
SOURCE_CHECKPOINT="$ROOT_DIR/checkpoints/training/fase_04_100k/ayris_best.pt"
RUN_NAME="chat_finetune_v2"
RUN_DIR="$ROOT_DIR/checkpoints/training/$RUN_NAME"
LATEST_CHECKPOINT="$RUN_DIR/ayris_latest.pt"

# Configuração do experimento-piloto atual.
STEPS=18
LEARNING_RATE="0.00001"
WARMUP_STEPS=2
EVAL_INTERVAL=20
SAVE_INTERVAL=10
THREADS=2

show_help() {
    cat <<HELP
AYRIS — fine-tuning conversacional

Uso:
  ./treinar_ayris_chat.sh novo       Inicia um experimento novo.
  ./treinar_ayris_chat.sh retomar    Retoma o experimento salvo.
  ./treinar_ayris_chat.sh status     Mostra os arquivos e métricas do experimento.
  ./treinar_ayris_chat.sh ajuda      Mostra esta ajuda.

Configuração atual:
  Run name:       $RUN_NAME
  Steps:          $STEPS
  Learning rate:  $LEARNING_RATE
  Warmup:         $WARMUP_STEPS
  Avaliação:      a cada $EVAL_INTERVAL passos
  Salvamento:     a cada $SAVE_INTERVAL passos
  Threads CPU:    $THREADS

Para iniciar outro experimento, edite RUN_NAME no início deste arquivo.
Não altere os parâmetros ao retomar uma execução: o treinador exige a
mesma configuração usada no checkpoint.
HELP
}

fail() {
    printf '\nERRO: %s\n' "$1" >&2
    exit 1
}

check_project() {
    [[ -x "$PYTHON" ]] || fail "Python da .venv não encontrado: $PYTHON"
    [[ -f "$ROOT_DIR/src/training/chat_trainer.py" ]] || fail "Não encontrei src/training/chat_trainer.py."
    [[ -f "$ROOT_DIR/src/data/chat_dataset.py" ]] || fail "Não encontrei src/data/chat_dataset.py."
    [[ -f "$ROOT_DIR/src/training/chat_step.py" ]] || fail "Não encontrei src/training/chat_step.py."

    local required_files=(
        "$ROOT_DIR/data/chat/tokenized/manifest.json"
        "$ROOT_DIR/data/chat/tokenized/train.bin"
        "$ROOT_DIR/data/chat/tokenized/train.mask.bin"
        "$ROOT_DIR/data/chat/tokenized/train.index.jsonl"
        "$ROOT_DIR/data/chat/tokenized/validation.bin"
        "$ROOT_DIR/data/chat/tokenized/validation.mask.bin"
        "$ROOT_DIR/data/chat/tokenized/validation.index.jsonl"
        "$ROOT_DIR/data/chat/tokenized/test.bin"
        "$ROOT_DIR/data/chat/tokenized/test.mask.bin"
        "$ROOT_DIR/data/chat/tokenized/test.index.jsonl"
        "$ROOT_DIR/checkpoints/tokenizer/tokenizer.json"
    )

    local file
    for file in "${required_files[@]}"; do
        [[ -f "$file" ]] || fail "Arquivo necessário não encontrado: ${file#$ROOT_DIR/}"
    done
}

run_trainer() {
    local mode="$1"
    cd "$ROOT_DIR"
    export PYTHONUNBUFFERED=1

    local common_args=(
        --steps "$STEPS"
        --learning-rate "$LEARNING_RATE"
        --warmup-steps "$WARMUP_STEPS"
        --eval-interval "$EVAL_INTERVAL"
        --save-interval "$SAVE_INTERVAL"
        --threads "$THREADS"
        --run-name "$RUN_NAME"
    )

    printf '\nAYRIS — fine-tuning conversacional\n'
    printf 'Projeto: %s\n' "$ROOT_DIR"
    printf 'Experimento: %s\n' "$RUN_NAME"
    printf 'Modo: %s\n\n' "$mode"

    if [[ "$mode" == "novo" ]]; then
        [[ -f "$SOURCE_CHECKPOINT" ]] || fail "Checkpoint inicial não encontrado: ${SOURCE_CHECKPOINT#$ROOT_DIR/}"
        if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
            fail "A pasta do experimento já contém arquivos: ${RUN_DIR#$ROOT_DIR/}. Para outro experimento, altere RUN_NAME neste script. Para continuar este, use 'retomar'."
        fi
        "$PYTHON" -m "$TRAINER_MODULE" "${common_args[@]}" --init-from "$SOURCE_CHECKPOINT"
    else
        [[ -f "$LATEST_CHECKPOINT" ]] || fail "Checkpoint para retomada não encontrado: ${LATEST_CHECKPOINT#$ROOT_DIR/}. Inicie primeiro com './treinar_ayris_chat.sh novo'."
        "$PYTHON" -m "$TRAINER_MODULE" "${common_args[@]}" --resume
    fi
}

status() {
    printf 'Experimento: %s\n' "$RUN_NAME"
    printf 'Diretório: %s\n' "$RUN_DIR"
    if [[ ! -d "$RUN_DIR" ]]; then
        printf 'Ainda não iniciado.\n'
        return
    fi
    printf '\nArquivos:\n'
    find "$RUN_DIR" -maxdepth 1 -type f -printf '%f (%s bytes)\n' | sort
    if [[ -f "$RUN_DIR/metrics.jsonl" ]]; then
        printf '\nÚltimos registros de métricas:\n'
        tail -n 8 "$RUN_DIR/metrics.jsonl"
    fi
}

case "${1:-ajuda}" in
    novo)
        check_project
        run_trainer novo
        ;;
    retomar|resume)
        check_project
        run_trainer retomar
        ;;
    status)
        status
        ;;
    ajuda|help|-h|--help)
        show_help
        ;;
    *)
        show_help
        fail "Opção desconhecida: $1"
        ;;
esac
