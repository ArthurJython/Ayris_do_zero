#!/usr/bin/env bash
set -Eeuo pipefail

# Launcher do treinamento normal de linguagem/conhecimento da AYRIS.
# Execute de qualquer diretório; a raiz do projeto é descoberta pelo local deste script.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
VENV_PYTHON="$ROOT/.venv/bin/python"
TRAINER_MODULE="src.training.trainer"
RUN_NAME="${AYRIS_KNOWLEDGE_RUN_NAME:-fase_05_conhecimento}"
RUN_DIR="$ROOT/checkpoints/training/$RUN_NAME"

# Configuração correspondente ao treinamento literário da fase 4.
STEPS="${AYRIS_KNOWLEDGE_STEPS:-100000}"
BATCH_SIZE="${AYRIS_KNOWLEDGE_BATCH_SIZE:-3}"
THREADS="${AYRIS_KNOWLEDGE_THREADS:-4}"
LEARNING_RATE="${AYRIS_KNOWLEDGE_LR:-0.0001}"
WARMUP_STEPS="${AYRIS_KNOWLEDGE_WARMUP:-1000}"
EVAL_BATCHES="${AYRIS_KNOWLEDGE_EVAL_BATCHES:-128}"
EVAL_INTERVAL="${AYRIS_KNOWLEDGE_EVAL_INTERVAL:-5000}"
SAVE_INTERVAL="${AYRIS_KNOWLEDGE_SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${AYRIS_KNOWLEDGE_LOG_INTERVAL:-250}"
SEED="${AYRIS_KNOWLEDGE_SEED:-18}"

# Por padrão, mantém o ramo de conhecimento baseado no melhor checkpoint literário.
INIT_CHECKPOINT="${AYRIS_KNOWLEDGE_INIT:-$ROOT/checkpoints/training/fase_04_100k/ayris_best.pt}"

usage() {
    cat <<USAGE
Uso: ./treinar_ayris_conhecimento.sh [novo|retomar|status|ajuda]

Comandos:
  novo      Inicia um novo estágio de treinamento normal de linguagem.
  retomar   Retoma o estágio $RUN_NAME a partir de ayris_latest.pt.
  status    Mostra os arquivos e as últimas métricas desse estágio.
  ajuda     Mostra esta ajuda.

Configuração padrão:
  Passos:             $STEPS
  Batch size:         $BATCH_SIZE
  Threads CPU:        $THREADS
  Learning rate máx.: $LEARNING_RATE
  Warmup:             $WARMUP_STEPS
  Avaliação:          $EVAL_BATCHES janelas a cada $EVAL_INTERVAL passos
  Checkpoint latest:  a cada $SAVE_INTERVAL passos
  Log:                a cada $LOG_INTERVAL passos
  Inicialização:      $INIT_CHECKPOINT
  Execução:           $RUN_NAME

É possível ajustar valores por variáveis de ambiente AYRIS_KNOWLEDGE_*.
Exemplo:
  AYRIS_KNOWLEDGE_THREADS=2 ./treinar_ayris_conhecimento.sh novo
USAGE
}

fail() {
    echo "ERRO: $*" >&2
    exit 1
}

check_environment() {
    [[ -x "$VENV_PYTHON" ]] || fail "Python da .venv não encontrado: $VENV_PYTHON"
    [[ -f "$ROOT/src/training/trainer.py" ]] || fail "Não encontrei src/training/trainer.py em $ROOT"
    cd "$ROOT"
}

run_training() {
    local mode="$1"
    check_environment

    local -a common_args=(
        --steps "$STEPS"
        --batch-size "$BATCH_SIZE"
        --run-name "$RUN_NAME"
        --threads "$THREADS"
        --learning-rate "$LEARNING_RATE"
        --warmup-steps "$WARMUP_STEPS"
        --eval-batches "$EVAL_BATCHES"
        --eval-interval "$EVAL_INTERVAL"
        --save-interval "$SAVE_INTERVAL"
        --log-interval "$LOG_INTERVAL"
        --seed "$SEED"
    )

    echo "============================================================"
    echo "AYRIS — TREINAMENTO NORMAL DE LINGUAGEM / CONHECIMENTO"
    echo "============================================================"
    echo "Projeto:       $ROOT"
    echo "Execução:      $RUN_NAME"
    echo "Passos:        $STEPS"
    echo "Batch:         $BATCH_SIZE"
    echo "Threads:       $THREADS"
    echo "Learning rate: $LEARNING_RATE"
    echo "Modo:          $mode"
    echo "============================================================"

    if [[ "$mode" == "novo" ]]; then
        [[ -f "$INIT_CHECKPOINT" ]] || fail "Checkpoint inicial não encontrado: $INIT_CHECKPOINT"
        if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
            fail "$RUN_DIR já contém arquivos. Use 'retomar' para continuar ou escolha outro nome com AYRIS_KNOWLEDGE_RUN_NAME."
        fi
        mkdir -p "$RUN_DIR"
        exec "$VENV_PYTHON" -m "$TRAINER_MODULE" "${common_args[@]}" --init-from "$INIT_CHECKPOINT"
    else
        [[ -f "$RUN_DIR/ayris_latest.pt" ]] || fail "Checkpoint de retomada não encontrado: $RUN_DIR/ayris_latest.pt"
        exec "$VENV_PYTHON" -m "$TRAINER_MODULE" "${common_args[@]}" --resume
    fi
}

show_status() {
    echo "Estágio: $RUN_NAME"
    echo "Diretório: $RUN_DIR"
    if [[ ! -d "$RUN_DIR" ]]; then
        echo "Ainda não existe."
        return 0
    fi

    echo
    echo "Arquivos:"
    find "$RUN_DIR" -maxdepth 2 -type f -printf '%f (%s bytes)\n' | sort

    if [[ -f "$RUN_DIR/metrics.jsonl" ]]; then
        echo
        echo "Últimos registros de métricas:"
        tail -n 8 "$RUN_DIR/metrics.jsonl"
    fi
}

case "${1:-ajuda}" in
    novo)
        run_training novo
        ;;
    retomar)
        run_training retomar
        ;;
    status)
        show_status
        ;;
    ajuda|-h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
