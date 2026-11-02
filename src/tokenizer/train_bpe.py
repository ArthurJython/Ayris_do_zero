
import argparse
import json
import unicodedata

from pathlib import Path
from importlib.metadata import version

import pyarrow.parquet as pq

from tokenizers import Tokenizer, decoders, models
from tokenizers import normalizers, pre_tokenizers, trainers


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DEFAULT_VOCAB_SIZE = 4096
DEFAULT_MIN_FREQUENCY = 2
BATCH_SIZE = 128

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<SYS>",
    "<USR>",
    "<AST>",
]

EXPECTED_SPECIAL_IDS = {
    token: index
    for index, token in enumerate(SPECIAL_TOKENS)
}


# ============================================================
# LEITURA DO PARQUET
# ============================================================

def text_iterator(parquet_path):
    """
    Produz os textos gradualmente.
    Não carrega o dataset inteiro na memória.
    """

    parquet = pq.ParquetFile(parquet_path)

    for batch in parquet.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["text"]
    ):

        texts = batch.column(0).to_pylist()

        for text in texts:

            if text is None:
                continue

            if text:
                yield text


# ============================================================
# CRIAÇÃO DO TOKENIZER
# ============================================================

def create_tokenizer():

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<UNK>"
        )
    )

    # Preservar acentos e normalizar Unicode.
    tokenizer.normalizer = normalizers.NFC()

    # Transformação Byte-Level.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True
    )

    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


# ============================================================
# TREINAMENTO BPE
# ============================================================

def train_tokenizer(
    input_path,
    output_dir,
    vocab_size,
    min_frequency,
    overwrite
):

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if vocab_size < len(SPECIAL_TOKENS) + 256:
        raise ValueError(
            "O vocabulário precisa comportar os "
            "256 símbolos iniciais e os tokens especiais."
        )

    if min_frequency < 1:
        raise ValueError(
            "min_frequency precisa ser maior ou igual a 1."
        )

    tokenizer_path = output_dir / "tokenizer.json"
    config_path = output_dir / "tokenizer_config.json"

    if not overwrite and (
        tokenizer_path.exists() or config_path.exists()
    ):
        raise FileExistsError(
            f"Já existem arquivos em {output_dir}. "
            "Use --overwrite para substituir."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Abrir dataset
    # --------------------------------------------------------

    parquet = pq.ParquetFile(input_path)

    total_rows = parquet.metadata.num_rows

    print("\n" + "=" * 65)
    print("TREINAMENTO DO TOKENIZER — AYRIS")
    print("=" * 65)

    print(f"\nDataset: {input_path}")
    print(f"Registros: {total_rows:,}")
    print(f"Vocabulário desejado: {vocab_size:,}")
    print(f"Frequência mínima: {min_frequency}")

    # --------------------------------------------------------
    # Alfabeto Byte-Level
    # --------------------------------------------------------

    alphabet = sorted(
        pre_tokenizers.ByteLevel.alphabet()
    )

    if len(alphabet) != 256:
        raise RuntimeError(
            f"Alfabeto Byte-Level inesperado: {len(alphabet)}"
        )

    print(f"\nAlfabeto inicial: {len(alphabet)} símbolos")
    print(f"Tokens especiais: {len(SPECIAL_TOKENS)}")

    # --------------------------------------------------------
    # Criar tokenizer e trainer
    # --------------------------------------------------------

    tokenizer = create_tokenizer()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=alphabet,
    )

    # --------------------------------------------------------
    # Treinamento
    # --------------------------------------------------------

    print("\nIniciando treinamento BPE...\n")

    tokenizer.train_from_iterator(
        text_iterator(input_path),
        trainer=trainer,
        length=total_rows
    )

    # --------------------------------------------------------
    # Verificar tokens especiais
    # --------------------------------------------------------

    special_ids = {
        token: tokenizer.token_to_id(token)
        for token in SPECIAL_TOKENS
    }

    if special_ids != EXPECTED_SPECIAL_IDS:
        raise RuntimeError(
            "Os IDs dos tokens especiais não correspondem "
            f"ao esperado.\nEsperado: {EXPECTED_SPECIAL_IDS}"
            f"\nObtido: {special_ids}"
        )

    # --------------------------------------------------------
    # Verificar alfabeto
    # --------------------------------------------------------

    missing_symbols = [
        symbol
        for symbol in alphabet
        if tokenizer.token_to_id(symbol) is None
    ]

    if missing_symbols:
        raise RuntimeError(
            f"Símbolos Byte-Level ausentes: "
            f"{len(missing_symbols)}"
        )

    # --------------------------------------------------------
    # Verificar tamanho do vocabulário
    # --------------------------------------------------------

    actual_vocab_size = tokenizer.get_vocab_size(
        with_added_tokens=True
    )

    print("\n[ VOCABULÁRIO ]")

    print(
        f"Tamanho solicitado: {vocab_size:,}"
    )

    print(
        f"Tamanho obtido: {actual_vocab_size:,}"
    )

    print(
        f"Símbolos Byte-Level: {len(alphabet)}"
    )

    print(
        f"Tokens especiais: {len(special_ids)}"
    )

    if actual_vocab_size > vocab_size:
        raise RuntimeError(
            "O vocabulário ultrapassou o tamanho solicitado."
        )

    # --------------------------------------------------------
    # Teste de codificação e decodificação
    # --------------------------------------------------------

    samples = [
        "Olá, AYRIS!",
        "Inteligência artificial em português brasileiro.",
        "Ação, coração, ciência, informação e evolução.",
        "Espaços, pontuação: vírgula, ponto. Travessão — 2026.",
        "Texto com emoji: 😀 🚀",
        "Linha um.\nLinha dois.\nLinha três.",
        "Caracteres Unicode: ç, ã, é, ü, ñ.",
    ]

    print("\n[ TESTES DE CODIFICAÇÃO ]")

    for sample in samples:

        encoding = tokenizer.encode(
            sample,
            add_special_tokens=False
        )

        decoded = tokenizer.decode(
            encoding.ids,
            skip_special_tokens=False
        )

        expected = unicodedata.normalize(
            "NFC",
            sample
        )

        if decoded != expected:
            raise RuntimeError(
                "Falha no teste de reconstrução.\n"
                f"Original: {expected!r}\n"
                f"Decodificado: {decoded!r}"
            )

        print(
            f"OK | {len(encoding.ids):4d} tokens | "
            f"{sample[:65]!r}"
        )

    # --------------------------------------------------------
    # Salvar tokenizer
    # --------------------------------------------------------

    tokenizer.save(
        str(tokenizer_path)
    )

    # Reabrir o arquivo salvo para validar persistência.
    loaded = Tokenizer.from_file(
        str(tokenizer_path)
    )

    for sample in samples:

        encoding = loaded.encode(
            sample,
            add_special_tokens=False
        )

        decoded = loaded.decode(
            encoding.ids,
            skip_special_tokens=False
        )

        expected = unicodedata.normalize(
            "NFC",
            sample
        )

        if decoded != expected:
            raise RuntimeError(
                "O tokenizer salvo falhou na reconstrução."
            )

    # --------------------------------------------------------
    # Salvar metadados
    # --------------------------------------------------------

    metadata = {

        "name": "AYRIS",

        "algorithm": "Byte-Level BPE",

        "tokenizers_version": version("tokenizers"),

        "vocab_size": actual_vocab_size,

        "requested_vocab_size": vocab_size,

        "min_frequency": min_frequency,

        "normalization": "NFC",

        "pre_tokenizer": "ByteLevel",

        "decoder": "ByteLevel",

        "initial_alphabet_size": len(alphabet),

        "special_tokens": SPECIAL_TOKENS,

        "special_token_ids": special_ids,

        "training_dataset": str(input_path),

        "training_rows": total_rows,

        "batch_size": BATCH_SIZE,

    }

    with config_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    print("\n[ RESULTADO ]")

    print("Tokenizer treinado com sucesso.")

    print(f"Arquivo: {tokenizer_path}")

    print(f"Configuração: {config_path}")

    print(
        f"Vocabulário final: {actual_vocab_size:,}"
    )

    print(
        f"Tokens especiais: {special_ids}"
    )

    print("\nVALIDAÇÃO: APROVADA")

    print("=" * 65)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Treinamento Byte-Level BPE da AYRIS."
    )

    parser.add_argument(
        "--input",
        default="data/splits/train.parquet"
    )

    parser.add_argument(
        "--output-dir",
        default="checkpoints/tokenizer"
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE
    )

    parser.add_argument(
        "--min-frequency",
        type=int,
        default=DEFAULT_MIN_FREQUENCY
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()

    train_tokenizer(
        input_path=args.input,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
