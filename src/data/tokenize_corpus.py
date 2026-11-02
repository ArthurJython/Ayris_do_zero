
import argparse
import hashlib
import json

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_PATH = ROOT / "checkpoints/tokenizer/tokenizer.json"

OUTPUT_DIR = ROOT / "data/tokenized"

SPLITS = {
    "train": ROOT / "data/splits/train.parquet",
    "validation": ROOT / "data/splits/validation.parquet",
    "test": ROOT / "data/splits/test.parquet",
}

BATCH_SIZE = 128

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
    "<SYS>": 4,
    "<USR>": 5,
    "<AST>": 6,
}

TOKEN_DTYPE = np.dtype("<u2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_tokenizer() -> Tokenizer:

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(TOKENIZER_PATH)

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    vocab_size = tokenizer.get_vocab_size()

    if vocab_size != 4096:
        raise ValueError(
            f"Vocabulário inesperado: {vocab_size}"
        )

    for token, expected_id in SPECIAL_TOKENS.items():

        actual_id = tokenizer.token_to_id(token)

        if actual_id != expected_id:
            raise ValueError(
                f"ID incorreto para {token}: "
                f"esperado {expected_id}, recebido {actual_id}"
            )

    return tokenizer


def inspect_document_bounds(parquet_path: Path):

    parquet = pq.ParquetFile(parquet_path)

    bounds = {}
    seen_ids = set()

    total_rows = 0

    for batch in parquet.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["id", "filename"],
    ):

        ids = batch.column("id").to_pylist()
        filenames = batch.column("filename").to_pylist()

        for record_id, filename in zip(ids, filenames):

            if record_id is None:
                raise ValueError("ID nulo encontrado.")

            if not isinstance(filename, str) or not filename:
                raise ValueError("Filename inválido.")

            if record_id in seen_ids:
                raise ValueError(
                    f"ID duplicado encontrado: {record_id}"
                )

            seen_ids.add(record_id)

            if filename not in bounds:

                bounds[filename] = {
                    "first_id": record_id,
                    "last_id": record_id,
                }

            else:

                bounds[filename]["first_id"] = min(
                    bounds[filename]["first_id"],
                    record_id,
                )

                bounds[filename]["last_id"] = max(
                    bounds[filename]["last_id"],
                    record_id,
                )

            total_rows += 1

    if total_rows != parquet.metadata.num_rows:
        raise RuntimeError(
            "Quantidade de registros inconsistente."
        )

    return bounds


def tokenize_split(
    tokenizer: Tokenizer,
    split_name: str,
    parquet_path: Path,
    output_dir: Path,
):

    print(f"\n{'=' * 60}")
    print(f"TOKENIZANDO: {split_name.upper()}")
    print(f"{'=' * 60}")

    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    parquet = pq.ParquetFile(parquet_path)

    # Primeira passagem:
    # identifica os limites de ID de cada documento.
    print("Analisando fronteiras dos documentos...")

    bounds = inspect_document_bounds(parquet_path)

    bos_id = SPECIAL_TOKENS["<BOS>"]
    eos_id = SPECIAL_TOKENS["<EOS>"]

    output_bin = output_dir / f"{split_name}.bin"
    output_index = output_dir / f"{split_name}.index.jsonl"

    temp_bin = output_dir / f"{split_name}.bin.tmp"
    temp_index = output_dir / f"{split_name}.index.jsonl.tmp"

    total_tokens = 0
    total_text_tokens = 0
    total_fragments = 0
    short_fragments = 0

    min_sequence_length = None
    max_sequence_length = 0

    try:

        # Segunda passagem:
        # tokeniza cada fragmento sem concatená-lo.
        with (
            temp_bin.open("wb") as bin_file,
            temp_index.open("w", encoding="utf-8") as index_file,
        ):

            for batch in parquet.iter_batches(
                batch_size=BATCH_SIZE,
                columns=["id", "filename", "text", "n_chars"],
            ):

                ids = batch.column("id").to_pylist()
                filenames = batch.column("filename").to_pylist()
                texts = batch.column("text").to_pylist()
                lengths = batch.column("n_chars").to_pylist()

                for record_id, filename, text, n_chars in zip(
                    ids,
                    filenames,
                    texts,
                    lengths,
                ):

                    if text is None:
                        raise ValueError(
                            f"Texto nulo no registro {record_id}"
                        )

                    if len(text) != n_chars:
                        raise ValueError(
                            f"n_chars incorreto no ID {record_id}"
                        )

                    document = bounds[filename]

                    is_first = (
                        record_id == document["first_id"]
                    )

                    is_last = (
                        record_id == document["last_id"]
                    )

                    # Não adicionamos tokens especiais
                    # automaticamente pelo tokenizer.
                    encoded = tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    )

                    text_ids = encoded.ids

                    if not text_ids:
                        raise ValueError(
                            f"Fragmento vazio: {record_id}"
                        )

                    sequence = []

                    # BOS somente no início real do PDF.
                    if is_first:
                        sequence.append(bos_id)

                    sequence.extend(text_ids)

                    # EOS somente no final real do PDF.
                    if is_last:
                        sequence.append(eos_id)

                    if max(sequence) >= 65536:
                        raise ValueError(
                            "ID incompatível com uint16."
                        )

                    array = np.asarray(
                        sequence,
                        dtype=TOKEN_DTYPE,
                    )

                    offset = total_tokens
                    sequence_length = len(array)

                    # Escreve os IDs no arquivo binário.
                    bin_file.write(array.tobytes())

                    # Índice em unidades de tokens,
                    # não em bytes.
                    metadata = {
                        "offset": offset,
                        "length": sequence_length,
                        "record_id": int(record_id),
                        "filename": filename,
                        "text_tokens": len(text_ids),
                        "has_bos": is_first,
                        "has_eos": is_last,
                    }

                    index_file.write(
                        json.dumps(
                            metadata,
                            ensure_ascii=False,
                        ) + "\n"
                    )

                    total_tokens += sequence_length
                    total_text_tokens += len(text_ids)
                    total_fragments += 1

                    min_sequence_length = (
                        sequence_length
                        if min_sequence_length is None
                        else min(
                            min_sequence_length,
                            sequence_length,
                        )
                    )

                    max_sequence_length = max(
                        max_sequence_length,
                        sequence_length,
                    )

                    # O DataLoader precisará de pelo menos
                    # 513 tokens para produzir X e Y de
                    # comprimento 512.
                    if sequence_length < 513:
                        short_fragments += 1

        # Verifica o tamanho físico antes de publicar.
        expected_bytes = total_tokens * TOKEN_DTYPE.itemsize
        actual_bytes = temp_bin.stat().st_size

        if actual_bytes != expected_bytes:
            raise RuntimeError(
                "Tamanho do arquivo binário inconsistente."
            )

        # Publicação após conclusão bem-sucedida.
        temp_bin.replace(output_bin)
        temp_index.replace(output_index)

    except Exception:

        temp_bin.unlink(missing_ok=True)
        temp_index.unlink(missing_ok=True)

        raise

    result = {
        "records": total_fragments,
        "documents": len(bounds),
        "text_tokens": total_text_tokens,
        "total_tokens": total_tokens,
        "bytes": total_tokens * TOKEN_DTYPE.itemsize,
        "min_sequence_length": min_sequence_length,
        "max_sequence_length": max_sequence_length,
        "short_fragments": short_fragments,
        "binary_file": str(output_bin.relative_to(ROOT)),
        "index_file": str(output_index.relative_to(ROOT)),
    }

    print("\nResultado:")
    print("Fragmentos:", total_fragments)
    print("Documentos:", len(bounds))
    print("Tokens textuais:", f"{total_text_tokens:,}")
    print("Tokens totais:", f"{total_tokens:,}")
    print("Tamanho binário:", f"{result['bytes'] / 1024 / 1024:.2f} MiB")
    print("Menor sequência:", min_sequence_length)
    print("Maior sequência:", max_sequence_length)
    print("Fragmentos menores que 513 tokens:", short_fragments)

    return result


def main():

    parser = argparse.ArgumentParser(
        description="Prepara os dados tokenizados da AYRIS."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Substitui arquivos existentes.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = load_tokenizer()

    expected_outputs = [
        OUTPUT_DIR / f"{name}.bin"
        for name in SPLITS
    ] + [
        OUTPUT_DIR / f"{name}.index.jsonl"
        for name in SPLITS
    ] + [
        OUTPUT_DIR / "manifest.json"
    ]

    existing = [
        path for path in expected_outputs
        if path.exists()
    ]

    if existing and not args.overwrite:
        raise FileExistsError(
            "Arquivos já existem. Use --overwrite para substituí-los:\n"
            + "\n".join(str(path) for path in existing)
        )

    print("=" * 60)
    print("PREPARAÇÃO DO DATASET — AYRIS")
    print("=" * 60)

    results = {}

    for split_name, parquet_path in SPLITS.items():

        results[split_name] = tokenize_split(
            tokenizer=tokenizer,
            split_name=split_name,
            parquet_path=parquet_path,
            output_dir=OUTPUT_DIR,
        )

    manifest = {
        "project": "AYRIS",
        "format": "uint16",
        "byte_order": "little",
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": sha256_file(TOKENIZER_PATH),
        "special_tokens": SPECIAL_TOKENS,
        "context_length": 512,
        "boundary_policy": (
            "Fragments are tokenized independently. "
            "No artificial concatenation. BOS is added only "
            "to the lowest ID fragment of each filename; "
            "EOS only to the highest ID fragment. "
            "Training windows must never cross fragment boundaries."
        ),
        "splits": results,
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    temp_manifest = OUTPUT_DIR / "manifest.json.tmp"

    temp_manifest.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temp_manifest.replace(manifest_path)

    print("\n" + "=" * 60)
    print("PREPARAÇÃO CONCLUÍDA")
    print("=" * 60)

    print("Manifesto:", manifest_path)
    print("Diretório:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
