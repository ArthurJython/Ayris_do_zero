
import argparse
import hashlib
import json

from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# CONFIGURAÇÃO
# ============================================================

SPLIT_SALT = "AYRIS_SPLIT_V1"

DEFAULT_TRAIN_RATIO = 0.96
DEFAULT_VALIDATION_RATIO = 0.02
DEFAULT_TEST_RATIO = 0.02

BATCH_SIZE = 128

SPLITS = ("train", "validation", "test")


# ============================================================
# ATRIBUIÇÃO DETERMINÍSTICA
# ============================================================

def assign_split(
    filename,
    train_ratio,
    validation_ratio
):
    """
    Mantém todos os registros de um mesmo PDF
    no mesmo conjunto de dados.
    """

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(
            "Foi encontrado um filename nulo ou inválido."
        )

    key = f"{SPLIT_SALT}\0{filename}".encode("utf-8")

    digest = hashlib.blake2b(
        key,
        digest_size=8
    ).digest()

    value = int.from_bytes(
        digest,
        byteorder="big"
    ) / (2 ** 64)

    if value < train_ratio:
        return "train"

    if value < train_ratio + validation_ratio:
        return "validation"

    return "test"


# ============================================================
# PROCESSAMENTO
# ============================================================

def split_dataset(
    input_path,
    output_dir,
    train_ratio,
    validation_ratio,
    test_ratio,
    apply,
    overwrite
):

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # --------------------------------------------------------
    # Validações
    # --------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {input_path}"
        )

    ratios = (
        train_ratio,
        validation_ratio,
        test_ratio
    )

    if any(ratio <= 0 for ratio in ratios):
        raise ValueError(
            "Todas as proporções precisam ser maiores que zero."
        )

    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(
            "A soma das proporções precisa ser igual a 1."
        )

    # --------------------------------------------------------
    # Abrir Parquet
    # --------------------------------------------------------

    parquet = pq.ParquetFile(input_path)

    schema = parquet.schema_arrow

    required_columns = {
        "filename",
        "text"
    }

    missing = required_columns - set(schema.names)

    if missing:
        raise ValueError(
            f"Colunas ausentes: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Preparar saída
    # --------------------------------------------------------

    output_paths = {
        split: output_dir / f"{split}.parquet"
        for split in SPLITS
    }

    manifest_path = output_dir / "split_manifest.json"

    if apply:

        existing = [
            str(path)
            for path in [
                *output_paths.values(),
                manifest_path
            ]
            if path.exists()
        ]

        if existing and not overwrite:
            raise FileExistsError(
                "Arquivos de saída já existem:\n"
                + "\n".join(existing)
                + "\nUse --overwrite para substituí-los."
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    writers = {}

    if apply:

        for split in SPLITS:

            writers[split] = pq.ParquetWriter(
                output_paths[split],
                schema,
                compression="zstd"
            )

    # --------------------------------------------------------
    # Estatísticas
    # --------------------------------------------------------

    stats = {

        split: {
            "rows": 0,
            "characters": 0,
            "documents": set(),
            "null_texts": 0
        }

        for split in SPLITS
    }

    total_rows = 0
    total_characters = 0

    # --------------------------------------------------------
    # Leitura em lotes
    # --------------------------------------------------------

    try:

        for batch in parquet.iter_batches(
            batch_size=BATCH_SIZE
        ):

            data = batch.to_pydict()

            filenames = data["filename"]
            texts = data["text"]

            grouped_indices = defaultdict(list)

            for index, (filename, text) in enumerate(
                zip(filenames, texts)
            ):

                split = assign_split(
                    filename,
                    train_ratio,
                    validation_ratio
                )

                grouped_indices[split].append(index)

                stats[split]["rows"] += 1
                stats[split]["documents"].add(filename)

                total_rows += 1

                if text is None:

                    stats[split]["null_texts"] += 1

                else:

                    length = len(text)

                    stats[split]["characters"] += length
                    total_characters += length

            # ------------------------------------------------
            # Escrever os grupos do lote
            # ------------------------------------------------

            if apply:

                table = pa.Table.from_batches([batch])

                for split, indices in grouped_indices.items():

                    if not indices:
                        continue

                    selected_table = table.take(
                        pa.array(
                            indices,
                            type=pa.int64()
                        )
                    )

                    writers[split].write_table(
                        selected_table
                    )

    finally:

        for writer in writers.values():
            writer.close()

    # ========================================================
    # VERIFICAÇÕES
    # ========================================================

    total_documents = sum(
        len(stats[split]["documents"])
        for split in SPLITS
    )

    document_sets = [
        stats[split]["documents"]
        for split in SPLITS
    ]

    overlap = (
        document_sets[0] & document_sets[1]
        | document_sets[0] & document_sets[2]
        | document_sets[1] & document_sets[2]
    )

    documents_without_overlap = len(overlap) == 0

    rows_sum = sum(
        stats[split]["rows"]
        for split in SPLITS
    )

    characters_sum = sum(
        stats[split]["characters"]
        for split in SPLITS
    )

    rows_ok = rows_sum == total_rows

    characters_ok = (
        characters_sum == total_characters
    )

    # ========================================================
    # RELATÓRIO
    # ========================================================

    print("\n" + "=" * 65)
    print("DIVISÃO DO CORPUS — AYRIS")
    print("=" * 65)

    print(
        f"\nModo: {'APLICAÇÃO' if apply else 'SIMULAÇÃO'}"
    )

    print(f"Registros analisados: {total_rows:,}")
    print(f"Documentos únicos: {total_documents:,}")
    print(f"Caracteres analisados: {total_characters:,}")

    print("\n[ DIVISÃO ]")

    for split in SPLITS:

        current = stats[split]

        char_ratio = (
            current["characters"] / total_characters * 100
            if total_characters
            else 0
        )

        print(f"\n{split.upper()}")

        print(
            f"Documentos: "
            f"{len(current['documents']):,}"
        )

        print(
            f"Registros: "
            f"{current['rows']:,}"
        )

        print(
            f"Caracteres: "
            f"{current['characters']:,}"
        )

        print(
            f"Participação em caracteres: "
            f"{char_ratio:.2f}%"
        )

        print(
            f"Textos nulos: "
            f"{current['null_texts']:,}"
        )

    print("\n[ VALIDAÇÕES ]")

    print(
        f"Sem documentos compartilhados: "
        f"{'OK' if documents_without_overlap else 'FALHOU'}"
    )

    print(
        f"Contagem de registros: "
        f"{'OK' if rows_ok else 'FALHOU'}"
    )

    print(
        f"Contagem de caracteres: "
        f"{'OK' if characters_ok else 'FALHOU'}"
    )

    # --------------------------------------------------------
    # Manifesto
    # --------------------------------------------------------

    manifest = {

        "input": str(input_path),

        "method": "blake2b",

        "salt": SPLIT_SALT,

        "requested_ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio
        },

        "statistics": {},

        "validation": {
            "documents_without_overlap": documents_without_overlap,
            "rows_match": rows_ok,
            "characters_match": characters_ok
        }

    }

    for split in SPLITS:

        current = stats[split]

        manifest["statistics"][split] = {

            "documents": len(current["documents"]),

            "rows": current["rows"],

            "characters": current["characters"],

            "null_texts": current["null_texts"],

            "filenames": sorted(current["documents"])

        }

    # --------------------------------------------------------
    # Salvar manifesto
    # --------------------------------------------------------

    if apply:

        with manifest_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                manifest,
                file,
                ensure_ascii=False,
                indent=2
            )

        print("\n[ ARQUIVOS GERADOS ]")

        for split in SPLITS:
            print(output_paths[split])

        print(manifest_path)

    else:

        print("\nNenhum arquivo foi criado.")
        print("Utilize --apply para executar a divisão.")

    print("\n" + "=" * 65)

    if not (
        documents_without_overlap
        and rows_ok
        and characters_ok
    ):

        raise RuntimeError(
            "A validação da divisão apresentou falhas."
        )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Divisão do corpus da AYRIS por documento."
    )

    parser.add_argument(
        "input",
        help="Parquet limpo de entrada."
    )

    parser.add_argument(
        "--output-dir",
        default="data/splits",
        help="Diretório de saída."
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Salva os conjuntos de dados."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite substituir os arquivos existentes."
    )

    args = parser.parse_args()

    split_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        apply=args.apply,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
