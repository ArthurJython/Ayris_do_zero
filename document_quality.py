
import sys
import re
from pathlib import Path
from collections import defaultdict

import pyarrow.parquet as pq


CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)

REPLACEMENT_RE = re.compile(r"\ufffd")


def analyze(file_path):
    path = Path(file_path)

    if not path.exists():
        print(f"Arquivo não encontrado: {path}")
        return

    parquet = pq.ParquetFile(path)

    documents = defaultdict(lambda: {
        "rows": 0,
        "chars": 0,
        "controls": 0,
        "replacement": 0,
        "soft_hyphens": 0,
        "affected_rows": 0,
    })

    worst_records = []

    for batch in parquet.iter_batches(
        batch_size=256,
        columns=["id", "filename", "text"]
    ):
        data = batch.to_pydict()

        for record_id, filename, text in zip(
            data["id"],
            data["filename"],
            data["text"]
        ):
            if not text:
                continue

            stats = documents[filename]

            controls = len(CONTROL_RE.findall(text))
            replacements = len(REPLACEMENT_RE.findall(text))
            soft_hyphens = text.count("\u00ad")

            bad = controls + replacements

            stats["rows"] += 1
            stats["chars"] += len(text)
            stats["controls"] += controls
            stats["replacement"] += replacements
            stats["soft_hyphens"] += soft_hyphens

            if bad:
                stats["affected_rows"] += 1

            ratio = bad / len(text) * 100

            if bad:
                match = CONTROL_RE.search(text)

                if match:
                    start = max(0, match.start() - 70)
                    end = min(len(text), match.end() + 110)
                    sample = repr(text[start:end])
                else:
                    match = REPLACEMENT_RE.search(text)
                    start = max(0, match.start() - 70)
                    end = min(len(text), match.end() + 110)
                    sample = repr(text[start:end])

                worst_records.append({
                    "id": record_id,
                    "filename": filename,
                    "chars": len(text),
                    "controls": controls,
                    "replacement": replacements,
                    "bad": bad,
                    "ratio": ratio,
                    "sample": sample
                })

    print("=" * 75)
    print("QUALIDADE POR DOCUMENTO — AYRIS")
    print("=" * 75)

    print("\n[ 15 DOCUMENTOS COM MAIS ANOMALIAS ]")

    ordered = sorted(
        documents.items(),
        key=lambda item: (
            item[1]["controls"] + item[1]["replacement"]
        ),
        reverse=True
    )

    for filename, stats in ordered[:15]:
        bad = stats["controls"] + stats["replacement"]

        ratio = (
            bad / stats["chars"] * 100
            if stats["chars"] else 0
        )

        print(f"\n{filename}")
        print(f"  Registros: {stats['rows']:,}")
        print(f"  Caracteres: {stats['chars']:,}")
        print(f"  Controles: {stats['controls']:,}")
        print(f"  U+FFFD: {stats['replacement']:,}")
        print(f"  Soft hyphens: {stats['soft_hyphens']:,}")
        print(f"  Registros afetados: {stats['affected_rows']:,}")
        print(f"  Corrupção observada: {ratio:.4f}%")

    print("\n" + "=" * 75)
    print("[ 15 REGISTROS COM MAIOR PROPORÇÃO DE ANOMALIAS ]")
    print("=" * 75)

    worst_records.sort(
        key=lambda item: item["ratio"],
        reverse=True
    )

    for record in worst_records[:15]:
        print(f"\nID: {record['id']}")
        print(f"Arquivo: {record['filename']}")
        print(f"Caracteres: {record['chars']:,}")
        print(f"Controles: {record['controls']:,}")
        print(f"U+FFFD: {record['replacement']:,}")
        print(f"Proporção: {record['ratio']:.4f}%")
        print(f"Trecho: {record['sample']}")

    print("\n" + "=" * 75)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 75)


if __name__ == "__main__":
    file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data_0.parquet"
    )

    analyze(file_path)
