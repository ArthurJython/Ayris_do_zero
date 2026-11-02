
import sys
import re
from pathlib import Path
from collections import defaultdict

import pyarrow.parquet as pq


# Controles C0: caracteres não imprimíveis
C0_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Controles C1: precisam de investigação separada
C1_PATTERN = re.compile(r"[\x7f-\x9f]")

# Caracteres irrecuperáveis
REPLACEMENT = "\ufffd"


def analyze(file_path):
    path = Path(file_path)

    if not path.exists():
        print(f"[ERRO] Arquivo não encontrado: {path}")
        return

    parquet = pq.ParquetFile(path)

    total_rows = 0
    total_chars = 0

    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]

    discarded = {
        threshold: {"rows": 0, "chars": 0}
        for threshold in thresholds
    }

    worst_records = []
    documents = defaultdict(
        lambda: {"rows": 0, "c0": 0, "c1": 0, "fffd": 0}
    )

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

            total_rows += 1
            length = len(text)
            total_chars += length

            c0 = len(C0_PATTERN.findall(text))
            c1 = len(C1_PATTERN.findall(text))
            fffd = text.count(REPLACEMENT)

            # C0 e U+FFFD são contabilizados como anomalias graves.
            severe = c0 + fffd
            ratio = severe / length

            doc = documents[filename]
            doc["rows"] += 1
            doc["c0"] += c0
            doc["c1"] += c1
            doc["fffd"] += fffd

            for threshold in thresholds:
                if ratio >= threshold:
                    discarded[threshold]["rows"] += 1
                    discarded[threshold]["chars"] += length

            if severe:
                worst_records.append({
                    "id": record_id,
                    "filename": filename,
                    "chars": length,
                    "c0": c0,
                    "c1": c1,
                    "fffd": fffd,
                    "soft": text.count("\u00ad"),
                    "ratio": ratio * 100
                })

    print("=" * 65)
    print("DISTRIBUIÇÃO DA CORRUPÇÃO — AYRIS")
    print("=" * 65)

    print(f"\nRegistros: {total_rows:,}")
    print(f"Caracteres: {total_chars:,}")

    print("\n[ IMPACTO DE EXCLUIR REGISTROS ]")

    for threshold, stats in discarded.items():
        percentage = (
            stats["chars"] / total_chars * 100
            if total_chars else 0
        )

        print(
            f"\nLimite: {threshold * 100:.1f}% de anomalias"
            f"\n  Registros descartados: {stats['rows']:,}"
            f"\n  Caracteres descartados: {stats['chars']:,}"
            f"\n  Percentual do corpus: {percentage:.4f}%"
        )

    print("\n[ 20 REGISTROS MAIS COMPROMETIDOS ]")

    worst_records.sort(
        key=lambda record: record["ratio"],
        reverse=True
    )

    for record in worst_records[:20]:
        print(
            f"\nID: {record['id']}"
            f"\nArquivo: {record['filename']}"
            f"\nCaracteres: {record['chars']:,}"
            f"\nC0: {record['c0']:,}"
            f"\nC1: {record['c1']:,}"
            f"\nU+FFFD: {record['fffd']:,}"
            f"\nSoft hyphens: {record['soft']:,}"
            f"\nAnomalias graves: {record['ratio']:.4f}%"
        )

    print("\n[ DOCUMENTOS COM MAIS ANOMALIAS C0/U+FFFD ]")

    ordered = sorted(
        documents.items(),
        key=lambda item: item[1]["c0"] + item[1]["fffd"],
        reverse=True
    )

    for filename, stats in ordered[:15]:
        print(
            f"{filename}: "
            f"C0={stats['c0']:,}, "
            f"C1={stats['c1']:,}, "
            f"U+FFFD={stats['fffd']:,}"
        )

    print("\n" + "=" * 65)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 65)


if __name__ == "__main__":
    file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data_0.parquet"
    )

    analyze(file_path)
