
from pathlib import Path
from collections import Counter
from statistics import median

import pyarrow.parquet as pq


SPLITS = {
    "train": Path("data/splits/train.parquet"),
    "validation": Path("data/splits/validation.parquet"),
    "test": Path("data/splits/test.parquet"),
}

reports = {}


for split_name, path in SPLITS.items():
    parquet = pq.ParquetFile(path)

    counts = Counter()
    previous_ids = {}
    seen_ids = set()
    closed_documents = set()

    current_filename = None

    noncontiguous = set()
    id_order_errors = 0
    duplicate_ids = 0

    for batch in parquet.iter_batches(
        batch_size=1024,
        columns=["id", "filename"],
    ):
        ids = batch.column("id").to_pylist()
        filenames = batch.column("filename").to_pylist()

        for record_id, filename in zip(ids, filenames):

            counts[filename] += 1

            if filename != current_filename:
                if filename in closed_documents:
                    noncontiguous.add(filename)

                if current_filename is not None:
                    closed_documents.add(current_filename)

                current_filename = filename

            if record_id in seen_ids:
                duplicate_ids += 1
            else:
                seen_ids.add(record_id)

            previous_id = previous_ids.get(filename)

            if previous_id is not None and record_id <= previous_id:
                id_order_errors += 1

            previous_ids[filename] = record_id

    values = list(counts.values())

    reports[split_name] = set(counts.keys())

    print(f"\n{'=' * 55}")
    print(f"DIVISÃO: {split_name}")
    print(f"{'=' * 55}")

    print("Registros:", parquet.metadata.num_rows)
    print("Documentos:", len(counts))
    print("Fragmentos por documento:")
    print("  Mínimo:", min(values))
    print("  Mediana:", median(values))
    print("  Máximo:", max(values))

    print("Documentos não contíguos:", len(noncontiguous))
    print("Erros de ordenação de ID:", id_order_errors)
    print("IDs duplicados:", duplicate_ids)

print(f"\n{'=' * 55}")
print("VERIFICAÇÃO ENTRE DIVISÕES")
print(f"{'=' * 55}")

names = list(reports.keys())

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a = names[i]
        b = names[j]

        overlap = reports[a] & reports[b]

        print(f"{a} x {b}: {len(overlap)} documentos compartilhados")
