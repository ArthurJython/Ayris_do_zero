
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


SPLITS = {
    "train": Path("data/splits/train.parquet"),
    "validation": Path("data/splits/validation.parquet"),
    "test": Path("data/splits/test.parquet"),
}


for split_name, path in SPLITS.items():

    parquet = pq.ParquetFile(path)

    records = []
    positions = defaultdict(list)

    row_index = 0

    for batch in parquet.iter_batches(
        batch_size=1024,
        columns=["id", "filename"],
    ):
        ids = batch.column("id").to_pylist()
        filenames = batch.column("filename").to_pylist()

        for record_id, filename in zip(ids, filenames):
            records.append((row_index, record_id, filename))
            positions[filename].append((row_index, record_id))
            row_index += 1

    print(f"\n{'=' * 65}")
    print(f"DIVISÃO: {split_name}")
    print(f"{'=' * 65}")

    for filename, entries in positions.items():

        # Identificar interrupções na sequência de linhas.
        runs = []
        current_run = [entries[0]]

        for previous, current in zip(entries, entries[1:]):

            if current[0] != previous[0] + 1:
                runs.append(current_run)
                current_run = [current]

            else:
                current_run.append(current)

        runs.append(current_run)

        # Identificar IDs fora de ordem.
        order_errors = []

        for previous, current in zip(entries, entries[1:]):

            if current[1] <= previous[1]:
                order_errors.append((previous, current))

        if len(runs) > 1 or order_errors:

            print(f"\nDocumento: {filename}")

            if len(runs) > 1:
                print(f"  Fragmentado em {len(runs)} regiões:")

                for run in runs:
                    print(
                        f"    Linhas {run[0][0]}-{run[-1][0]} | "
                        f"IDs {run[0][1]}-{run[-1][1]} | "
                        f"{len(run)} registros"
                    )

            if order_errors:
                print("  IDs fora de ordem:")

                for previous, current in order_errors:
                    print(
                        f"    Linha {previous[0]}: ID {previous[1]} "
                        f"-> Linha {current[0]}: ID {current[1]}"
                    )
