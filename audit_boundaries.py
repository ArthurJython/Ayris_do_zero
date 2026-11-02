
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


TARGETS = {
    "train": {
        "ue000323.pdf",
        "ue000125.pdf",
        "cp000020.pdf",
        "vo000022.pdf",
        "ue000318.pdf",
        "bv000298.pdf",
        "cp000121.pdf",
        "cp000176.pdf",
        "cp000077.pdf",
        "ci000069.pdf",
        "cp000112.pdf",
        "cp000033.pdf",
        "cp000085.pdf",
    },
    "validation": {"cp000043.pdf"},
    "test": {"cp000005.pdf"},
}

PATHS = {
    "train": Path("data/splits/train.parquet"),
    "validation": Path("data/splits/validation.parquet"),
    "test": Path("data/splits/test.parquet"),
}


for split, path in PATHS.items():

    records = defaultdict(list)
    row_index = 0

    parquet = pq.ParquetFile(path)

    for batch in parquet.iter_batches(
        batch_size=512,
        columns=["id", "filename", "text"],
    ):
        ids = batch.column("id").to_pylist()
        filenames = batch.column("filename").to_pylist()
        texts = batch.column("text").to_pylist()

        for record_id, filename, text in zip(ids, filenames, texts):

            if filename in TARGETS[split]:
                records[filename].append(
                    (record_id, row_index, text)
                )

            row_index += 1

    print(f"\n{'=' * 65}")
    print(f"DIVISÃO: {split}")
    print(f"{'=' * 65}")

    for filename, fragments in sorted(records.items()):

        fragments.sort(key=lambda item: item[0])

        print(f"\nDOCUMENTO: {filename}")

        for previous, current in zip(
            fragments,
            fragments[1:],
        ):

            previous_id, previous_row, previous_text = previous
            current_id, current_row, current_text = current

            # Mostra as junções entre regiões separadas no Parquet.
            if current_row != previous_row + 1:

                print(
                    f"\nFragmento {previous_id} -> {current_id}"
                )

                print(
                    "Final anterior:",
                    repr(previous_text[-120:])
                )

                print(
                    "Início seguinte:",
                    repr(current_text[:120])
                )

                print(
                    "Espaço no final:",
                    previous_text[-1:].isspace()
                )

                print(
                    "Espaço no início:",
                    current_text[:1].isspace()
                )

        print("Total de fragmentos:", len(fragments))
