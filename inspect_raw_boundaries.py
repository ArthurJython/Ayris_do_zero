
from pathlib import Path

import pyarrow.parquet as pq


SOURCE = Path("data_0.parquet")

PAIRS = {
    "ue000323.pdf": (103918, 103919),
    "ue000125.pdf": (100323, 100324),
    "cp000020.pdf": (32631, 32632),
    "cp000033.pdf": (33215, 33216),
    "cp000077.pdf": (34960, 34961),
    "cp000112.pdf": (35986, 35987),
    "cp000121.pdf": (36219, 36220),
    "cp000176.pdf": (37907, 37908),
    "cp000043.pdf": (33660, 33661),
    "cp000005.pdf": (32115, 32116),
}

if not SOURCE.exists():
    raise SystemExit(f"Arquivo não encontrado: {SOURCE}")

parquet = pq.ParquetFile(SOURCE)

print("Arquivo:", SOURCE)
print("Registros:", parquet.metadata.num_rows)
print("Schema:", parquet.schema_arrow)

targets = {
    (filename, record_id)
    for filename, pair in PAIRS.items()
    for record_id in pair
}

found = {}

for batch in parquet.iter_batches(
    batch_size=2048,
    columns=["id", "filename", "text"],
):
    ids = batch.column("id").to_pylist()
    filenames = batch.column("filename").to_pylist()
    texts = batch.column("text").to_pylist()

    for record_id, filename, text in zip(ids, filenames, texts):
        key = (filename, record_id)

        if key in targets:
            found[key] = text

for filename, (previous_id, next_id) in PAIRS.items():

    previous = found.get((filename, previous_id))
    following = found.get((filename, next_id))

    print("\n" + "=" * 65)
    print(f"{filename}: {previous_id} -> {next_id}")

    if previous is None or following is None:
        print("REGISTROS NÃO ENCONTRADOS")
        continue

    print("Final anterior:", repr(previous[-120:]))
    print("Início seguinte:", repr(following[:120]))

    print("Espaço no final:", previous[-1:].isspace())
    print("Espaço no início:", following[:1].isspace())

    print(
        "Último caractere:",
        f"U+{ord(previous[-1]):04X}"
    )

    print(
        "Primeiro caractere:",
        f"U+{ord(following[0]):04X}"
    )

    print("Total de caracteres:", len(previous) + len(following))

print("\nRegistros encontrados:", len(found), "/", len(targets))
