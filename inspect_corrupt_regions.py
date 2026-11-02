
import sys
from pathlib import Path

import pyarrow.parquet as pq


# Caracteres C0 problemáticos, exceto TAB, LF e CR.
C0 = (
    set(range(0x00, 0x09))
    | {0x0B, 0x0C}
    | set(range(0x0E, 0x20))
)

BAD_CHARS = C0 | {0xFFFD}


def find_groups(text, max_gap=8, min_count=5):
    positions = [
        i for i, char in enumerate(text)
        if ord(char) in BAD_CHARS
    ]

    groups = []

    if not positions:
        return groups

    start = positions[0]
    previous = positions[0]
    count = 1

    for position in positions[1:]:
        if position - previous <= max_gap:
            count += 1
        else:
            if count >= min_count:
                groups.append((start, previous, count))

            start = position
            count = 1

        previous = position

    if count >= min_count:
        groups.append((start, previous, count))

    return groups


def inspect(file_path):
    path = Path(file_path)

    if not path.exists():
        print(f"Arquivo não encontrado: {path}")
        return

    parquet = pq.ParquetFile(path)

    candidates = []

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

            for start, end, count in find_groups(text):
                span = end - start + 1
                density = count / span

                radius = 100

                candidates.append({
                    "id": record_id,
                    "filename": filename,
                    "start": start,
                    "end": end,
                    "count": count,
                    "density": density,
                    "context": repr(
                        text[
                            max(0, start - radius):
                            min(len(text), end + radius + 1)
                        ]
                    )
                })

    candidates.sort(
        key=lambda item: (
            item["count"],
            item["density"]
        ),
        reverse=True
    )

    print("=" * 70)
    print("REGIÕES CORROMPIDAS — AYRIS")
    print("=" * 70)

    print(f"Regiões suspeitas identificadas: {len(candidates)}")

    for item in candidates[:30]:
        print("\n" + "-" * 70)

        print(f"ID: {item['id']}")
        print(f"PDF: {item['filename']}")
        print(f"Posição inicial: {item['start']}")
        print(f"Posição final: {item['end']}")
        print(f"Caracteres problemáticos: {item['count']}")
        print(f"Densidade: {item['density']:.2%}")

        print(f"\nContexto:\n{item['context']}")

    print("\n" + "=" * 70)
    print("INVESTIGAÇÃO CONCLUÍDA")
    print("=" * 70)


if __name__ == "__main__":
    file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data_0.parquet"
    )

    inspect(file_path)
