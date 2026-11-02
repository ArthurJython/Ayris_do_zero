
import sys
import re
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

import pyarrow.parquet as pq


CONTROL_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)

REPLACEMENT_PATTERN = re.compile(r"\ufffd+")


def get_context(text, start, end, radius=60):
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return repr(text[left:right])


def inspect(file_path):
    path = Path(file_path)

    if not path.exists():
        print(f"Arquivo não encontrado: {path}")
        return

    parquet = pq.ParquetFile(path)

    counts = Counter()
    affected_rows = Counter()

    samples = defaultdict(list)
    replacement_samples = []

    total_rows = 0

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
            total_rows += 1

            if not text:
                continue

            found_in_row = set()

            # Caracteres de controle
            for match in CONTROL_PATTERN.finditer(text):
                char = match.group()
                cp = ord(char)

                counts[cp] += 1
                found_in_row.add(cp)

                if len(samples[cp]) < 3:
                    name = unicodedata.name(
                        char,
                        "SEM NOME"
                    )

                    samples[cp].append(
                        {
                            "id": record_id,
                            "file": filename,
                            "name": name,
                            "context": get_context(
                                text,
                                match.start(),
                                match.end()
                            )
                        }
                    )

            for cp in found_in_row:
                affected_rows[cp] += 1

            # Sequências de caracteres corrompidos
            for match in REPLACEMENT_PATTERN.finditer(text):
                if len(replacement_samples) < 15:
                    replacement_samples.append(
                        {
                            "id": record_id,
                            "file": filename,
                            "count": len(match.group()),
                            "context": get_context(
                                text,
                                match.start(),
                                match.end()
                            )
                        }
                    )

    print("=" * 65)
    print("INSPEÇÃO DETALHADA — AYRIS")
    print("=" * 65)

    print(f"\nRegistros analisados: {total_rows:,}")

    print("\n[ CARACTERES DE CONTROLE ]")

    for cp, count in counts.most_common():
        char = chr(cp)

        print(
            f"\nU+{cp:04X} | "
            f"{repr(char)} | "
            f"Ocorrências: {count:,} | "
            f"Registros: {affected_rows[cp]:,}"
        )

        # Mostrar possível interpretação Windows-1252
        if 0x80 <= cp <= 0x9F:
            try:
                decoded = bytes([cp]).decode("cp1252")
                print(
                    f"Possível Windows-1252: {repr(decoded)}"
                )
            except UnicodeDecodeError:
                print("Windows-1252: byte sem mapeamento")

        for sample in samples[cp]:
            print(
                f"  ID: {sample['id']} | "
                f"Arquivo: {sample['file']}"
            )
            print(f"  Contexto: {sample['context']}")

    print("\n" + "=" * 65)
    print("[ CARACTERES U+FFFD ]")
    print("=" * 65)

    for sample in replacement_samples:
        print(
            f"\nID: {sample['id']} | "
            f"Arquivo: {sample['file']} | "
            f"Sequência: {sample['count']} caracteres"
        )
        print(f"Contexto: {sample['context']}")

    print("\n" + "=" * 65)
    print("INSPEÇÃO CONCLUÍDA")
    print("=" * 65)


if __name__ == "__main__":
    file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data_0.parquet"
    )

    inspect(file_path)
