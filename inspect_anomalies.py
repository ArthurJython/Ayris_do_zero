
import sys
import re
import hashlib
import unicodedata
from pathlib import Path
from collections import Counter

import pyarrow.parquet as pq


CONTROL_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)

HYPHEN_PATTERN = re.compile(
    r"(?P<left>\w)"
    r"(?P<hyphen>[-\u2010\u2011\u2012\u2013\u2014])"
    r"(?P<space>\s+)"
    r"(?P<right>\w)",
    re.UNICODE
)


def context(text, start, end, radius=45):
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return repr(text[left:right])


def inspect(file_path):
    path = Path(file_path)

    if not path.exists():
        print(f"Arquivo não encontrado: {path}")
        return

    parquet = pq.ParquetFile(path)

    control_counts = Counter()

    soft_hyphen_total = 0
    replacement_total = 0
    nfc_total = 0
    newline_total = 0

    duplicate_total = 0
    hashes = {}

    soft_examples = []
    replacement_examples = []
    hyphen_examples = []
    control_examples = []
    duplicate_examples = []

    row_number = 0

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
            row_number += 1

            if not text:
                continue

            # Unicode NFC
            if unicodedata.normalize("NFC", text) != text:
                nfc_total += 1

            # Soft hyphens
            positions = [
                match.start()
                for match in re.finditer("\u00ad", text)
            ]

            soft_hyphen_total += len(positions)

            for position in positions:
                if len(soft_examples) < 8:
                    soft_examples.append(
                        context(text, position, position + 1)
                    )

            # Caracteres de substituição
            positions = [
                match.start()
                for match in re.finditer("\ufffd", text)
            ]

            replacement_total += len(positions)

            for position in positions:
                if len(replacement_examples) < 8:
                    replacement_examples.append(
                        context(text, position, position + 1)
                    )

            # Caracteres de controle
            matches = list(CONTROL_PATTERN.finditer(text))

            control_counts.update(
                match.group() for match in matches
            )

            for match in matches:
                if len(control_examples) < 8:
                    char = match.group()
                    name = unicodedata.name(char, "SEM NOME")

                    control_examples.append(
                        f"{repr(char)} U+{ord(char):04X} "
                        f"{name}: {context(text, match.start(), match.end())}"
                    )

            # Quebras de linha
            newline_total += text.count("\n")

            # Hifenizações suspeitas
            for match in HYPHEN_PATTERN.finditer(text):
                if len(hyphen_examples) < 12:
                    whitespace = match.group("space")

                    hyphen_examples.append(
                        f"Trecho: {context(text, match.start(), match.end())}\n"
                        f"Separador: {repr(whitespace)}"
                    )

            # Duplicação exata
            digest = hashlib.blake2b(
                text.encode("utf-8"),
                digest_size=16
            ).digest()

            if digest in hashes:
                duplicate_total += 1

                if len(duplicate_examples) < 8:
                    duplicate_examples.append(
                        {
                            "atual": (record_id, filename),
                            "original": hashes[digest]
                        }
                    )
            else:
                hashes[digest] = (record_id, filename)

    print("=" * 65)
    print("INVESTIGAÇÃO DE ANOMALIAS — AYRIS")
    print("=" * 65)

    print("\n[CONTAGENS]")
    print(f"Registros que precisam de NFC: {nfc_total:,}")
    print(f"Soft hyphens: {soft_hyphen_total:,}")
    print(f"Caracteres U+FFFD: {replacement_total:,}")
    print(f"Quebras de linha: {newline_total:,}")
    print(f"Duplicatas adicionais: {duplicate_total:,}")

    print("\n[CARACTERES DE CONTROLE]")
    for char, count in control_counts.most_common():
        name = unicodedata.name(char, "SEM NOME")
        print(f"{repr(char)} U+{ord(char):04X} {name}: {count:,}")

    print("\n[EXEMPLOS DE SOFT HYPHENS]")
    for example in soft_examples:
        print(example)

    print("\n[EXEMPLOS DE U+FFFD]")
    for example in replacement_examples:
        print(example)

    print("\n[EXEMPLOS DE CONTROLE]")
    for example in control_examples:
        print(example)

    print("\n[EXEMPLOS DE HIFENIZAÇÃO]")
    for example in hyphen_examples:
        print(example)

    print("\n[EXEMPLOS DE DUPLICATAS]")
    for example in duplicate_examples:
        print(example)

    print("\n" + "=" * 65)
    print("INVESTIGAÇÃO CONCLUÍDA")
    print("=" * 65)


if __name__ == "__main__":
    file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data_0.parquet"
    )

    inspect(file_path)
