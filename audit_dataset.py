
import sys
import re
import hashlib
import unicodedata
from pathlib import Path
from collections import Counter

import pyarrow.parquet as pq


def audit_dataset(file_path):
    path = Path(file_path)

    if not path.exists():
        print(f"[ERRO] Arquivo não encontrado: {path}")
        return

    parquet = pq.ParquetFile(path)

    total_rows = 0
    total_chars = 0
    total_words = 0

    null_texts = 0
    empty_texts = 0
    nfc_rows = 0
    soft_hyphens = 0
    replacement_chars = 0
    control_chars = 0
    invalid_lengths = 0

    duplicate_texts = 0
    hashes = set()

    filenames = Counter()
    book_types = Counter()

    lengths = []
    hyphen_examples = []

    hyphen_pattern = re.compile(
        r"(?<=[^\W_])[-‐‑]\s+(?=[^\W_])",
        re.UNICODE
    )

    print("=" * 60)
    print("AUDITORIA DO DATASET — AYRIS")
    print("=" * 60)

    for batch in parquet.iter_batches(
        batch_size=256,
        columns=["text", "n_chars", "filename", "book_type"]
    ):
        data = batch.to_pydict()

        for text, declared_chars, filename, book_type in zip(
            data["text"],
            data["n_chars"],
            data["filename"],
            data["book_type"]
        ):
            total_rows += 1

            if filename is not None:
                filenames[filename] += 1

            if book_type is not None:
                book_types[book_type] += 1

            if text is None:
                null_texts += 1
                continue

            if not text.strip():
                empty_texts += 1
                continue

            current_length = len(text)

            total_chars += current_length
            total_words += len(text.split())
            lengths.append(current_length)

            if declared_chars is not None:
                if declared_chars != current_length:
                    invalid_lengths += 1

            if unicodedata.normalize("NFC", text) != text:
                nfc_rows += 1

            soft_hyphens += text.count("\u00ad")
            replacement_chars += text.count("\ufffd")

            control_chars += sum(
                1 for char in text
                if ord(char) < 32 and char not in "\n\r\t"
            )

            digest = hashlib.blake2b(
                text.encode("utf-8"),
                digest_size=16
            ).digest()

            if digest in hashes:
                duplicate_texts += 1
            else:
                hashes.add(digest)

            for match in hyphen_pattern.finditer(text):
                if len(hyphen_examples) < 10:
                    start = max(0, match.start() - 25)
                    end = min(len(text), match.end() + 25)

                    hyphen_examples.append(
                        repr(text[start:end])
                    )

    print("\n[ VOLUME ]")
    print(f"Registros: {total_rows:,}")
    print(f"Caracteres: {total_chars:,}")
    print(f"Palavras aproximadas: {total_words:,}")

    if lengths:
        print(f"Média de caracteres por registro: {total_chars / len(lengths):,.1f}")
        print(f"Menor registro: {min(lengths):,}")
        print(f"Maior registro: {max(lengths):,}")

    print("\n[ QUALIDADE ]")
    print(f"Textos nulos: {null_texts:,}")
    print(f"Textos vazios: {empty_texts:,}")
    print(f"Textos que precisam de NFC: {nfc_rows:,}")
    print(f"Soft hyphens: {soft_hyphens:,}")
    print(f"Caracteres de substituição U+FFFD: {replacement_chars:,}")
    print(f"Caracteres de controle: {control_chars:,}")
    print(f"Registros com n_chars divergente: {invalid_lengths:,}")

    print("\n[ DUPLICAÇÃO ]")
    print(f"Textos duplicados adicionais: {duplicate_texts:,}")
    print(f"Textos únicos: {len(hashes):,}")

    print("\n[ ORIGEM ]")
    print(f"Arquivos diferentes: {len(filenames)}")

    for name, count in filenames.most_common(10):
        print(f"  {name}: {count:,}")

    print("\nTipos de livro:")

    for name, count in book_types.most_common():
        print(f"  {name}: {count:,}")

    print("\n[ POSSÍVEIS HIFENIZAÇÕES ]")

    for example in hyphen_examples:
        print(f"  {example}")

    print("\n" + "=" * 60)
    print("AUDITORIA CONCLUÍDA")
    print("=" * 60)


if __name__ == "__main__":
    file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data_0.parquet"
    )

    audit_dataset(file_path)
