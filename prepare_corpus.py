
import hashlib
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

PATH = Path("data/processed/data_0_clean.parquet")

if not PATH.exists():
    raise FileNotFoundError(PATH)

parquet = pq.ParquetFile(PATH)

rows = 0
chars = 0
nulls = 0
bad_c0 = 0
bad_c1 = 0
replacement = 0
not_nfc = 0
length_errors = 0
duplicates = 0

seen = set()

for batch in parquet.iter_batches(
    batch_size=128,
    columns=["text", "n_chars"]
):
    data = batch.to_pydict()

    for text, n_chars in zip(
        data["text"],
        data["n_chars"]
    ):
        rows += 1

        if text is None:
            nulls += 1
            continue

        chars += len(text)

        if len(text) != n_chars:
            length_errors += 1

        if unicodedata.normalize("NFC", text) != text:
            not_nfc += 1

        for char in text:
            code = ord(char)

            if code < 32 and code not in (9, 10, 13):
                bad_c0 += 1

            if 0x7F <= code <= 0x9F:
                bad_c1 += 1

            if code == 0xFFFD:
                replacement += 1

        digest = hashlib.blake2b(
            text.encode("utf-8"),
            digest_size=32
        ).digest()

        if digest in seen:
            duplicates += 1
        else:
            seen.add(digest)

print("\n=== VALIDAÇÃO DO CORPUS AYRIS ===")
print(f"Registros: {rows:,}")
print(f"Caracteres: {chars:,}")
print(f"Nulos: {nulls}")
print(f"Controles C0: {bad_c0}")
print(f"Controles C1: {bad_c1}")
print(f"U+FFFD: {replacement}")
print(f"Textos fora de NFC: {not_nfc}")
print(f"Erros em n_chars: {length_errors}")
print(f"Duplicatas: {duplicates}")

print("\n=== RESULTADO ===")

checks = {
    "Registros": rows == 27400,
    "Nulos": nulls == 0,
    "Controles C0": bad_c0 == 0,
    "Controles C1": bad_c1 == 0,
    "U+FFFD": replacement == 0,
    "NFC": not_nfc == 0,
    "n_chars": length_errors == 0,
    "Duplicatas": duplicates == 0,
}

for name, result in checks.items():
    print(f"{name}: {'OK' if result else 'FALHOU'}")

print(
    "\nCORPUS:",
    "APROVADO" if all(checks.values()) else "REVISAR"
)
