
import sys
from pathlib import Path

import pyarrow.parquet as pq


def inspect_parquet(file_path):
    path = Path(file_path)

    if not path.exists():
        print(f"[ERRO] Arquivo não encontrado: {path}")
        return

    print("=" * 60)
    print("ANÁLISE DO DATASET — AYRIS")
    print("=" * 60)

    print(f"\nArquivo: {path.resolve()}")
    print(f"Tamanho: {path.stat().st_size / (1024**2):.2f} MB")

    try:
        parquet = pq.ParquetFile(path)
    except Exception as error:
        print(f"[ERRO] Não foi possível abrir o Parquet: {error}")
        return

    metadata = parquet.metadata

    print(f"\nTotal de registros: {metadata.num_rows:,}")
    print(f"Total de colunas: {metadata.num_columns}")
    print(f"Grupos de linhas: {metadata.num_row_groups}")
    print(f"Formato Parquet: {metadata.format_version}")

    print("\n" + "=" * 60)
    print("ESTRUTURA DAS COLUNAS")
    print("=" * 60)

    schema = parquet.schema_arrow

    for index, field in enumerate(schema):
        print(f"{index}. {field.name} | Tipo: {field.type} | Nullable: {field.nullable}")

    print("\n" + "=" * 60)
    print("AMOSTRA DOS DADOS (5 REGISTROS)")
    print("=" * 60)

    try:
        batch = next(parquet.iter_batches(batch_size=5))
        table = batch.to_pydict()

        for column, values in table.items():
            print(f"\nCOLUNA: {column}")

            for index, value in enumerate(values):
                representation = repr(value)

                if len(representation) > 300:
                    representation = representation[:300] + "..."

                print(f"  [{index}] {representation}")

    except StopIteration:
        print("Dataset vazio.")
    except Exception as error:
        print(f"[ERRO] Falha ao ler amostra: {error}")

    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 60)


if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else "dataset.parquet"
    inspect_parquet(file)
