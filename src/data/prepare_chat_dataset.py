
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/chat/conversations.txt"
OUTPUT = ROOT / "data/chat/tokenized"
TOKENIZER_PATH = ROOT / "checkpoints/tokenizer/tokenizer.json"

VOCAB_SIZE = 4096
CONTEXT_LENGTH = 512
SEED = 42

SPECIAL_IDS = {
    "<BOS>": 2,
    "<EOS>": 3,
    "<SYS>": 4,
    "<USR>": 5,
    "<AST>": 6,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_conversations():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    text = SOURCE.read_text(encoding="utf-8").strip()

    pattern = re.compile(r"<BOS>.*?<EOS>", re.DOTALL)
    conversations = pattern.findall(text)

    if not conversations:
        raise ValueError("Nenhuma conversa encontrada.")

    remaining = pattern.sub("", text).strip()
    if remaining:
        raise ValueError(
            "Existe conteúdo fora dos blocos BOS/EOS. "
            "Revise conversations.txt."
        )

    for number, conversation in enumerate(conversations, start=1):
        if conversation.count("<BOS>") != 1:
            raise ValueError(f"Conversa {number}: BOS inválido.")

        if conversation.count("<EOS>") != 1:
            raise ValueError(f"Conversa {number}: EOS inválido.")

        if conversation.count("<SYS>") != 1:
            raise ValueError(f"Conversa {number}: SYS inválido.")

        if conversation.count("<USR>") != conversation.count("<AST>"):
            raise ValueError(
                f"Conversa {number}: quantidade de USR e AST diferente."
            )

        if conversation.count("<USR>") == 0:
            raise ValueError(f"Conversa {number}: sem mensagens.")

    return conversations


def split_conversations(conversations):
    shuffled = list(conversations)
    random.Random(SEED).shuffle(shuffled)

    total = len(shuffled)

    validation_count = max(1, round(total * 0.10))
    test_count = max(1, round(total * 0.10))
    train_count = total - validation_count - test_count

    if train_count <= 0:
        raise ValueError("Poucas conversas para criar os splits.")

    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count:train_count + validation_count],
        "test": shuffled[train_count + validation_count:],
    }


def build_tokenizer():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(TOKENIZER_PATH)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    for marker, expected_id in SPECIAL_IDS.items():
        actual_ids = tokenizer.encode(
            marker,
            add_special_tokens=False,
        ).ids

        if actual_ids != [expected_id]:
            raise ValueError(
                f"Marcador {marker}: esperado [{expected_id}], "
                f"recebido {actual_ids}."
            )

    return tokenizer


def encode_conversation(conversation, tokenizer):
    ids = tokenizer.encode(
        conversation,
        add_special_tokens=False,
    ).ids

    if not ids:
        raise ValueError("Conversa produziu zero tokens.")

    if ids[0] != SPECIAL_IDS["<BOS>"]:
        raise ValueError("Conversa não começa com BOS.")

    if ids[-1] != SPECIAL_IDS["<EOS>"]:
        raise ValueError("Conversa não termina com EOS.")

    mask = []
    assistant_turn = False

    for token_id in ids:
        if token_id in (
            SPECIAL_IDS["<BOS>"],
            SPECIAL_IDS["<SYS>"],
            SPECIAL_IDS["<USR>"],
        ):
            assistant_turn = False
            mask.append(0)

        elif token_id == SPECIAL_IDS["<AST>"]:
            assistant_turn = True
            mask.append(0)

        elif token_id == SPECIAL_IDS["<EOS>"]:
            mask.append(1 if assistant_turn else 0)
            assistant_turn = False

        else:
            mask.append(1 if assistant_turn else 0)

    if len(ids) != len(mask):
        raise RuntimeError("Tokens e máscara possuem tamanhos diferentes.")

    if sum(mask) == 0:
        raise ValueError("Conversa não possui tokens de resposta.")

    if max(ids) >= VOCAB_SIZE:
        raise ValueError("Token fora do vocabulário.")

    return ids, mask



def save_split(name, conversations, tokenizer):
    all_tokens = []
    all_mask = []
    records = []
    offset = 0

    for conversation_number, conversation in enumerate(
        conversations, start=1
    ):
        ids, mask = encode_conversation(conversation, tokenizer)

        records.append({
            "offset": offset,
            "length": len(ids),
            "conversation_id": conversation_number,
        })

        all_tokens.extend(ids)
        all_mask.extend(mask)
        offset += len(ids)

    if not all_tokens:
        raise ValueError(f"Split {name} está vazio.")

    if sum(all_mask) == 0:
        raise ValueError(f"Split {name} não possui respostas.")

    token_path = OUTPUT / f"{name}.bin"
    mask_path = OUTPUT / f"{name}.mask.bin"
    index_path = OUTPUT / f"{name}.index.jsonl"

    np.asarray(all_tokens, dtype="<u2").tofile(token_path)
    np.asarray(all_mask, dtype=np.uint8).tofile(mask_path)

    with index_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"\n{name.upper()}")
    print(f"Conversas: {len(conversations)}")
    print(f"Tokens: {len(all_tokens):,}")
    print(f"Tokens de resposta: {sum(all_mask):,}")
    print(f"Fragmentos individuais: {len(records)}")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    conversations = load_conversations()
    tokenizer = build_tokenizer()

    splits = split_conversations(conversations)

    print("=" * 55)
    print("PREPARAÇÃO DO DATASET CONVERSACIONAL — AYRIS")
    print("=" * 55)
    print(f"Conversas encontradas: {len(conversations)}")

    for name, split in splits.items():
        save_split(name, split, tokenizer)

    manifest = {
        "dataset": "AYRIS conversational fine-tuning",
        "format_version": 1,
        "vocab_size": VOCAB_SIZE,
        "context_length": CONTEXT_LENGTH,
        "seed": SEED,
        "tokenizer_sha256": sha256_file(TOKENIZER_PATH),
        "source_sha256": sha256_file(SOURCE),
        "mask_policy": {
            "description": "Loss somente nos tokens gerados pela AYRIS.",
            "included": ["assistant_response", "assistant_eos"],
            "excluded": ["system", "user", "assistant_marker", "bos"],
        },
        "splits": {
            name: {
                "conversations": len(items),
            }
            for name, items in splits.items()
        },
    }

    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("DATASET PREPARADO")
    print(f"Manifest: {manifest_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
