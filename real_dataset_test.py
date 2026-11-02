
import torch
import torch.nn.functional as F

from src.data.dataset import create_dataloader
from src.model.ayris import AYRISModel


# 1. Carregar um lote real.
loader = create_dataloader(
    data_dir="data/tokenized",
    split="train",
    batch_size=1,
    block_size=512,
    stride=512,
    shuffle=False,
)

print("Janelas disponíveis:", len(loader.dataset))

x, y = next(iter(loader))

print("\n=== DATASET ===")
print("Entrada:", x.shape)
print("Alvo:", y.shape)
print("Tipo:", x.dtype)

# 2. Validar deslocamento dos alvos.
assert torch.equal(x[:, 1:], y[:, :-1])

print("Deslocamento dos tokens: OK")

# 3. Validar IDs.
assert x.min() >= 0
assert y.min() >= 0
assert x.max() < 4096
assert y.max() < 4096

print("IDs dentro do vocabulário: OK")

# 4. Carregar a arquitetura completa.
model = AYRISModel()
model.eval()

print("\n=== MODELO ===")
print("Parâmetros:", f"{model.count_parameters():,}")

# 5. Executar inferência.
with torch.inference_mode():
    logits = model(x)

print("\n=== SAÍDA ===")
print("Logits:", logits.shape)
print("Tipo:", logits.dtype)

# 6. Calcular a primeira loss.
loss = F.cross_entropy(
    logits.reshape(-1, 4096),
    y.reshape(-1),
)

print("Cross Entropy:", loss.item())

assert torch.isfinite(loss).item()

print("\nINTEGRAÇÃO: APROVADA")
