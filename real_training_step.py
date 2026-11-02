
import torch

from src.data.dataset import create_dataloader
from src.model.ayris import AYRISModel
from src.training.step import train_step


torch.manual_seed(42)

print("Carregando dataset...")

loader = create_dataloader(
    data_dir="data/tokenized",
    split="train",
    batch_size=1,
    block_size=512,
    stride=512,
    shuffle=False,
)

input_ids, target_ids = next(iter(loader))

print("Carregando AYRIS...")

model = AYRISModel(
    dropout=0.0,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

print("Executando primeira atualização...")

result = train_step(
    model=model,
    optimizer=optimizer,
    input_ids=input_ids,
    target_ids=target_ids,
    max_grad_norm=1.0,
)

print("\n========== RESULTADO ==========")
print("Loss:", result["loss"])
print("Gradiente:", result["grad_norm"])
print("Learning rate:", result["learning_rate"])
print("Parâmetros:", f"{model.count_parameters():,}")
print("Atualização: CONCLUÍDA")
