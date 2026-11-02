import torch
import torch.nn.functional as F


def train_step(
    model,
    optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    max_grad_norm: float = 1.0,
):
    """Executa forward, Cross Entropy, backpropagation e AdamW."""
    if input_ids.ndim != 2 or target_ids.ndim != 2:
        raise ValueError("Entradas devem ter formato [batch, sequence].")

    if input_ids.shape != target_ids.shape:
        raise ValueError("Entrada e alvo devem ter dimensões iguais.")

    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("Batch e sequência não podem ser vazios.")

    if input_ids.dtype != torch.long or target_ids.dtype != torch.long:
        raise TypeError("Entrada e alvo devem usar torch.long.")

    model_device = next(model.parameters()).device
    if input_ids.device != model_device or target_ids.device != model_device:
        raise ValueError("Dados e modelo devem estar no mesmo dispositivo.")

    if input_ids.shape[1] > model.max_seq_len:
        raise ValueError("Sequência excede o contexto máximo.")

    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm deve ser positivo.")

    if input_ids.min().item() < 0 or input_ids.max().item() >= model.vocab_size:
        raise ValueError("ID de entrada fora do vocabulário.")

    if target_ids.min().item() < 0 or target_ids.max().item() >= model.vocab_size:
        raise ValueError("ID de alvo fora do vocabulário.")

    if torch.any(input_ids == model.padding_idx).item():
        raise ValueError("PAD não é permitido nas entradas deste dataset.")

    if torch.any(target_ids == model.padding_idx).item():
        raise ValueError("PAD não é permitido nos alvos.")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)
    loss = F.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        target_ids.reshape(-1),
    )

    if not torch.isfinite(loss).item():
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Loss não finita.")

    loss.backward()

    # clip_grad_norm_ retorna a norma ANTES do clipping.
    grad_norm_before = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm,
        error_if_nonfinite=True,
    ).item()

    # Coeficiente efetivamente aplicado pelo clip_grad_norm_.
    clip_coefficient = min(
        1.0,
        max_grad_norm / (grad_norm_before + 1e-6),
    )
    grad_norm_after = grad_norm_before * clip_coefficient

    optimizer.step()

    return {
        "loss": float(loss.item()),
        # Alias mantido para compatibilidade com os testes e scripts anteriores.
        "grad_norm": float(grad_norm_before),
        "grad_norm_before_clip": float(grad_norm_before),
        "grad_norm_after_clip": float(grad_norm_after),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
