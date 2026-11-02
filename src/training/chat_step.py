
import torch
import torch.nn.functional as F


def masked_cross_entropy(logits, target_ids, loss_mask):
    """
    Calcula a Cross Entropy somente nos tokens selecionados pela máscara.
    """
    if logits.ndim != 3:
        raise ValueError("logits deve ter formato [batch, sequence, vocab].")

    if target_ids.ndim != 2 or loss_mask.ndim != 2:
        raise ValueError("Targets e máscara devem ter formato [batch, sequence].")

    if logits.shape[:2] != target_ids.shape:
        raise ValueError("Dimensões de logits e targets incompatíveis.")

    if target_ids.shape != loss_mask.shape:
        raise ValueError("Targets e máscara possuem dimensões diferentes.")

    if loss_mask.dtype != torch.bool:
        raise TypeError("loss_mask deve usar torch.bool.")

    selected_logits = logits[loss_mask]
    selected_targets = target_ids[loss_mask]

    if selected_targets.numel() == 0:
        raise ValueError("A máscara não selecionou nenhum token.")

    loss = F.cross_entropy(
        selected_logits,
        selected_targets,
        reduction="mean",
    )

    if not torch.isfinite(loss).item():
        raise FloatingPointError("Loss mascarada não finita.")

    return loss, selected_targets.numel()


def train_chat_step(
    model,
    optimizer,
    input_ids,
    target_ids,
    loss_mask,
    max_grad_norm=1.0,
):
    """
    Executa uma atualização usando somente os alvos selecionados
    por loss_mask.
    """
    if input_ids.ndim != 2 or target_ids.ndim != 2:
        raise ValueError("Entradas devem ter formato [batch, sequence].")

    if input_ids.shape != target_ids.shape:
        raise ValueError("Entrada e alvo devem ter dimensões iguais.")

    if input_ids.shape != loss_mask.shape:
        raise ValueError("Entrada e máscara devem ter dimensões iguais.")

    if input_ids.dtype != torch.long or target_ids.dtype != torch.long:
        raise TypeError("Tokens devem usar torch.long.")

    if loss_mask.dtype != torch.bool:
        raise TypeError("loss_mask deve usar torch.bool.")

    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("Batch e sequência não podem ser vazios.")

    if input_ids.shape[1] > model.max_seq_len:
        raise ValueError("Sequência excede o contexto máximo.")

    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm deve ser positivo.")

    model_device = next(model.parameters()).device

    if (
        input_ids.device != model_device
        or target_ids.device != model_device
        or loss_mask.device != model_device
    ):
        raise ValueError("Dados e modelo devem estar no mesmo dispositivo.")

    if input_ids.min().item() < 0 or input_ids.max().item() >= model.vocab_size:
        raise ValueError("ID de entrada fora do vocabulário.")

    if target_ids.min().item() < 0 or target_ids.max().item() >= model.vocab_size:
        raise ValueError("ID de alvo fora do vocabulário.")

    if torch.any(input_ids == model.padding_idx).item():
        raise ValueError("PAD não é permitido nas entradas.")

    if not loss_mask.any().item():
        raise ValueError("Batch sem tokens de resposta selecionados.")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)

    loss, response_tokens = masked_cross_entropy(
        logits,
        target_ids,
        loss_mask,
    )

    loss.backward()

    grad_norm_before = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm,
        error_if_nonfinite=True,
    ).item()

    clip_coefficient = min(
        1.0,
        max_grad_norm / (grad_norm_before + 1e-6),
    )
    grad_norm_after = grad_norm_before * clip_coefficient

    optimizer.step()

    return {
        "loss": float(loss.item()),
        "response_tokens": int(response_tokens),
        "grad_norm_before_clip": float(grad_norm_before),
        "grad_norm_after_clip": float(grad_norm_after),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
