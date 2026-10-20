"""
Sparse-to-Sparse (S2S) knowledge distillation losses of LiteDoc (paper Sec. 3.3).

    L_hidden = 1/M * sum_i MSE(H_S^i(x), H_T^i(x))              (Eq. 6-7, layer-wise alignment)
    L_KL     = 1/n * sum_j sum_k p_S log(p_S / p_T)            (Eq. 8, output logit alignment)
    L        = alpha * L_KL + (1 - alpha) * L_hidden            (alpha = 0.5 in the paper)

The student keeps the teacher's depth and hidden size, only the number of routed experts per
layer differs, so hidden states can be compared layer by layer without any projection.
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def logit_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    direction: str = "reverse",
) -> torch.Tensor:
    """
    Token-averaged KL divergence between next-token distributions.

    Args:
        student_logits, teacher_logits: [batch, seq, vocab].
        mask: [batch, seq] bool, positions that contribute to the loss.
        temperature: softmax temperature (loss is scaled by T^2 as in Hinton et al.).
        direction: "reverse" = KL(p_S || p_T), exactly Eq. (8) of the paper;
                   "forward" = KL(p_T || p_S), the classic Hinton formulation.
    """
    s_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_logp = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    if direction == "reverse":
        per_token = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)
    elif direction == "forward":
        per_token = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
    else:
        raise ValueError(f"unknown KL direction '{direction}'")
    # KL >= 0; clamp float round-off over large vocabularies
    per_token = per_token.clamp_min(0.0)
    return masked_mean(per_token, mask) * (temperature ** 2)


def hidden_mse_loss(
    student_hidden: Sequence[torch.Tensor],
    teacher_hidden: Sequence[torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    layers: Optional[Sequence[int]] = None,
    normalize: str = "none",
) -> torch.Tensor:
    """
    Eq. (6)-(7): mean over the M aligned layers of the element-wise MSE (averaged over the d
    hidden units and the n valid tokens).

    Args:
        student_hidden, teacher_hidden: tuples of [batch, seq, d] tensors (one per layer).
        mask: [batch, seq] bool, tokens included in the average (e.g. the attention mask).
        layers: indices into the tuples to align (default: all).
        normalize: "none" (paper) or "rms" - divide both states by the teacher's per-token RMS,
            which keeps the loss scale comparable across layers with large residual activations.
    """
    if len(student_hidden) != len(teacher_hidden):
        raise ValueError(f"student has {len(student_hidden)} hidden states but teacher has {len(teacher_hidden)}")
    indices = list(range(len(student_hidden))) if layers is None else list(layers)
    if not indices:
        raise ValueError("no layers selected for hidden-state alignment")
    losses = []
    for i in indices:
        hs = student_hidden[i].float()
        ht = teacher_hidden[i].float().to(hs.device)
        if normalize == "rms":
            scale = ht.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            hs, ht = hs / scale, ht / scale
        elif normalize != "none":
            raise ValueError(f"unknown hidden normalization '{normalize}'")
        per_token = (hs - ht).pow(2).mean(dim=-1)
        losses.append(masked_mean(per_token, mask))
    return torch.stack(losses).mean()


def s2s_distillation_loss(
    student_outputs,
    teacher_outputs,
    logit_mask: Optional[torch.Tensor],
    hidden_mask: Optional[torch.Tensor],
    alpha: float = 0.5,
    temperature: float = 1.0,
    kl_direction: str = "reverse",
    hidden_layers: Optional[Sequence[int]] = None,
    hidden_normalize: str = "none",
    objective: str = "combined",
) -> Dict[str, torch.Tensor]:
    """
    Combined S2S objective. `objective` selects the ablation variants of Sec. 5.3:
    "combined" (alpha * KL + (1 - alpha) * MSE), "output" (KL only) or "layer" (MSE only).
    """
    device = student_outputs.logits.device
    zero = torch.zeros((), device=device)
    kl = zero
    mse = zero
    if objective in ("combined", "output"):
        kl = logit_kl_loss(student_outputs.logits, teacher_outputs.logits.to(device),
                           logit_mask, temperature, kl_direction)
    if objective in ("combined", "layer"):
        # hidden_states[0] is the (shared, frozen) input embedding; align the decoder layer outputs
        s_hidden = student_outputs.hidden_states[1:]
        t_hidden = teacher_outputs.hidden_states[1:]
        mse = hidden_mse_loss(s_hidden, t_hidden, hidden_mask, hidden_layers, hidden_normalize)

    if objective == "combined":
        loss = alpha * kl + (1.0 - alpha) * mse
    elif objective == "output":
        loss = kl
    elif objective == "layer":
        loss = mse
    else:
        raise ValueError(f"unknown objective '{objective}'")
    return {"loss": loss, "kl": kl.detach(), "hidden_mse": mse.detach()}


def answer_token_mask(prompt_lengths: Sequence[int], attention_mask: torch.Tensor) -> torch.Tensor:
    """[batch, seq] bool mask of the assistant-answer tokens of right-padded sequences."""
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)[None, :]
    starts = torch.as_tensor(list(prompt_lengths), device=attention_mask.device)[:, None]
    return (positions >= starts) & attention_mask.bool()


def build_loss_masks(attention_mask: torch.Tensor, images_seq_mask: torch.Tensor,
                     kd_tokens: str = "text", answer_mask: Optional[torch.Tensor] = None):
    """
    Masks for the two losses.

      * logit mask: positions whose next-token distribution is distilled. "text" = every non-image,
        non-padding position (the paper's sum over the sequence); "answer" = only positions that
        predict an answer token (needs `answer_mask`, see `answer_token_mask`).
      * hidden mask: every non-padding token (image tokens included).
    """
    attention_mask = attention_mask.bool()
    hidden_mask = attention_mask
    if kd_tokens == "answer":
        if answer_mask is None:
            raise ValueError("kd_tokens='answer' needs answer_mask")
        logit_mask = torch.zeros_like(attention_mask)
        # logits at position t predict token t + 1
        logit_mask[:, :-1] = answer_mask[:, 1:].to(attention_mask.device)
    elif kd_tokens == "text":
        logit_mask = attention_mask & ~images_seq_mask.bool()
    else:
        raise ValueError(f"unknown kd_tokens '{kd_tokens}'")
    return logit_mask, hidden_mask
