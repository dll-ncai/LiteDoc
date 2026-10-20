"""
Helpers to inspect, profile and structurally prune the routed experts of the
DeepSeek-VL2 MoE language model.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from deepseek_vl2.models.modeling_deepseek import DeepseekV2MoE


def get_language_model(model: nn.Module) -> nn.Module:
    """Return the DeepseekV2ForCausalLM inside a DeepseekVLV2ForCausalLM (or the model itself)."""
    return getattr(model, "language", model)


def get_language_config(model: nn.Module):
    return get_language_model(model).config


def iter_moe_layers(model: nn.Module) -> List[Tuple[int, DeepseekV2MoE]]:
    """All (decoder_layer_idx, DeepseekV2MoE) pairs of the language model, in depth order."""
    language = get_language_model(model)
    decoder = language.model if hasattr(language, "model") else language
    layers = []
    for layer_idx, layer in enumerate(decoder.layers):
        if isinstance(layer.mlp, DeepseekV2MoE):
            layers.append((layer_idx, layer.mlp))
    return layers


def num_decoder_layers(model: nn.Module) -> int:
    language = get_language_model(model)
    decoder = language.model if hasattr(language, "model") else language
    return len(decoder.layers)


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not only_trainable)


def count_active_parameters(model: nn.Module) -> int:
    """
    Parameters touched per token: everything except the routed experts, plus top-k routed experts
    per MoE layer (all experts of a layer have the same size).
    """
    total = count_parameters(model)
    for _, moe in iter_moe_layers(model):
        experts = [e for e in moe.experts if e is not None]
        if not experts:
            continue
        per_expert = sum(p.numel() for p in experts[0].parameters())
        total -= per_expert * len(experts)
        total += per_expert * min(moe.num_experts_per_tok, len(experts))
    return total


def experts_per_layer(model: nn.Module) -> Dict[int, int]:
    return {layer_idx: len(moe.experts) for layer_idx, moe in iter_moe_layers(model)}


# ---------------------------------------------------------------------------
# Calibration statistics
# ---------------------------------------------------------------------------

class ExpertStatsCollector:
    """
    Registers forward hooks on every selected MoE layer to collect, while the teacher
    runs over document-understanding calibration data:

      * routing statistics: how many tokens each routed expert receives (top-k counts) and the
        total gating weight it receives (Strategy 1 of the paper, also used as the activation
        criterion that decides which expert of a redundant pair is dropped);
      * a uniform random sample (reservoir sampling) of the MoE layer *inputs*, so that every
        expert can afterwards be evaluated on exactly the same tokens for CKA (Strategy 3).

    Only tokens with `token_mask == True` (set via `set_token_mask`) are used.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: Optional[Sequence[int]] = None,
        max_tokens_per_layer: int = 2048,
        collect_inputs: bool = True,
        seed: int = 42,
    ):
        self.max_tokens = max_tokens_per_layer
        self.collect_inputs = collect_inputs
        self.generator = torch.Generator().manual_seed(seed)
        self.layers = [
            (idx, moe) for idx, moe in iter_moe_layers(model)
            if layer_indices is None or idx in set(layer_indices)
        ]
        self.token_counts: Dict[int, torch.Tensor] = {}
        self.gate_mass: Dict[int, torch.Tensor] = {}
        self.inputs: Dict[int, Optional[torch.Tensor]] = {}
        self.seen_tokens: Dict[int, int] = {}
        self._handles = []
        self._token_mask: Optional[torch.Tensor] = None

        for idx, moe in self.layers:
            num_experts = len(moe.experts)
            self.token_counts[idx] = torch.zeros(num_experts, dtype=torch.long)
            self.gate_mass[idx] = torch.zeros(num_experts, dtype=torch.float64)
            self.inputs[idx] = None
            self.seen_tokens[idx] = 0
            self._handles.append(moe.register_forward_pre_hook(self._make_input_hook(idx)))
            self._handles.append(moe.gate.register_forward_hook(self._make_gate_hook(idx)))

    def set_token_mask(self, mask: Optional[torch.Tensor]):
        """mask: [batch, seq_len] bool tensor of tokens to record (e.g. attention_mask), or None for all."""
        self._token_mask = mask

    def _flat_mask(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        if self._token_mask is None:
            return None
        mask = self._token_mask
        if mask.shape != hidden_states.shape[:2]:
            # e.g. during incremental decoding; fall back to using every token
            return None
        return mask.reshape(-1).bool().to(hidden_states.device)

    def _make_input_hook(self, idx: int):
        def hook(module, args):
            if not self.collect_inputs:
                return
            hidden_states = args[0]
            flat = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
            mask = self._flat_mask(hidden_states)
            if mask is not None:
                flat = flat[mask]
            self._reservoir_add(idx, flat.float().cpu())
        return hook

    def _make_gate_hook(self, idx: int):
        def hook(module, args, output):
            topk_idx, topk_weight = output[0], output[1]
            hidden_states = args[0]
            mask = self._flat_mask(hidden_states)
            if mask is not None:
                topk_idx = topk_idx[mask]
                topk_weight = topk_weight[mask]
            num_experts = self.token_counts[idx].numel()
            flat_idx = topk_idx.reshape(-1).long().cpu()
            self.token_counts[idx] += torch.bincount(flat_idx, minlength=num_experts)
            self.gate_mass[idx] += torch.bincount(
                flat_idx, weights=topk_weight.reshape(-1).double().cpu(), minlength=num_experts
            )
        return hook

    def _reservoir_add(self, idx: int, tokens: torch.Tensor):
        """Algorithm R (vectorized per batch): keep a uniform sample of at most max_tokens rows."""
        if tokens.shape[0] == 0:
            return
        buf = self.inputs[idx]
        seen = self.seen_tokens[idx]
        if buf is None:
            take = min(self.max_tokens, tokens.shape[0])
            self.inputs[idx] = tokens[:take].clone()
            rest = tokens[take:]
            seen += take
            buf = self.inputs[idx]
        else:
            rest = tokens
            free = self.max_tokens - buf.shape[0]
            if free > 0:
                buf = torch.cat([buf, rest[:free]], dim=0)
                self.inputs[idx] = buf
                seen += min(free, rest.shape[0])
                rest = rest[free:]

        if rest.shape[0] > 0:
            positions = torch.arange(seen + 1, seen + rest.shape[0] + 1, dtype=torch.float64)
            draws = (torch.rand(rest.shape[0], generator=self.generator, dtype=torch.float64) * positions).long()
            replace = draws < self.max_tokens
            for row, slot in zip(rest[replace], draws[replace].tolist()):
                buf[slot] = row
            seen += rest.shape[0]
        self.seen_tokens[idx] = seen

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def routing_frequency(self, idx: int) -> torch.Tensor:
        counts = self.token_counts[idx].double()
        total = counts.sum()
        return counts / total if total > 0 else counts


@torch.no_grad()
def compute_expert_outputs(
    moe: DeepseekV2MoE,
    inputs: torch.Tensor,
    device: Optional[torch.device] = None,
    batch_tokens: int = 1024,
) -> torch.Tensor:
    """
    Run *every* routed expert of `moe` densely on the same calibration tokens.

    Args:
        moe: the MoE layer.
        inputs: [n, d] MoE-layer inputs collected by ExpertStatsCollector.

    Returns:
        [E, n, d] float32 tensor of expert outputs (on `device`).
    """
    first = next(e for e in moe.experts if e is not None)
    param = next(first.parameters())
    device = device or param.device
    outputs = []
    for expert in moe.experts:
        chunks = []
        for start in range(0, inputs.shape[0], batch_tokens):
            x = inputs[start:start + batch_tokens].to(device=param.device, dtype=param.dtype)
            chunks.append(expert(x).float().to(device))
        outputs.append(torch.cat(chunks, dim=0))
    return torch.stack(outputs, dim=0)


# ---------------------------------------------------------------------------
# Structural pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def prune_moe_layer(moe: DeepseekV2MoE, keep: Sequence[int]):
    """
    Physically remove routed experts from one MoE layer, keeping `keep` (indices into the
    layer's *current* experts, in the order they should be stored). Router rows are sliced
    accordingly, so the remaining experts keep exactly the routing logits they had before.
    """
    keep = sorted(int(i) for i in keep)
    num_experts = len(moe.experts)
    if len(set(keep)) != len(keep) or any(i < 0 or i >= num_experts for i in keep):
        raise ValueError(f"invalid expert indices to keep: {keep}")
    if moe.ep_size > 1:
        raise NotImplementedError("pruning an expert-parallel (ep_size > 1) model is not supported")
    if len(keep) < moe.num_experts_per_tok:
        raise ValueError(
            f"cannot keep {len(keep)} experts when the router selects top-{moe.num_experts_per_tok}"
        )

    moe.experts = nn.ModuleList([moe.experts[i] for i in keep])
    moe.experts_per_rank = len(keep)
    moe.n_routed_experts = len(keep)

    gate = moe.gate
    index = torch.tensor(keep, dtype=torch.long, device=gate.weight.device)
    gate.weight = nn.Parameter(gate.weight.data.index_select(0, index).clone(), requires_grad=gate.weight.requires_grad)
    if hasattr(gate, "e_score_correction_bias"):
        bias = gate.e_score_correction_bias
        gate.e_score_correction_bias = nn.Parameter(
            bias.data.index_select(0, index.to(bias.device)).clone(), requires_grad=bias.requires_grad
        )
    gate.n_routed_experts = len(keep)
    if len(keep) != gate.config.n_routed_experts:
        # same rule as MoEGate.__init__ so that a reloaded checkpoint routes identically
        gate.n_group = 1
        gate.topk_group = 1


def apply_expert_pruning(model: nn.Module, keep_map: Dict[int, Sequence[int]]):
    """
    Prune several MoE layers and record the result in the language config so that
    `save_pretrained` / `from_pretrained` round-trips the pruned architecture.

    Args:
        keep_map: {decoder_layer_idx: indices (in the current model) of the experts to keep}.
    """
    config = get_language_config(model)
    num_experts = dict(config.moe_layer_num_experts or {})
    kept_original = dict(config.moe_layer_kept_experts or {})
    layers = dict(iter_moe_layers(model))

    for layer_idx, keep in keep_map.items():
        if layer_idx not in layers:
            raise KeyError(f"decoder layer {layer_idx} is not an MoE layer")
        keep = sorted(int(i) for i in keep)
        # map current positions back to indices of the original (unpruned) model
        previous = kept_original.get(str(layer_idx), list(range(len(layers[layer_idx].experts))))
        prune_moe_layer(layers[layer_idx], keep)
        kept_original[str(layer_idx)] = [int(previous[i]) for i in keep]
        num_experts[str(layer_idx)] = len(keep)

    config.moe_layer_num_experts = num_experts
    config.moe_layer_kept_experts = kept_original
    return model
