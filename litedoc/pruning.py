"""
Expert-selection criteria for LiteDoc Stage 1 (paper Sec. 3.2 and the ablation of Sec. 5.3).

  * "cka"              (adopted) - build a redundancy graph whose edges are expert pairs with
                        CKA > tau and drop one expert of every highly correlated pair; inside a pair
                        the expert with the lower routing activation on DU data is removed.
  * "frequency"        (Strategy 1 / ablation) - drop the experts that receive the fewest tokens.
  * "weight_cosine"    (Strategy 2) - same redundancy graph, but edges come from cosine similarity
                        of the flattened expert weights.
  * "weight_magnitude" (ablation) - drop the experts with the smallest L2 weight norm.

All selection functions return, for every layer, the sorted list of experts to KEEP.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from deepseek_vl2.models.modeling_deepseek import DeepseekV2MoE

STRATEGIES = ("cka", "frequency", "weight_cosine", "weight_magnitude")


def flatten_expert_weights(moe: DeepseekV2MoE) -> torch.Tensor:
    """[E, P] matrix, one row per routed expert (gate_proj | up_proj | down_proj weights)."""
    rows = []
    for expert in moe.experts:
        rows.append(torch.cat([
            expert.gate_proj.weight.detach().float().reshape(-1).cpu(),
            expert.up_proj.weight.detach().float().reshape(-1).cpu(),
            expert.down_proj.weight.detach().float().reshape(-1).cpu(),
        ]))
    return torch.stack(rows, dim=0)


def weight_cosine_similarity(moe: DeepseekV2MoE) -> torch.Tensor:
    """Eq. (2): sim(W_i, W_j) = <W_i, W_j> / (||W_i|| ||W_j||)."""
    weights = flatten_expert_weights(moe)
    weights = weights / weights.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return weights @ weights.T


def weight_magnitude(moe: DeepseekV2MoE) -> torch.Tensor:
    return flatten_expert_weights(moe).norm(dim=1)


def num_to_prune(num_experts: int, prune_ratio: Optional[float], min_keep: int) -> int:
    """Upper bound of experts removed from one layer."""
    limit = max(0, num_experts - min_keep)
    if prune_ratio is None:
        return limit
    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError(f"prune_ratio must be in [0, 1), got {prune_ratio}")
    return min(limit, int(round(num_experts * prune_ratio)))


def protected_experts(activation: torch.Tensor, protect_top_fraction: float) -> set:
    """Activation-based thresholding: the most-used experts are never pruned."""
    if protect_top_fraction <= 0:
        return set()
    k = int(math.ceil(protect_top_fraction * activation.numel()))
    return set(torch.argsort(activation, descending=True)[:k].tolist())


def _sorted_pairs(similarity: torch.Tensor) -> List[Tuple[float, int, int]]:
    n = similarity.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    values = similarity[iu[0], iu[1]]
    order = torch.argsort(values, descending=True)
    return [(float(values[o]), int(iu[0][o]), int(iu[1][o])) for o in order]


def select_by_redundancy(
    similarity_per_layer: Dict[int, torch.Tensor],
    activation_per_layer: Dict[int, torch.Tensor],
    threshold: Optional[float],
    prune_ratio: Optional[float],
    min_keep: int,
    budget: str = "per_layer",
    protect_top_fraction: float = 0.0,
) -> Dict[int, List[int]]:
    """
    Greedy pruning on the redundancy graph (edges = expert pairs with similarity > threshold).

    Pairs are visited from the most to the least similar. While both experts of a pair are
    still alive, the one with the lower activation (routing frequency) is removed, so exactly
    one expert of each highly correlated pair is pruned.

    Args:
        similarity_per_layer: {layer: [E, E] similarity (CKA or weight cosine)}.
        activation_per_layer: {layer: [E] routing frequency}; the less active expert of a pair is pruned.
        threshold: tau. Only pairs with similarity > tau are pruned (None = no threshold).
        prune_ratio: fraction of experts to remove (None = remove every expert covered by tau).
            With `budget="per_layer"` the ratio applies to each layer independently (uniform pruning);
            with `budget="global"` it applies to the total number of experts and the redundancy
            graph of all layers is processed jointly, which yields a layer-dependent retention.
        min_keep: never keep fewer experts than this in a layer (>= router top-k).
        protect_top_fraction: experts in the top fraction by activation are never pruned.
    """
    if threshold is None and prune_ratio is None:
        raise ValueError("set at least one of threshold (tau) or prune_ratio")
    if budget not in ("per_layer", "global"):
        raise ValueError(f"unknown budget '{budget}'")

    alive = {layer: set(range(sim.shape[0])) for layer, sim in similarity_per_layer.items()}
    protected = {
        layer: protected_experts(activation_per_layer[layer], protect_top_fraction)
        for layer in similarity_per_layer
    }
    layer_cap = {
        layer: num_to_prune(sim.shape[0], prune_ratio if budget == "per_layer" else None, min_keep)
        for layer, sim in similarity_per_layer.items()
    }
    pruned = {layer: 0 for layer in similarity_per_layer}

    if budget == "global":
        total = sum(sim.shape[0] for sim in similarity_per_layer.values())
        global_cap = total if prune_ratio is None else int(round(total * prune_ratio))
        pairs = [
            (score, layer, i, j)
            for layer, sim in similarity_per_layer.items()
            for score, i, j in _sorted_pairs(sim)
        ]
        pairs.sort(key=lambda p: p[0], reverse=True)
    else:
        global_cap = None
        pairs = [
            (score, layer, i, j)
            for layer, sim in similarity_per_layer.items()
            for score, i, j in _sorted_pairs(sim)
        ]

    total_pruned = 0
    for score, layer, i, j in pairs:
        if threshold is not None and score <= threshold:
            if budget == "global":
                break
            continue
        if global_cap is not None and total_pruned >= global_cap:
            break
        if pruned[layer] >= layer_cap[layer]:
            continue
        if i not in alive[layer] or j not in alive[layer]:
            continue
        activation = activation_per_layer[layer]
        candidates = sorted((i, j), key=lambda e: (float(activation[e]), e))
        victim = next((e for e in candidates if e not in protected[layer]), None)
        if victim is None:
            continue
        alive[layer].discard(victim)
        pruned[layer] += 1
        total_pruned += 1

    return {layer: sorted(keep) for layer, keep in alive.items()}


def select_by_score(
    score_per_layer: Dict[int, torch.Tensor],
    prune_ratio: float,
    min_keep: int,
    budget: str = "per_layer",
) -> Dict[int, List[int]]:
    """Remove the experts with the LOWEST score (routing frequency or weight norm)."""
    if prune_ratio is None:
        raise ValueError("score-based strategies need --prune_ratio")
    keep = {}
    if budget == "per_layer":
        for layer, score in score_per_layer.items():
            k = num_to_prune(score.numel(), prune_ratio, min_keep)
            drop = set(torch.argsort(score)[:k].tolist())
            keep[layer] = [e for e in range(score.numel()) if e not in drop]
        return keep

    if budget != "global":
        raise ValueError(f"unknown budget '{budget}'")
    total = sum(s.numel() for s in score_per_layer.values())
    global_cap = int(round(total * prune_ratio))
    # normalize per layer so that scores are comparable across depth
    candidates = []
    for layer, score in score_per_layer.items():
        s = score.double()
        s = s / s.sum().clamp_min(1e-12)
        candidates.extend((float(s[e]), layer, e) for e in range(s.numel()))
    candidates.sort(key=lambda c: c[0])
    alive = {layer: set(range(s.numel())) for layer, s in score_per_layer.items()}
    pruned = 0
    for _, layer, e in candidates:
        if pruned >= global_cap:
            break
        if len(alive[layer]) <= min_keep:
            continue
        alive[layer].discard(e)
        pruned += 1
    return {layer: sorted(k) for layer, k in alive.items()}


def select_experts(
    strategy: str,
    layers: Sequence[int],
    activation_per_layer: Dict[int, torch.Tensor],
    similarity_per_layer: Optional[Dict[int, torch.Tensor]] = None,
    magnitude_per_layer: Optional[Dict[int, torch.Tensor]] = None,
    threshold: Optional[float] = None,
    prune_ratio: Optional[float] = None,
    min_keep: int = 6,
    budget: str = "per_layer",
    protect_top_fraction: float = 0.0,
) -> Dict[int, List[int]]:
    """Dispatch to the requested pruning strategy; returns {layer: experts to keep}."""
    if strategy in ("cka", "weight_cosine"):
        sims = {layer: similarity_per_layer[layer] for layer in layers}
        acts = {layer: activation_per_layer[layer] for layer in layers}
        return select_by_redundancy(sims, acts, threshold, prune_ratio, min_keep, budget, protect_top_fraction)
    if strategy == "frequency":
        return select_by_score({layer: activation_per_layer[layer] for layer in layers}, prune_ratio, min_keep, budget)
    if strategy == "weight_magnitude":
        return select_by_score({layer: magnitude_per_layer[layer] for layer in layers}, prune_ratio, min_keep, budget)
    raise ValueError(f"unknown pruning strategy '{strategy}', choose from {STRATEGIES}")
