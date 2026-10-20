"""
Centered Kernel Alignment (Kornblith et al., 2019) used by LiteDoc to measure the
functional similarity of MoE experts in feature space (paper Sec. 3.2, Eq. 3-4).

For two experts e_i, e_j evaluated on the same n tokens, let X, Y in R^{n x d} be
their outputs, K = X X^T, L = Y Y^T and H = I_n - (1/n) J_n the centering matrix:

    HSIC(K, L) = tr(K H L H) / (n - 1)^2
    CKA(K, L)  = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

Because CKA compares activations instead of weights it is invariant to the
arbitrary permutation of the experts' hidden neurons.
"""

from typing import Optional

import torch


def centering_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """H = I_n - (1/n) * 1 1^T"""
    return torch.eye(n, device=device, dtype=dtype) - torch.full((n, n), 1.0 / n, device=device, dtype=dtype)


def hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Biased HSIC estimator of Eq. (3) on two n x n Gram matrices."""
    n = K.shape[0]
    if n < 2:
        raise ValueError("HSIC needs at least 2 samples")
    H = centering_matrix(n, device=K.device, dtype=K.dtype)
    return torch.trace(K @ H @ L @ H) / (n - 1) ** 2


def cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Reference (unvectorized) linear CKA of Eq. (4) between activations X, Y of shape [n, d].
    Kept deliberately close to the paper's formula; `pairwise_linear_cka` is the fast version.
    """
    X = X.double()
    Y = Y.double()
    K = X @ X.T
    L = Y @ Y.T
    return hsic(K, L) / (torch.sqrt(hsic(K, K) * hsic(L, L)) + eps)


def _centered_gram(X: torch.Tensor) -> torch.Tensor:
    # H X X^T H == (H X)(H X)^T, i.e. the Gram matrix of column-centered features
    Xc = X - X.mean(dim=0, keepdim=True)
    return Xc @ Xc.T


@torch.no_grad()
def pairwise_linear_cka(
    features: torch.Tensor,
    eps: float = 1e-12,
    compute_dtype: torch.dtype = torch.float32,
    expert_chunk: Optional[int] = None,
) -> torch.Tensor:
    """
    Pairwise linear CKA between E sets of activations evaluated on the same n inputs.

    Args:
        features: [E, n, d] tensor, features[i] are the outputs of expert i on the n calibration tokens.
        eps: numerical stabilizer for the normalization.
        compute_dtype: dtype used for the Gram matrices.
        expert_chunk: number of experts whose flattened Gram matrices are kept in memory at once
            (None = all). Lower it if the [E, n*n] buffer does not fit in memory.

    Returns:
        [E, E] symmetric CKA matrix with ones on the diagonal (for non-degenerate experts).

    Since tr(K H L H) = <vec(H K H), vec(H L H)>, the (n-1)^2 factors cancel and CKA is the
    cosine similarity between the flattened centered Gram matrices.
    """
    if features.dim() != 3:
        raise ValueError(f"expected features of shape [E, n, d], got {tuple(features.shape)}")
    num_experts, n, _ = features.shape
    if n < 2:
        raise ValueError("CKA needs at least 2 calibration tokens")

    chunk = num_experts if expert_chunk is None else max(1, int(expert_chunk))

    def flat_grams(start: int, end: int) -> torch.Tensor:
        grams = [_centered_gram(features[i].to(compute_dtype)).reshape(-1) for i in range(start, end)]
        return torch.stack(grams, dim=0)

    norms = torch.empty(num_experts, dtype=compute_dtype, device=features.device)
    sim = torch.empty(num_experts, num_experts, dtype=compute_dtype, device=features.device)

    for a in range(0, num_experts, chunk):
        a_end = min(a + chunk, num_experts)
        ga = flat_grams(a, a_end)
        norms[a:a_end] = ga.norm(dim=1)
        for b in range(a, num_experts, chunk):
            b_end = min(b + chunk, num_experts)
            gb = ga if b == a else flat_grams(b, b_end)
            block = ga @ gb.T
            sim[a:a_end, b:b_end] = block
            sim[b:b_end, a:a_end] = block.T
            del gb
        del ga

    denom = norms[:, None] * norms[None, :] + eps
    return (sim / denom).clamp_(-1.0, 1.0).float()
