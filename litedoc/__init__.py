"""
LiteDoc: distilling large document models into efficient task-specific encoders.

Stage 1 - CKA-based expert pruning of the DeepSeek-VL2 MoE language model   (litedoc.pruning, litedoc.cka)
Stage 2 - Sparse-to-Sparse knowledge distillation from the frozen teacher    (litedoc.distill)
Evaluation on document-understanding benchmarks                              (litedoc.data, litedoc.evaluation)
"""

__version__ = "1.0.0"
