"""Loading, saving and generation helpers for (pruned) DeepSeek-VL2 checkpoints."""

import json
import os
import shutil
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

from .data import build_conversation, sample_images

# files of a checkpoint directory that are NOT model weights/config and must travel with a pruned model
_PROCESSOR_PATTERNS = (
    "processor_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
)


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
            "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32}[name]


def local_model_dir(model_path: str) -> str:
    """Local directory of a checkpoint; HuggingFace hub ids are downloaded (non-weight files only)."""
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path, allow_patterns=["*.json", "*.model", "*.txt"])


def load_processor(model_path: str) -> DeepseekVLV2Processor:
    return DeepseekVLV2Processor.from_pretrained(model_path)


def load_model(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = None,
    attn_implementation: Optional[str] = None,
) -> DeepseekVLV2ForCausalLM:
    """
    Load a DeepSeek-VL2 (or LiteDoc-pruned) checkpoint. Pruned checkpoints carry
    `language_config.moe_layer_num_experts`, which the modified MoE layer reads to build the
    right number of experts per layer, so a plain `from_pretrained` is enough.
    """
    kwargs: Dict[str, Any] = dict(trust_remote_code=True, torch_dtype=dtype)
    if device_map is not None:
        kwargs["device_map"] = device_map
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if device_map is None:
        model = model.to(device)
    return model.eval()


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def copy_processor_files(src_model_path: str, dst_dir: str):
    src_dir = local_model_dir(src_model_path)
    os.makedirs(dst_dir, exist_ok=True)
    for name in _PROCESSOR_PATTERNS:
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst_dir, name))


def save_litedoc_model(model: torch.nn.Module, src_model_path: str, out_dir: str,
                       metadata: Optional[Dict[str, Any]] = None, safe_serialization: bool = True):
    """Save weights + config (including the per-layer expert counts) + processor/tokenizer files."""
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=safe_serialization)
    copy_processor_files(src_model_path, out_dir)
    if metadata is not None:
        with open(os.path.join(out_dir, "litedoc.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def prepare_sample_inputs(
    processor: DeepseekVLV2Processor,
    sample: Dict[str, Any],
    answer: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Tokenize one sample. With `answer`, the assistant turn is included (training / teacher forcing)
    and `labels` mark the answer tokens; without it, the prompt is prepared for generation.
    """
    conversation = build_conversation(sample, answer)
    inputs = processor(
        conversations=conversation,
        images=sample_images(sample),
        force_batchify=True,
        inference_mode=answer is None,
        system_prompt="",
    )
    if device is not None:
        inputs = inputs.to(device, dtype=dtype)
    return inputs


@torch.no_grad()
def generate_answer(
    model: DeepseekVLV2ForCausalLM,
    processor: DeepseekVLV2Processor,
    sample: Dict[str, Any],
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.4,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    chunk_size: int = -1,
    dtype: torch.dtype = torch.bfloat16,
    skip_special_tokens: bool = True,
) -> Dict[str, Any]:
    """Greedy (default) or sampled generation for one sample; returns text and token count."""
    tokenizer = processor.tokenizer
    inputs = prepare_sample_inputs(processor, sample, device=model_device(model), dtype=dtype)

    if chunk_size == -1:
        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        past_key_values = None
    else:
        inputs_embeds, past_key_values = model.incremental_prefilling(
            input_ids=inputs.input_ids,
            images=inputs.images,
            images_seq_mask=inputs.images_seq_mask,
            images_spatial_crop=inputs.images_spatial_crop,
            attention_mask=inputs.attention_mask,
            chunk_size=chunk_size,
        )

    generate_kwargs = dict(
        inputs_embeds=inputs_embeds,
        input_ids=inputs.input_ids,
        images=inputs.images,
        images_seq_mask=inputs.images_seq_mask,
        images_spatial_crop=inputs.images_spatial_crop,
        attention_mask=inputs.attention_mask,
        past_key_values=past_key_values,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    if do_sample:
        generate_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p,
                               repetition_penalty=repetition_penalty)
    else:
        generate_kwargs.update(do_sample=False, repetition_penalty=repetition_penalty)

    outputs = model.generate(**generate_kwargs)
    new_tokens = outputs[0][len(inputs.input_ids[0]):].cpu().tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens)
    return {"text": text, "num_new_tokens": len(new_tokens), "num_input_tokens": int(inputs.input_ids.shape[1])}
