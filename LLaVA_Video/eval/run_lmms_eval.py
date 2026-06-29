#!/usr/bin/env python
"""
Wrapper script to register LlavaVidPruned in lmms-eval and then run evaluation.
Also patches the vision tower to process frames in chunks to avoid OOM on 24GB GPUs.

Usage:
    python eval/run_lmms_eval.py --model llava_vid_pruned \
        --model_args pretrained=/path/to/model,pruner_type=cdpruner,visual_token_num=64 \
        --tasks mvbench --batch_size 1 --output_path outputs/

    # For baseline (no pruning):
    python eval/run_lmms_eval.py --model llava_vid \
        --model_args pretrained=/path/to/model,max_frames_num=64,conv_template=qwen_1_5 \
        --tasks mvbench --batch_size 1 --output_path outputs/
"""

import os
import sys

# Add LLaVA_Video root to path so model/ package is importable
llava_video_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, llava_video_root)

# Register our custom model in lmms-eval's model lookup table.
# This version of lmms-eval uses AVAILABLE_SIMPLE_MODELS (not MODEL_REGISTRY),
# so @register_model alone is insufficient.
import eval.llava_vid_pruned  # noqa: F401
import lmms_eval.models
lmms_eval.models.AVAILABLE_SIMPLE_MODELS["llava_vid_pruned"] = \
    "eval.llava_vid_pruned.LlavaVidPruned"

import torch
from lmms_eval.models.simple.llava_vid import LlavaVid

_original_init = LlavaVid.__init__

def _patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    _patch_chunked_encode(self.model)

def _patch_chunked_encode(model, chunk_size=8):
    """Monkey-patch encode_images to process frames in smaller chunks.

    The SigLIP vision tower uses vanilla attention (O(N^2) memory per frame batch).
    With 64 frames in a single batch, the attention matrices exceed 24GB GPU memory.
    By processing in chunks of `chunk_size`, peak VRAM is reduced ~8x while output
    is bit-for-bit identical (each frame is encoded independently).
    """
    original_encode = model.encode_images

    @torch.no_grad()
    def chunked_encode_images(images):
        if images.shape[0] <= chunk_size:
            return original_encode(images)
        chunks = torch.split(images, chunk_size, dim=0)
        encoded_chunks = []
        for chunk in chunks:
            encoded_chunks.append(original_encode(chunk))
            torch.cuda.empty_cache()
        return torch.cat(encoded_chunks, dim=0)

    model.encode_images = chunked_encode_images


def _patch_logits_to_keep():
    """Patch LlavaQwenForCausalLM.forward to pass num_logits_to_keep=1.

    transformers 4.45 computes logits for ALL positions by default (num_logits_to_keep=0).
    With 10k+ visual tokens this allocates ~6GB just for the lm_head output.
    LlavaQwen's forward() doesn't accept/pass this param, so we replace it at
    the class level to always forward num_logits_to_keep=1 to Qwen2ForCausalLM.
    """
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    def forward_with_logit_keep(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_sizes=None,
        return_dict=None,
        modalities=["image"],
        dpo_forward=False,
        cache_position=None,
        num_logits_to_keep=1,
        **kwargs,
    ):
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values,
             inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values,
                labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        return Qwen2ForCausalLM.forward(
            self,
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, labels=labels,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

    LlavaQwenForCausalLM.forward = forward_with_logit_keep

LlavaVid.__init__ = _patched_init
_patch_logits_to_keep()

# Now run lmms-eval's CLI
from lmms_eval.__main__ import cli_evaluate
cli_evaluate()
