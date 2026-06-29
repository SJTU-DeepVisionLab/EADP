"""
Qwen3-VL fixed-resolution wrappers for baseline, CDPruner, and EADP.

This mirrors the Qwen2.5-VL fixed-resolution path, but uses Qwen3's
patch_size=16 / spatial_merge_size=2 convention. A 1024x1024 image therefore
produces a 64x64 patch grid and 32x32 = 1024 merged visual tokens.
"""

from __future__ import annotations

import logging
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from .model import Qwen3VLChat, ensure_image_url, ensure_video_url


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


QWEN3_FIXED_RESOLUTION = 1024


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def get_spatial_merge_size(model) -> int:
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    return int(getattr(vision_config, "spatial_merge_size", 2))


def get_llm_hidden_size(model) -> int:
    config = getattr(model, "config", None)
    for obj in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "vision_config", None),
    ):
        if obj is None:
            continue
        for attr in ("hidden_size", "out_hidden_size"):
            value = getattr(obj, attr, None)
            if value is not None:
                return int(value)
    raise AttributeError("Could not infer Qwen3 hidden size from model config.")


def unwrap_visual_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Unsupported Qwen3 visual output type: {type(output)}")


def qwen3_vision_attention_with_importance(
    attn_module,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings,
):
    """Run one Qwen3 vision attention block and return per-patch attention importance."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        attn_module.qkv(hidden_states)
        .reshape(seq_length, 3, attn_module.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(
        query_states, key_states, cos, sin
    )

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    query_splits, key_splits, value_splits = [
        torch.split(tensor, lengths, dim=2)
        for tensor in (query_states, key_states, value_states)
    ]

    attn_outputs = []
    importance_parts = []
    for q, k, v in zip(query_splits, key_splits, value_splits):
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * attn_module.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = torch.dropout(attn_weights, p=0.0, train=False)
        attn_output = torch.matmul(attn_weights, v)
        attn_outputs.append(attn_output.transpose(1, 2).contiguous())
        importance_parts.append(attn_weights[0].mean(dim=0).mean(dim=0))

    attn_output = torch.cat(attn_outputs, dim=1)
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = attn_module.proj(attn_output)
    importance = torch.cat(importance_parts, dim=0)
    return attn_output, importance


def qwen3_visual_forward_with_importance(visual, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
    """Qwen3 vision forward path that also records HiPrune-style merged-token importance."""
    hidden_states = visual.patch_embed(hidden_states)

    pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = visual.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    attn_list = []
    deepstack_feature_lists = []
    merge_unit = int(getattr(visual, "spatial_merge_unit", visual.spatial_merge_size ** 2))

    for layer_num, blk in enumerate(visual.blocks):
        attn_out, patch_importance = qwen3_vision_attention_with_importance(
            blk.attn,
            blk.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + blk.mlp(blk.norm2(hidden_states))

        merged_importance = patch_importance.view(
            patch_importance.shape[0] // merge_unit, merge_unit
        ).mean(dim=-1)
        attn_list.append(merged_importance)

        if layer_num in visual.deepstack_visual_indexes:
            deepstack_feature = visual.deepstack_merger_list[
                visual.deepstack_visual_indexes.index(layer_num)
            ](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = visual.merger(hidden_states)
    return hidden_states, deepstack_feature_lists, attn_list


class Qwen3VLChatFixedRes(Qwen3VLChat):
    """Qwen3-VL with fixed 1024x1024 image preprocessing."""

    def _prepare_content(self, inputs, dataset=None):
        content = []
        for s in inputs:
            if s["type"] == "image":
                item = {
                    "type": "image",
                    "image": ensure_image_url(s["value"]),
                }
                if self.min_pixels is not None:
                    item["min_pixels"] = self.min_pixels
                if self.max_pixels is not None:
                    item["max_pixels"] = self.max_pixels
            elif s["type"] == "video":
                item = {
                    "type": "video",
                    "video": ensure_video_url(s["value"]),
                }
                if self.min_pixels is not None:
                    item["min_pixels"] = self.min_pixels
                if self.max_pixels is not None:
                    item["max_pixels"] = self.max_pixels
                if self.fps is not None:
                    item["fps"] = self.fps
            elif s["type"] == "text":
                item = {"type": "text", "text": s["value"]}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, message, dataset=None):
        return self.generate_inner_transformers_fixed(message, dataset=dataset)

    def _post_process_response(self, response):
        if self.post_process:
            resp = response.split("\\boxed{")[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == "{":
                    counter += 1
                elif resp[i] == "}":
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                if i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]
        return response

    def _build_messages(self, message, dataset=None):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({
            "role": "user",
            "content": self._prepare_content(message, dataset=dataset),
        })
        return messages

    def _move_inputs_to_model(self, inputs):
        try:
            device = self.model.device
        except Exception:
            device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        if hasattr(self.model, "dtype"):
            inputs = inputs.to(self.model.dtype)
        return inputs

    def _processor_inputs(self, messages):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        if images is not None:
            image_mean = getattr(self.processor.image_processor, "image_mean", [0.5, 0.5, 0.5])
            bg_color = tuple(int(x * 255) for x in image_mean)
            images = [
                expand2square(img, bg_color).resize(
                    (QWEN3_FIXED_RESOLUTION, QWEN3_FIXED_RESOLUTION)
                )
                for img in images
            ]

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_resize=False,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        return self._move_inputs_to_model(inputs)

    def generate_inner_transformers_fixed(self, message, dataset=None):
        messages = self._build_messages(message, dataset=dataset)
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        inputs = self._processor_inputs(messages)
        generated_ids = self.model.generate(
            **inputs,
            do_sample=False,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        response = self._post_process_response(out[0])
        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response


class Qwen3VLChatCDPruner(Qwen3VLChatFixedRes):
    """Qwen3-VL with CDPruner."""

    def __init__(self, visual_token_num: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.visual_token_num = visual_token_num

        from model.cdpruner import CDPruner

        pruner_device = next(self.model.parameters()).device
        visual_dim = get_llm_hidden_size(self.model)
        spatial_merge_size = get_spatial_merge_size(self.model)

        logging.info(
            f"Loading Qwen3 CDPruner: tokens={visual_token_num}, "
            f"spatial_merge_size={spatial_merge_size}"
        )
        self.pruner = CDPruner(
            visual_token_num=visual_token_num,
            visual_dim=visual_dim,
            spatial_merge_size=spatial_merge_size,
            device=str(pruner_device),
        ).to(pruner_device)
        self.pruner.eval()
        torch.cuda.empty_cache()

    def _tokenize_instruction(self, message: list, dataset=None) -> torch.Tensor:
        content = self._prepare_content(message, dataset)
        instruction_parts = [c["text"] for c in content if c.get("type") == "text"]
        instruction_str = " ".join(instruction_parts).strip() if instruction_parts else "Describe this image."
        enc = self.processor.tokenizer(
            instruction_str,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.processor.tokenizer.model_max_length,
        )
        return enc.input_ids

    @torch.no_grad()
    def _get_instruction_sequence_embedding(self, message: list, dataset=None) -> torch.Tensor:
        device = next(self.model.parameters()).device
        ids = self._tokenize_instruction(message, dataset).to(device)
        return self.model.get_input_embeddings()(ids)

    @torch.no_grad()
    def _get_pruned_image_features(
        self,
        pixel_values,
        image_grid_thw,
        *,
        instruction_embeds_seq=None,
        message=None,
        dataset=None,
    ):
        pixel_values = pixel_values.type(self.model.visual.dtype)
        image_embeds = unwrap_visual_output(
            self.model.visual(pixel_values, grid_thw=image_grid_thw)
        )
        num_images = image_grid_thw.shape[0]

        instruction_embeds = instruction_embeds_seq.mean(dim=1)
        text_embeds_llm = instruction_embeds.expand(num_images, -1)
        pruned_embeds, pruned_split_sizes = self.pruner(
            image_embeds, text_embeds_llm, image_grid_thw
        )
        return pruned_embeds, pruned_split_sizes

    def _build_pruned_inputs(self, input_ids, attention_mask, pruned_image_embeds, pruned_split_sizes):
        device = input_ids.device
        image_token_id = self.model.config.image_token_id
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        new_embeds_list = []
        new_mask_list = []
        embed_idx = 0

        for b in range(input_ids.shape[0]):
            curr_ids = input_ids[b]
            curr_embeds = inputs_embeds[b]
            curr_mask = attention_mask[b]
            image_positions = (curr_ids == image_token_id).nonzero(as_tuple=True)[0]

            if len(image_positions) == 0:
                new_embeds_list.append(curr_embeds)
                new_mask_list.append(curr_mask)
                continue

            segments = []
            seg_start = image_positions[0].item()
            seg_end = seg_start
            for pos in image_positions[1:]:
                if pos.item() == seg_end + 1:
                    seg_end = pos.item()
                else:
                    segments.append((seg_start, seg_end + 1))
                    seg_start = pos.item()
                    seg_end = seg_start
            segments.append((seg_start, seg_end + 1))

            parts_e = []
            parts_m = []
            prev_end = 0

            for seg_start, seg_end in segments:
                if seg_start > prev_end:
                    parts_e.append(curr_embeds[prev_end:seg_start])
                    parts_m.append(curr_mask[prev_end:seg_start])

                if embed_idx >= len(pruned_split_sizes):
                    raise ValueError(
                        "More image-token segments than pruned image feature splits."
                    )
                n = pruned_split_sizes[embed_idx]
                s = sum(pruned_split_sizes[:embed_idx])
                img_e = pruned_image_embeds[s:s + n].to(
                    dtype=curr_embeds.dtype, device=device
                )
                parts_e.append(img_e)
                parts_m.append(torch.ones(n, dtype=curr_mask.dtype, device=device))
                embed_idx += 1
                prev_end = seg_end

            if prev_end < len(curr_ids):
                parts_e.append(curr_embeds[prev_end:])
                parts_m.append(curr_mask[prev_end:])

            new_embeds_list.append(torch.cat(parts_e, dim=0))
            new_mask_list.append(torch.cat(parts_m, dim=0))

        if embed_idx != len(pruned_split_sizes):
            raise ValueError(
                "Fewer image-token segments than pruned image feature splits."
            )

        max_len = max(e.shape[0] for e in new_embeds_list)
        padded_e = []
        padded_m = []
        for e, m in zip(new_embeds_list, new_mask_list):
            pad = max_len - e.shape[0]
            if pad > 0:
                padded_e.append(torch.cat([
                    e,
                    torch.zeros(pad, e.shape[-1], dtype=e.dtype, device=device),
                ]))
                padded_m.append(torch.cat([
                    m,
                    torch.zeros(pad, dtype=m.dtype, device=device),
                ]))
            else:
                padded_e.append(e)
                padded_m.append(m)

        return torch.stack(padded_e), torch.stack(padded_m)

    def generate_inner_transformers_fixed(self, message, dataset=None):
        messages = self._build_messages(message, dataset=dataset)
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        inputs = self._processor_inputs(messages)
        has_images = inputs.get("pixel_values", None) is not None
        image_grid_thw = inputs.get("image_grid_thw", None)

        if has_images and image_grid_thw is not None:
            instruction_embeds_seq = self._get_instruction_sequence_embedding(message, dataset=dataset)
            pruned_embeds, pruned_split_sizes = self._get_pruned_image_features(
                inputs["pixel_values"],
                inputs["image_grid_thw"],
                instruction_embeds_seq=instruction_embeds_seq,
                message=message,
                dataset=dataset,
            )
            new_inputs_embeds, new_attention_mask = self._build_pruned_inputs(
                inputs["input_ids"],
                inputs["attention_mask"],
                pruned_embeds,
                pruned_split_sizes,
            )
            generated_ids = self.model.generate(
                inputs_embeds=new_inputs_embeds,
                attention_mask=new_attention_mask,
                do_sample=False,
                **self.generate_kwargs,
            )
            out = self.processor.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                **self.generate_kwargs,
            )
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.processor.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        response = self._post_process_response(out[0])
        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response


class Qwen3VLChatEADP(Qwen3VLChatCDPruner):
    """Qwen3-VL with EADP."""

    def __init__(
        self,
        visual_token_num: int = 128,
        alpha: float = 0.5,
        beta: float = 1.0,
        **kwargs,
    ):
        Qwen3VLChatFixedRes.__init__(self, **kwargs)
        self.visual_token_num = visual_token_num
        self.alpha = alpha
        self.beta = beta

        from model.pruner import VisualTokenPruner

        pruner_device = next(self.model.parameters()).device
        visual_dim = get_llm_hidden_size(self.model)
        spatial_merge_size = get_spatial_merge_size(self.model)

        logging.info(
            f"Loading Qwen3 EADP: tokens={visual_token_num}, "
            f"alpha={alpha}, beta={beta}, spatial_merge_size={spatial_merge_size}"
        )
        self.pruner = VisualTokenPruner(
            visual_token_num=visual_token_num,
            alpha=alpha,
            beta=beta,
            visual_dim=visual_dim,
            spatial_merge_size=spatial_merge_size,
            device=str(pruner_device),
        ).to(pruner_device)
        self.pruner.eval()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _get_pruned_image_features(
        self,
        pixel_values,
        image_grid_thw,
        *,
        instruction_embeds_seq=None,
        message=None,
        dataset=None,
    ):
        pixel_values = pixel_values.type(self.model.visual.dtype)
        image_embeds = unwrap_visual_output(
            self.model.visual(pixel_values, grid_thw=image_grid_thw)
        )
        num_images = image_grid_thw.shape[0]

        instruction_embeds = instruction_embeds_seq.mean(dim=1)
        text_embeds_llm = instruction_embeds.expand(num_images, -1)
        text_embeds_seq_llm = instruction_embeds_seq.expand(num_images, -1, -1)
        pruned_embeds, pruned_split_sizes = self.pruner(
            image_embeds,
            text_embeds_llm,
            text_embeds_seq_llm,
            image_grid_thw,
        )
        return pruned_embeds, pruned_split_sizes


class Qwen3VLChatDivPruner(Qwen3VLChatCDPruner):
    """Qwen3-VL with DivPruner."""

    def __init__(self, visual_token_num: int = 128, **kwargs):
        Qwen3VLChatFixedRes.__init__(self, **kwargs)
        self.visual_token_num = visual_token_num

        from model.divpruner import DivPruner

        pruner_device = next(self.model.parameters()).device
        spatial_merge_size = get_spatial_merge_size(self.model)

        logging.info(
            f"Loading Qwen3 DivPruner: tokens={visual_token_num}, "
            f"spatial_merge_size={spatial_merge_size}"
        )
        self.pruner = DivPruner(
            visual_token_num=visual_token_num,
            spatial_merge_size=spatial_merge_size,
        ).to(pruner_device)
        self.pruner.eval()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _get_pruned_image_features(
        self,
        pixel_values,
        image_grid_thw,
        *,
        instruction_embeds_seq=None,
        message=None,
        dataset=None,
    ):
        pixel_values = pixel_values.type(self.model.visual.dtype)
        image_embeds = unwrap_visual_output(
            self.model.visual(pixel_values, grid_thw=image_grid_thw)
        )
        num_images = image_grid_thw.shape[0]

        instruction_embeds = instruction_embeds_seq.mean(dim=1)
        text_embeds_llm = instruction_embeds.expand(num_images, -1)
        pruned_embeds, pruned_split_sizes = self.pruner(
            image_embeds, text_embeds_llm, image_grid_thw
        )
        return pruned_embeds, pruned_split_sizes


class Qwen3VLChatHiPrune(Qwen3VLChatCDPruner):
    """Qwen3-VL with HiPrune-style vision-attention token selection."""

    def __init__(
        self,
        visual_token_num: int = 128,
        object_layer: int = 4,
        alpha: float = 0.5,
        retain: float = 0.5,
        **kwargs,
    ):
        Qwen3VLChatFixedRes.__init__(self, **kwargs)
        self.visual_token_num = visual_token_num
        self.object_layer = object_layer
        self.alpha = alpha
        self.retain = retain

        from model.hipruner import HiPruner

        pruner_device = next(self.model.parameters()).device
        spatial_merge_size = get_spatial_merge_size(self.model)

        logging.info(
            f"Loading Qwen3 HiPrune: tokens={visual_token_num}, "
            f"object_layer={object_layer}, alpha={alpha}, retain={retain}, "
            f"spatial_merge_size={spatial_merge_size}"
        )
        self.pruner = HiPruner(
            object_layer=object_layer,
            alpha=alpha,
            retain=retain,
            visual_token_num=visual_token_num,
            spatial_merge_size=spatial_merge_size,
        ).to(pruner_device)
        self.pruner.eval()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _get_pruned_image_features(
        self,
        pixel_values,
        image_grid_thw,
        *,
        instruction_embeds_seq=None,
        message=None,
        dataset=None,
    ):
        pixel_values = pixel_values.type(self.model.visual.dtype)
        image_embeds, _, attn_list = qwen3_visual_forward_with_importance(
            self.model.visual,
            pixel_values,
            image_grid_thw,
        )
        pruned_embeds, pruned_split_sizes = self.pruner(
            image_embeds, attn_list, image_grid_thw
        )
        return pruned_embeds, pruned_split_sizes
