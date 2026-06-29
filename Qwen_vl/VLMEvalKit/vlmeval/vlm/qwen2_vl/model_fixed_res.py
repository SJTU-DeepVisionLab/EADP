"""
Qwen2.5-VL with fixed resolution (1008x1008) for fair comparison.

Following CDPruner author's setup:
1. expand2square: pad image to square using image processor's mean color
2. resize to 36*28=1008 pixels
3. min_pixels = max_pixels = 1008*1008
4. do_sample = False (greedy search)

Reference: https://github.com/Theia-4869/CDPruner/issues/13
"""

from __future__ import annotations

import logging
import os
import sys

import torch
from PIL import Image

from .model import Qwen2VLChat, ensure_image_url, ensure_video_url

# Add project model directory to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def expand2square(pil_img, background_color):
    """Pad image to square with given background color."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class Qwen2VLChatFixedRes(Qwen2VLChat):
    """
    Qwen2.5-VL with fixed 1008x1008 resolution.
    Adds expand2square + resize preprocessing before generation.
    """

    def _prepare_content(self, inputs, dataset=None):
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {
                    'type': 'image',
                    'image': ensure_image_url(s['value']),
                }
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': ensure_video_url(s['value']),
                }
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.fps is not None:
                    item['fps'] = self.fps
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, message, dataset=None):
        return self.generate_inner_transformers_fixed(message, dataset=dataset)

    def generate_inner_transformers_fixed(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({
            'role': 'user',
            'content': self._prepare_content(message, dataset=dataset)
        })

        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )

        images, videos = process_vision_info([messages])

        if images is not None:
            bg_color = tuple(
                int(x * 255) for x in self.processor.image_processor.image_mean
            )
            images = [
                expand2square(img, bg_color).resize((36 * 28, 36 * 28))
                for img in images
            ]

        inputs = self.processor(
            text=text, images=images, videos=videos,
            padding=True, return_tensors='pt'
        )
        inputs = inputs.to('cuda')

        generated_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        response = out[0]

        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response


class Qwen2VLChatCDPruner(Qwen2VLChatFixedRes):
    """
    Qwen2.5-VL with CDPruner visual token pruning.
    """

    def __init__(
        self,
        visual_token_num: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_token_num = visual_token_num

        from model.cdpruner import CDPruner

        visual_dim = self.model.config.hidden_size
        pruner_device = next(self.model.parameters()).device

        logging.info(f"Loading CDPruner: visual_token_num={visual_token_num}")
        self.pruner = CDPruner(
            visual_token_num=visual_token_num,
            visual_dim=visual_dim,
            device=str(pruner_device),
        )
        
        self.pruner = self.pruner.to(pruner_device)
        self.pruner.eval()
        torch.cuda.empty_cache()

    def _tokenize_instruction(self, message: list, dataset=None) -> torch.Tensor:
        """Get instruction string from _prepare_content and tokenize; returns input_ids (1, L)."""
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
    def _get_instruction_embedding(self, message: list, dataset=None) -> torch.Tensor:
        """Get instruction embedding (mean of token embeds). Returns (1, D) in LLM space."""
        device = next(self.model.parameters()).device
        ids = self._tokenize_instruction(message, dataset).to(device)
        embeds = self.model.get_input_embeddings()(ids)
        return embeds.mean(dim=1)

    @torch.no_grad()
    def _get_instruction_sequence_embedding(self, message: list, dataset=None) -> torch.Tensor:
        """Get instruction token sequence embedding (1, L, D)."""
        device = next(self.model.parameters()).device
        ids = self._tokenize_instruction(message, dataset).to(device)
        embeds = self.model.get_input_embeddings()(ids)
        return embeds

    @torch.no_grad()
    def _get_pruned_image_features(
        self,
        pixel_values,
        image_grid_thw,
        *,
        text_prompt=None,
        instruction_embeds_seq=None,
        message=None,
        dataset=None,
    ):
        """Run ViT and apply pruner."""
        pixel_values = pixel_values.type(self.model.visual.dtype)
        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        num_images = image_grid_thw.shape[0]

        instruction_embeds = instruction_embeds_seq.mean(dim=1)
        text_embeds_llm = instruction_embeds.expand(num_images, -1)
        pruned_embeds, pruned_split_sizes = self.pruner(
            image_embeds, text_embeds_llm, image_grid_thw
        )
        
        return pruned_embeds, pruned_split_sizes

    def _build_pruned_inputs(self, input_ids, attention_mask, pruned_image_embeds, pruned_split_sizes):
        """Replace original image token embeddings with pruned ones."""
        device = input_ids.device
        batch_size = input_ids.shape[0]
        image_token_id = self.model.config.image_token_id

        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        new_embeds_list = []
        new_mask_list = []

        embed_idx = 0
        for b in range(batch_size):
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

                if embed_idx < len(pruned_split_sizes):
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

        max_len = max(e.shape[0] for e in new_embeds_list)
        padded_e = []
        padded_m = []
        for e, m in zip(new_embeds_list, new_mask_list):
            pad = max_len - e.shape[0]
            if pad > 0:
                padded_e.append(torch.cat([
                    e, torch.zeros(pad, e.shape[-1], dtype=e.dtype, device=device)
                ]))
                padded_m.append(torch.cat([
                    m, torch.zeros(pad, dtype=m.dtype, device=device)
                ]))
            else:
                padded_e.append(e)
                padded_m.append(m)

        return torch.stack(padded_e), torch.stack(padded_m)

    def generate_inner_transformers_fixed(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({
            'role': 'user',
            'content': self._prepare_content(message, dataset=dataset)
        })

        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )

        images, videos = process_vision_info([messages])

        if images is not None:
            bg_color = tuple(
                int(x * 255) for x in self.processor.image_processor.image_mean
            )
            images = [
                expand2square(img, bg_color).resize((36 * 28, 36 * 28))
                for img in images
            ]

        inputs = self.processor(
            text=text, images=images, videos=videos,
            padding=True, return_tensors='pt'
        )
        inputs = inputs.to('cuda')

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
                max_new_tokens=self.max_new_tokens,
            )
            out = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        else:
            # Standard generation (no images)
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

        response = out[0]

        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response


class Qwen2VLChatDivPruner(Qwen2VLChatCDPruner):
    """
    Qwen2.5-VL with DivPruner (diversity-based visual token pruning).
    """

    def __init__(
        self,
        visual_token_num: int = 128,
        **kwargs,
    ):
        Qwen2VLChatFixedRes.__init__(self, **kwargs)
        self.visual_token_num = visual_token_num

        from model.divpruner import DivPruner

        pruner_device = next(self.model.parameters()).device

        logging.info(f"Loading DivPruner: visual_token_num={visual_token_num}")
        self.pruner = DivPruner(visual_token_num=visual_token_num)
        self.pruner = self.pruner.to(pruner_device)
        self.pruner.eval()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _get_pruned_image_features(
        self,
        pixel_values,
        image_grid_thw,
        *,
        text_prompt=None,
        instruction_embeds_seq=None,
        message=None,
        dataset=None,
    ):
        """DivPruner expects LLM-space instruction embeddings."""
        pixel_values = pixel_values.type(self.model.visual.dtype)
        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        num_images = image_grid_thw.shape[0]

        if instruction_embeds_seq is None:
            raise ValueError("instruction_embeds_seq is required for DivPruner.")

        instruction_embeds = instruction_embeds_seq.mean(dim=1)
        text_embeds_llm = instruction_embeds.expand(num_images, -1)

        pruned_embeds, pruned_split_sizes = self.pruner(
            image_embeds, text_embeds_llm, image_grid_thw
        )
        return pruned_embeds, pruned_split_sizes


class Qwen2VLChatEADP(Qwen2VLChatCDPruner):
    """
    Qwen2.5-VL with EADP.
    """

    def __init__(
        self,
        visual_token_num: int = 128,
        alpha: float = 0.5,
        beta: float = 1.0,
        **kwargs,
    ):
        # Call grandparent init (Qwen2VLChatFixedRes), skip CDPruner's init
        Qwen2VLChatFixedRes.__init__(self, **kwargs)
        self.visual_token_num = visual_token_num
        self.alpha = alpha
        self.beta = beta

        from model.pruner import VisualTokenPruner

        visual_dim = self.model.config.hidden_size
        pruner_device = next(self.model.parameters()).device

        logging.info(
            f"Loading EADP: tokens={visual_token_num}, alpha={alpha}, beta={beta}"
        )
        self.pruner = VisualTokenPruner(
            visual_token_num=visual_token_num,
            alpha=alpha,
            beta=beta,
            visual_dim=visual_dim,
            device=str(pruner_device),
        )
        
        self.pruner = self.pruner.to(pruner_device)
        self.pruner.eval()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _get_pruned_image_features(
        self,
        pixel_values,
        image_grid_thw,
        *,
        text_prompt=None,
        instruction_embeds_seq=None,
        message=None,
        dataset=None,
    ):
        pixel_values = pixel_values.type(self.model.visual.dtype)
        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        num_images = image_grid_thw.shape[0]

        if instruction_embeds_seq is None:
            raise ValueError("instruction_embeds_seq is required for EADP.")

        instruction_embeds = instruction_embeds_seq.mean(dim=1)
        text_embeds_llm = instruction_embeds.expand(num_images, -1)
        text_embeds_seq_llm = instruction_embeds_seq.expand(num_images, -1, -1)
        pruned_embeds, pruned_split_sizes = self.pruner(
            image_embeds, text_embeds_llm, text_embeds_seq_llm, image_grid_thw
        )
        return pruned_embeds, pruned_split_sizes


class Qwen2VLChatHiPrune(Qwen2VLChatCDPruner):
    """
    Qwen2.5-VL with HiPrune (hierarchical pruning using vision attention).
    Replaces the vision encoder with one that returns attention; uses HiPruner for selection.
    """

    def __init__(
        self,
        visual_token_num: int = 128,
        object_layer: int = 4,
        alpha: float = 0.5,
        retain: float = 0.5,
        **kwargs,
    ):
        Qwen2VLChatFixedRes.__init__(self, **kwargs)
        self.visual_token_num = visual_token_num
        self.object_layer = object_layer
        self.alpha = alpha
        self.retain = retain

        from model.vision_hiprune import Qwen2_5_VisionTransformerReturnAttn
        from model.hipruner import HiPruner

        pruner_device = next(self.model.parameters()).device
        self.model.visual = self.model.visual.to(pruner_device)
        state = self.model.visual.state_dict()
        state_cpu = {k: v.cpu() for k, v in state.items()}
        del state

        vision_config = self.model.config.vision_config
        new_visual = Qwen2_5_VisionTransformerReturnAttn(vision_config)
        new_visual.load_state_dict(state_cpu, strict=True)
        del state_cpu
        self.model.visual = new_visual.to(pruner_device)
        logging.info(
            f"Loading HiPrune: visual_token_num={visual_token_num}, "
            f"object_layer={object_layer}, alpha={alpha}, retain={retain}"
        )
        self.pruner = HiPruner(
            object_layer=object_layer,
            alpha=alpha,
            retain=retain,
            visual_token_num=visual_token_num,
            spatial_merge_size=getattr(vision_config, "spatial_merge_size", 2),
        )
        self.pruner = self.pruner.to(pruner_device)
        self.pruner.eval()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _get_pruned_image_features(
        self,
        pixel_values,
        image_grid_thw,
        *,
        text_prompt=None,
        instruction_embeds_seq=None,
        message=None,
        dataset=None,
    ):
        """Run HiPrune visual (with attention) and apply HiPruner selection."""
        pixel_values = pixel_values.type(self.model.visual.dtype)
        image_embeds, attn_list = self.model.visual(
            pixel_values, grid_thw=image_grid_thw, return_attn=True
        )
        pruned_embeds, pruned_split_sizes = self.pruner(
            image_embeds, attn_list, image_grid_thw
        )
        return pruned_embeds, pruned_split_sizes
