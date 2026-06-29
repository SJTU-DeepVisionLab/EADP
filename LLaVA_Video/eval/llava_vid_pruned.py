"""
LlavaVidPruned: LlavaVid with visual token pruning support.

Inherits from lmms-eval's LlavaVid and adds per-frame token pruning
via monkey-patching `get_2dPool`. After pruning, switches to "flat"
merge to avoid grid-based newline token issues.

Usage:
    Registered as "llava_vid_pruned" in lmms-eval's model registry.
    Extra model_args: pruner_type, visual_token_num, alpha, beta, clip_model_name
"""

import copy
import glob
import math
import os
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.llava_vid import LlavaVid


@register_model("llava_vid_pruned")
class LlavaVidPruned(LlavaVid):
    """
    LlavaVid with visual token pruning.

    Supports:
    - pruner_type="cdpruner": Conditional DPP-based pruning
    - pruner_type="eadp": EADP (with alpha/beta)
    - pruner_type="divprune": Diversity-only pruning (no text, no CLIP)
    - pruner_type="hiprune": HiPrune-style anchor+buffer+register (training-free)
    - pruner_type="none": No pruning (baseline, same as LlavaVid)
    """

    def __init__(
        self,
        pruner_type: str = "none",
        visual_token_num: int = 64,
        alpha: float = 0.5,
        beta: float = 1.0,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        **kwargs,
    ):
        # Initialize parent LlavaVid (loads model, tokenizer, etc.)
        super().__init__(**kwargs)

        pruner_type = pruner_type.lower()

        self._pruner_type = pruner_type
        self._visual_token_num = int(visual_token_num)
        self._current_text = ""

        if pruner_type == "none":
            eval_logger.info("LlavaVidPruned: No pruning (baseline mode)")
            return

        # Add LLaVA_Video root to sys.path for model imports
        llava_video_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        if llava_video_root not in sys.path:
            sys.path.insert(0, llava_video_root)

        device = self._device
        visual_dim = self.model.config.hidden_size  # 3584

        # Use the same vision tower as the main model; load SigLIP text encoder once
        # (same checkpoint as vision tower) and pass to pruners for strict alignment
        vt = self.model.get_vision_tower()
        vision_tower_name = None
        siglip_text_encoder = None
        if vt is not None:
            vision_tower_name = getattr(vt, "vision_tower_name", None) or getattr(
                vt, "name", None
            )
        if pruner_type in ("cdpruner", "eadp") and vision_tower_name:
            siglip_text_encoder = self._load_shared_siglip_text_encoder(
                vision_tower_name
            )

        if pruner_type == "cdpruner":
            from model.cdpruner import CDPrunerVideo

            if vision_tower_name:
                eval_logger.info(
                    f"Initializing CDPruner: tokens={visual_token_num}, "
                    f"vision_tower={vision_tower_name} (aligned relevance)"
                )
            else:
                eval_logger.info(
                    f"Initializing CDPruner: tokens={visual_token_num}, device={device}"
                )
            pruner = CDPrunerVideo(
                visual_token_num=int(visual_token_num),
                clip_model_name=clip_model_name,
                visual_dim=visual_dim,
                clip_dim=768,
                device=str(device),
                vision_tower_name=vision_tower_name or None,
                siglip_text_encoder=siglip_text_encoder,
            )
        elif pruner_type == "eadp":
            from model.pruner import VisualTokenPrunerVideo

            if vision_tower_name:
                eval_logger.info(
                    f"Initializing EADP: tokens={visual_token_num}, "
                    f"alpha={alpha}, beta={beta}, vision_tower={vision_tower_name} (aligned)"
                )
            else:
                eval_logger.info(
                    f"Initializing EADP: tokens={visual_token_num}, "
                    f"alpha={alpha}, beta={beta}, device={device}"
                )
            pruner = VisualTokenPrunerVideo(
                visual_token_num=int(visual_token_num),
                alpha=float(alpha),
                beta=float(beta),
                clip_model_name=clip_model_name,
                visual_dim=visual_dim,
                clip_dim=768,
                device=str(device),
                vision_tower_name=vision_tower_name or None,
                siglip_text_encoder=siglip_text_encoder,
            )
        elif pruner_type == "divprune":
            from model.divpruner import DivPrunerVideo

            eval_logger.info(
                f"Initializing DivPrune: tokens={visual_token_num}, device={device}"
            )
            pruner = DivPrunerVideo(
                visual_token_num=int(visual_token_num),
                device=str(device),
            )
        elif pruner_type == "hiprune":
            from model.hipruner import HiPrunerVideo

            eval_logger.info(
                f"Initializing HiPrune: tokens={visual_token_num}, alpha={alpha}, device={device}"
            )
            pruner = HiPrunerVideo(
                visual_token_num=int(visual_token_num),
                alpha=float(alpha),
                device=str(device),
            )
        else:
            raise ValueError(f"Unknown pruner_type: {pruner_type}")

        pruner = pruner.to(device)
        pruner.eval()
        self._pruner = pruner
        torch.cuda.empty_cache()

        # Compute grid dimensions after pooling
        nps = self.model.get_vision_tower().num_patches_per_side  # 27
        pool_stride = getattr(self.model.config, "mm_spatial_pool_stride", 2)
        pool_mode = getattr(
            self.model.config, "mm_spatial_pool_mode", "average"
        )
        if pool_mode == "bilinear":
            pooled_side = math.ceil(nps / pool_stride)
        else:
            pooled_side = nps // pool_stride
        self._grid_h = pooled_side
        self._grid_w = pooled_side
        eval_logger.info(
            f"Tokens per frame after pooling: {pooled_side}x{pooled_side}="
            f"{pooled_side * pooled_side}, pruning to {visual_token_num}"
        )

        # Monkey-patch get_2dPool to apply pruning after spatial pooling
        original_get_2dPool = self.model.get_2dPool
        self._original_get_2dPool = original_get_2dPool
        pruner_ref = self._pruner
        pruner_type_ref = pruner_type
        grid_h_ref = self._grid_h
        grid_w_ref = self._grid_w
        instance_ref = self

        def pruned_get_2dPool(image_feature, stride=2):
            pooled = original_get_2dPool(image_feature, stride)
            # Shape check: (num_frames, tokens_per_frame, D) for video
            if instance_ref._pruner is not None and pooled.dim() == 3:
                eval_logger.debug(f"pruned_get_2dPool pooled.shape={pooled.shape}")
            text = instance_ref._current_text
            if not text:
                text = "Describe this video."

            vision_space_features = getattr(
                instance_ref, "_clip_visual_features_for_pruner", None
            )
            if pruner_type_ref == "eadp":
                pruned = pruner_ref(
                    pooled, text,
                    grid_h=grid_h_ref,
                    grid_w=grid_w_ref,
                    vision_space_features=vision_space_features,
                )
            elif pruner_type_ref == "hiprune":
                pruned = pruner_ref(
                    pooled, text,
                    grid_h=grid_h_ref,
                    grid_w=grid_w_ref,
                )
            elif pruner_type_ref == "divprune":
                pruned = pruner_ref(pooled, text)
            else:
                pruned = pruner_ref(
                    pooled, text,
                    vision_space_features=vision_space_features,
                )
            return pruned

        self.model.get_2dPool = pruned_get_2dPool

        # Switch to flat merge to avoid grid newline token issues after pruning
        self.model.config.mm_patch_merge_type = "flat"
        eval_logger.info("Set mm_patch_merge_type='flat' for pruned model")

    def _load_shared_siglip_text_encoder(
        self, vision_tower_name: str
    ) -> Tuple[object, object]:
        """Load SigLIP text encoder once (same checkpoint as model's vision tower).

        Returns (model, tokenizer) for pruners so they use the same instance
        as the vision tower's paired text encoder. Cached per vision_tower_name.
        """
        cache_key = "_siglip_text_encoder"
        if getattr(self, cache_key + "_name", None) == vision_tower_name:
            return getattr(self, "_siglip_text_model"), getattr(
                self, "_siglip_tokenizer"
            )
        try:
            from transformers import AutoProcessor, SiglipModel
        except ImportError:
            from transformers import AutoTokenizer, SiglipModel
            AutoProcessor = None  # noqa: F841
        siglip = SiglipModel.from_pretrained(vision_tower_name)
        siglip = siglip.to("cpu").eval()
        for p in siglip.parameters():
            p.requires_grad = False
        if AutoProcessor is not None:
            processor = AutoProcessor.from_pretrained(vision_tower_name)
            tokenizer = processor.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(vision_tower_name)
        setattr(self, "_siglip_text_model", siglip)
        setattr(self, "_siglip_tokenizer", tokenizer)
        setattr(self, cache_key + "_name", vision_tower_name)
        eval_logger.info(
            f"Loaded shared SigLIP text encoder: {vision_tower_name}"
        )
        return siglip, tokenizer

    def _compute_vision_space_features(
        self, video: torch.Tensor, chunk_size: int = 8
    ) -> None:
        """Compute vision encoder output + same pooling for pruner relevance.

        Uses the same vision tower as the main model (self.model.get_vision_tower(),
        i.e. the LLaVA vision tower, e.g. SigLIP) and the same get_2dPool so
        token layout matches the post-projector pooled features. Result is
        stored in self._clip_visual_features_for_pruner. This ensures strict
        alignment with the shared SigLIP text encoder loaded in LlavaVidPruned.
        """
        if not hasattr(self, "_original_get_2dPool"):
            self._clip_visual_features_for_pruner = None
            return
        vt = self.model.get_vision_tower()
        if vt is None:
            self._clip_visual_features_for_pruner = None
            return
        with torch.no_grad():
            num_frames = video.shape[0]
            if num_frames <= chunk_size:
                vision_feats = vt(video)
            else:
                chunks = torch.split(video, chunk_size, dim=0)
                feats_list = [vt(chunk) for chunk in chunks]
                vision_feats = torch.cat(feats_list, dim=0)
                torch.cuda.empty_cache()
            pooled = self._original_get_2dPool(vision_feats, 2)
            self._clip_visual_features_for_pruner = pooled

    def generate_until(self, requests) -> List[str]:
        """Override to store text prompt before model.generate for pruner."""
        from llava.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import (
            KeywordsStoppingCriteria,
            tokenizer_image_token,
        )
        from lmms_eval.models.model_utils.load_video import read_video_pyav
        from PIL import Image

        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            # Store text prompt for pruner
            self._current_text = contexts

            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if os.path.isdir(visuals[0]):
                visuals = glob.glob(visuals[0] + "/*")
            videos = []
            try:
                if len(visuals) == 1:
                    if self.video_decode_backend == "decord":
                        video, frame_time, video_time = self.load_video(
                            visuals[0],
                            self.max_frames_num,
                            self.fps,
                            force_sample=self.force_sample,
                        )
                    elif self.video_decode_backend == "pyav":
                        video, frame_time, video_time = read_video_pyav(
                            visuals[0],
                            self.max_frames_num,
                            self.fps,
                            force_sample=self.force_sample,
                        )
                    elif self.video_decode_backend == "image":
                        video = self.load_image(visuals[0])
                else:
                    if task == "seedbench":
                        video = visuals
                        frame_time = "1.00s"
                        video_time = 1
                    elif "mvbench" in task:
                        fps = 3
                        video_time = len(visuals) / fps
                        sampled_indices = np.linspace(
                            0, len(visuals) - 1, self.max_frames_num, dtype=int
                        )
                        frame_idx = sampled_indices.tolist()
                        frame_time = [i / fps for i in frame_idx]
                        frame_time = ",".join(
                            [f"{i:.2f}s" for i in frame_time]
                        )
                        video = np.stack(
                            [
                                np.array(Image.open(visuals[i]))
                                for i in frame_idx
                            ],
                            axis=0,
                        )

                video = self._image_processor.preprocess(
                    video, return_tensors="pt"
                )["pixel_values"].cuda()
                if self.torch_dtype == "bfloat16":
                    video = video.bfloat16()
                else:
                    video = video.half()
                videos.append(video)
            except Exception as e:
                eval_logger.info(f"{e}")
                eval_logger.info(
                    f"Video {visuals} can not load, check the source"
                )
                video_path = "\n".join(visuals)
                res.append(
                    f"Video {video_path} can not load, check the source"
                )
                pbar.update(1)
                continue

            qs = contexts
            if getattr(self, "add_time_instruction", False):
                time_instruciton = (
                    f"The video lasts for {video_time:.2f} seconds, and "
                    f"{len(video)} frames are uniformly sampled from it. "
                    f"These frames are located at {frame_time}."
                    "Please answer the following questions related to this video."
                )
                qs = f"{time_instruciton}\n{qs}"
            if self.model.config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN * len(videos) + "\n" + qs

            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .cuda()
            )
            pad_token_ids = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            if "llama_3" in self.conv_template:
                pad_token_ids = 0
            attention_masks = input_ids.ne(pad_token_ids).long().cuda()

            stop_str = (
                conv.sep
                if conv.sep_style != SeparatorStyle.TWO
                else conv.sep2
            )
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(
                keywords, self.tokenizer, input_ids
            )

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # For CDPruner/EADP with aligned dual-tower: compute vision-space features
            # (vision tower output + same pooling) for text-visual relevance
            if (
                self._pruner_type in ("cdpruner", "eadp")
                and getattr(self._pruner, "_use_vision_tower_features", False)
                and len(videos) > 0
            ):
                self._compute_vision_space_features(videos[0], chunk_size=8)
            else:
                self._clip_visual_features_for_pruner = None

            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=videos,
                    attention_mask=attention_masks,
                    modalities="video",
                    use_cache=self.use_cache,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True
                    if gen_kwargs["temperature"] > 0
                    else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )

            outputs = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()
            eval_logger.debug(f"Question: {contexts}")
            eval_logger.debug(f"Answer: {outputs}")
            res.append(outputs)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """Override to store text prompt before model forward for pruner."""
        from llava.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import tokenizer_image_token

        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            # Store text prompt for pruner
            self._current_text = contexts

            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(
                    self.task_dict[task][split][doc_id]
                )
            visuals = [
                doc_to_visual(self.task_dict[task][split][doc_id])
            ]
            visuals = self.flatten(visuals)
            videos = []
            for visual in visuals:
                video, frame_time, video_time = self.load_video(
                    visual,
                    self.max_frames_num,
                    self.fps,
                    force_sample=self.force_sample,
                )
                video = self._image_processor.preprocess(
                    video, return_tensors="pt"
                )["pixel_values"].cuda()
                if self.torch_dtype == "bfloat16":
                    video = video.bfloat16()
                else:
                    video = video.half()
                videos.append(video)

            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            contxt_id = (
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(self.device)
            )

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], continuation)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .cuda()
            )
            attention_masks = input_ids.ne(
                self.tokenizer.pad_token_id
            ).long().cuda()

            labels = input_ids.clone()
            labels[0, : contxt_id.shape[1]] = -100

            if (
                self._pruner_type in ("cdpruner", "eadp")
                and getattr(self._pruner, "_use_vision_tower_features", False)
                and len(videos) > 0
            ):
                self._compute_vision_space_features(videos[0], chunk_size=8)
            else:
                self._clip_visual_features_for_pruner = None

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    images=videos,
                    modalities="video",
                )

            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]
            greedy_tokens = greedy_tokens[
                :, contxt_id.shape[1] : input_ids.shape[1]
            ]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res
