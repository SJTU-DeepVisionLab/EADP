"""
CDPruner for LLaVA-Video
Conditional DPP-based visual token pruning applied per-frame to video features.

Adapted from Qwen2.5-VL CDPruner. Key difference:
- Input: (num_frames, tokens_per_frame, D) post-projector features
- Per-frame independent pruning (no cross-frame interaction)
- visual_dim = 3584 (LLaVA-Video uses Qwen2-7B with hidden_size=3584)

For MLLMs with aligned vision-text dual towers (e.g. LLaVA-Video with SigLIP),
we use vision encoder output + matching text encoder for relevance (no random proj).
Set vision_tower_name and pass vision_space_features in forward for this mode.
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Tuple


class CDPrunerVideo(nn.Module):
    """
    CDPruner for video: selects tokens per frame using Conditional DPP.

    Algorithm per frame:
    1. Cosine similarity in LLM space (post-projector features)
    2. Text-visual relevance in aligned vision-language space (see doc below)
    3. DPP kernel: K = relevance * similarity * relevance
    4. Fast MAP inference via Cholesky-like greedy selection

    Relevance (Step 2):
    - If vision_tower_name is set and vision_space_features are provided:
      use them with the matching text encoder (e.g. SigLIP for LLaVA-Video),
      so relevance is computed in the native aligned space (no random projector).
    - Otherwise: legacy path uses clip_model_name text encoder + visual_proj
      (random linear from post-projector dim to clip_dim); not recommended.
    """

    def __init__(
        self,
        visual_token_num: int = 64,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        visual_dim: int = 3584,
        clip_dim: int = 768,
        device: str = "cuda",
        vision_tower_name: Optional[str] = None,
        siglip_text_encoder: Optional[Tuple[Any, Any]] = None,
    ):
        super().__init__()
        self.visual_token_num = visual_token_num
        self.clip_dim = clip_dim
        self.visual_dim = visual_dim
        self.vision_tower_name = vision_tower_name
        # Use shared SigLIP text encoder from LlavaVidPruned when provided (same as vision tower)
        if siglip_text_encoder is not None:
            self._siglip_model, self._siglip_tokenizer = siglip_text_encoder
            self._use_vision_tower_features = True
            self.visual_proj = None
        elif vision_tower_name is not None and vision_tower_name != "":
            self._use_vision_tower_features = True
            self._load_siglip_text_encoder(vision_tower_name, device)
            self.visual_proj = None
        else:
            self._use_vision_tower_features = False
            self.visual_proj = nn.Linear(visual_dim, clip_dim, bias=False)
            nn.init.xavier_uniform_(self.visual_proj.weight)
            self._load_clip_text_encoder(clip_model_name, device)

    def _load_clip_text_encoder(self, clip_model_name: str, device: str) -> None:
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        clip_text_model = CLIPTextModelWithProjection.from_pretrained(
            clip_model_name
        )
        clip_text_model = clip_text_model.to("cpu").eval()
        for p in clip_text_model.parameters():
            p.requires_grad = False
        object.__setattr__(self, "_clip_text_model", clip_text_model)

    def _load_siglip_text_encoder(self, vision_tower_name: str, device: str) -> None:
        """Load SigLIP model (same hub name as vision tower) for aligned text embeddings."""
        try:
            from transformers import AutoProcessor, SiglipModel
        except ImportError:
            from transformers import SiglipModel
            AutoProcessor = None
        # Full SiglipModel provides get_text_features() in the shared projection space
        siglip = SiglipModel.from_pretrained(vision_tower_name)
        siglip = siglip.to("cpu").eval()
        for p in siglip.parameters():
            p.requires_grad = False
        object.__setattr__(self, "_siglip_model", siglip)
        if AutoProcessor is not None:
            processor = AutoProcessor.from_pretrained(vision_tower_name)
            object.__setattr__(self, "_siglip_tokenizer", processor.tokenizer)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(vision_tower_name)
            object.__setattr__(self, "_siglip_tokenizer", tokenizer)

    def encode_text(self, texts: list, device: torch.device) -> torch.Tensor:
        if self._use_vision_tower_features:
            return self._encode_text_siglip(texts, device)
        return self._encode_text_clip(texts, device)

    def _encode_text_clip(self, texts: list, device: torch.device) -> torch.Tensor:
        max_length = self.clip_tokenizer.model_max_length
        all_text_embeds = []
        for text in texts:
            tokens = self.clip_tokenizer(
                text, return_tensors="pt", padding="max_length",
                max_length=max_length, truncation=True
            ).to("cpu")
            with torch.no_grad():
                outputs = self._clip_text_model(**tokens)
            all_text_embeds.append(outputs.text_embeds)
        return torch.cat(all_text_embeds, dim=0).to(device)

    def _encode_text_siglip(self, texts: list, device: torch.device) -> torch.Tensor:
        tokenizer = self._siglip_tokenizer
        model = self._siglip_model
        model_device = next(model.parameters()).device
        max_length = getattr(tokenizer, "model_max_length", 64)
        all_text_embeds = []
        for text in texts:
            tokens = tokenizer(
                text, return_tensors="pt", padding="max_length",
                max_length=max_length, truncation=True
            )
            tokens = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}
            with torch.no_grad():
                emb = model.get_text_features(**tokens)
            all_text_embeds.append(emb)
        return torch.cat(all_text_embeds, dim=0).to(device)

    @torch.no_grad()
    def forward(
        self,
        image_features: torch.Tensor,
        text: str,
        vision_space_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prune visual tokens per frame using Conditional DPP.

        Args:
            image_features: (num_frames, tokens_per_frame, D) post-projector,
                            post-spatial-pool video features
            text: text prompt for relevance computation
            vision_space_features: optional (num_frames, tokens_per_frame, D_vision)
                from the model's vision tower (same pooling). When set, relevance
                is computed in aligned vision-language space (no random proj).

        Returns:
            pruned_features: (num_frames, visual_token_num, D) pruned features
        """
        device = image_features.device
        num_frames, N, D = image_features.shape
        T = min(self.visual_token_num, N)

        if T >= N:
            return image_features

        text_embeds = self.encode_text([text], device)
        text_embeds_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Use aligned vision-space features for relevance when provided
        use_vision_space = (
            vision_space_features is not None
            and vision_space_features.shape[0] == num_frames
            and vision_space_features.shape[1] == N
        )


        pruned_list = []
        for f_idx in range(num_frames):
            frame_feats = image_features[f_idx].unsqueeze(0)  # (1, N, D)

            # Step 1: Cosine similarity in LLM space
            feat_norm = frame_feats / frame_feats.norm(dim=-1, keepdim=True)
            feat_norm = feat_norm.float()
            similarity = torch.matmul(feat_norm, feat_norm.transpose(1, 2))  # (1, N, N)

            # Step 2: Text-visual relevance in aligned space (vision tower + text encoder)
            if use_vision_space:
                clip_feats = vision_space_features[f_idx].unsqueeze(0).float().to(device)
                clip_feats = clip_feats / clip_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                relevance = torch.matmul(clip_feats, text_embeds_norm.t())  # (1, N, M)
                relevance = (-relevance).mean(dim=-1)  # (1, N)
            elif self.visual_proj is not None:
                clip_feats = self.visual_proj(frame_feats.float())
                clip_feats = clip_feats / clip_feats.norm(dim=-1, keepdim=True)
                relevance = torch.matmul(clip_feats, text_embeds_norm.t())  # (1, N, M)
                relevance = (-relevance).mean(dim=-1)  # (1, N)
            else:
                # Expected vision_space_features but not provided; use uniform relevance
                relevance = torch.ones(1, N, device=device, dtype=frame_feats.dtype)
            relevance = (relevance - relevance.min(dim=-1, keepdim=True).values + 1e-6) / (
                relevance.max(dim=-1, keepdim=True).values
                - relevance.min(dim=-1, keepdim=True).values + 1e-6
            )

            # Step 3: DPP kernel
            kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)

            # Step 4: Fast MAP inference
            B = 1
            cis = torch.zeros((T, B, N), device=device, dtype=torch.float32)
            di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()
            select_idx = torch.empty((T, B), dtype=torch.long, device=device)

            for i in range(T):
                j = torch.argmax(di2s, dim=-1)
                select_idx[i] = j
                batch_range = torch.arange(B, device=device)
                eis_numer = kernel[batch_range, j] - torch.einsum(
                    'tb,tbn->bn', cis[:i, batch_range, j], cis[:i]
                )
                eis_denom = torch.sqrt(di2s[batch_range, j].clamp(min=1e-8)).unsqueeze(-1)
                eis = eis_numer / eis_denom
                cis[i, :, :] = eis
                di2s -= torch.square(eis)
                di2s[batch_range, j] = -float('inf')

            idx_sorted = torch.sort(select_idx.t()).values[0]  # (T,)
            pruned_list.append(image_features[f_idx][idx_sorted])  # (T, D)

        return torch.stack(pruned_list, dim=0)  # (num_frames, T, D)
