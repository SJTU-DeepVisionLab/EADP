"""
EADP for LLaVA-Video
Per-frame submodular visual token selection with text-guided importance.

Adapted from Qwen2.5-VL EADP. Key difference:
- Input: (num_frames, tokens_per_frame, D) post-projector features
- Per-frame independent pruning
- visual_dim = 3584 (LLaVA-Video uses Qwen2-7B)

When vision_tower_name is set (e.g. LLaVA-Video SigLIP), importance is computed
in aligned vision-language space using vision_space_features + SigLIP text;
otherwise falls back to CLIP text + visual_proj.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple


class VisualTokenPrunerVideo(nn.Module):
    """
    EADP for video: submodular selection with text-guided importance.

    If vision_tower_name is set and vision_space_features are provided, uses
    aligned vision-language space (e.g. SigLIP); else falls back to
    clip_model_name + visual_proj.
    """

    def __init__(
        self,
        visual_token_num: int = 64,
        alpha: float = 0.5,
        beta: float = 1.0,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        visual_dim: int = 3584,
        clip_dim: int = 768,
        device: str = "cuda",
        vision_tower_name: Optional[str] = None,
        siglip_text_encoder: Optional[Tuple[Any, Any]] = None,
    ):
        super().__init__()
        self.visual_token_num = visual_token_num
        self.alpha = alpha
        self.beta = beta
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
        """Load SigLIP (same hub as vision tower) for aligned text embeddings."""
        try:
            from transformers import AutoProcessor, SiglipModel
        except ImportError:
            from transformers import SiglipModel
            AutoProcessor = None
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

    def encode_text(self, texts: list, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._use_vision_tower_features:
            return self._encode_text_siglip(texts, device)
        return self._encode_text_clip(texts, device)

    def _encode_text_clip(self, texts: list, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        max_length = self.clip_tokenizer.model_max_length
        all_text_embeds = []
        all_text_embeds_seq = []

        for text in texts:
            tokens = self.clip_tokenizer(
                text, return_tensors="pt", padding="max_length",
                max_length=max_length, truncation=True
            ).to("cpu")
            with torch.no_grad():
                outputs = self._clip_text_model(**tokens)
            all_text_embeds.append(outputs.text_embeds)
            text_proj = self._clip_text_model.text_projection
            seq_proj = text_proj(outputs.last_hidden_state)
            all_text_embeds_seq.append(seq_proj)

        text_embeds = torch.cat(all_text_embeds, dim=0).to(device)
        text_embeds_seq = torch.cat(all_text_embeds_seq, dim=0).to(device)
        return text_embeds, text_embeds_seq

    def _encode_text_siglip(self, texts: list, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text with SigLIP: global (pooled) + per-token (last_hidden_state), same as CLIP logic."""
        tokenizer = self._siglip_tokenizer
        model = self._siglip_model
        model_device = next(model.parameters()).device
        max_length = getattr(tokenizer, "model_max_length", 64)
        all_text_embeds = []
        all_text_embeds_seq = []
        for text in texts:
            tokens = tokenizer(
                text, return_tensors="pt", padding="max_length",
                max_length=max_length, truncation=True
            )
            tokens = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}
            with torch.no_grad():
                # text_model returns (last_hidden_state, pooler_output)
                text_outputs = model.text_model(**tokens)
                pooled = text_outputs[1]  # (B, D) pooler_output
                last_hidden = text_outputs[0]  # (B, L, D) last_hidden_state, same D as shared space
                all_text_embeds.append(pooled)
                all_text_embeds_seq.append(last_hidden)
        text_embeds = torch.cat(all_text_embeds, dim=0).to(device)
        text_embeds_seq = torch.cat(all_text_embeds_seq, dim=0).to(device)  # (B, L, D)
        return text_embeds, text_embeds_seq

    def sim_visual(self, features: torch.Tensor) -> torch.Tensor:
        v = features / features.norm(dim=-1, keepdim=True)
        v = v.float()
        cos_sim = torch.matmul(v, v.transpose(1, 2))
        return 0.5 * (cos_sim.clamp(-1.0, 1.0) + 1.0)

    def sim_cross(self, text_embeds, text_embeds_seq, image_embeds):
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        text_embeds_seq = text_embeds_seq / text_embeds_seq.norm(dim=-1, keepdim=True)
        global_sim = -torch.einsum('bnc,mc->bnm', image_embeds, text_embeds)
        local_sim_all = -torch.einsum('bnc,mlc->bnml', image_embeds, text_embeds_seq)
        return global_sim, local_sim_all

    def entropy_filter(self, local_sim_all, T=100.0, entropy_keep_ratio=0.5):
        sim_logits = local_sim_all * T
        sim_probs = torch.softmax(sim_logits, dim=1)
        epsilon = 1e-12
        text_entropy = -torch.sum(sim_probs * torch.log(sim_probs + epsilon), dim=1)
        num_keep = max(1, int(text_entropy.shape[-1] * entropy_keep_ratio))
        top_entropy_vals, top_entropy_indices = torch.topk(
            text_entropy, k=num_keep, dim=-1, largest=False
        )
        B, N, M, L = local_sim_all.shape
        expanded_indices = top_entropy_indices.unsqueeze(1).expand(-1, N, -1, -1)
        filtered_local_sim = torch.gather(local_sim_all, dim=-1, index=expanded_indices)
        return filtered_local_sim, top_entropy_vals

    def local_aggregation(self, filtered_local_sim, top_entropy_vals, M_temp=0.01):
        H_expanded = top_entropy_vals.unsqueeze(1)
        H_safe = H_expanded + 1e-6
        logits = filtered_local_sim / (M_temp * H_safe)
        weights = torch.softmax(logits, dim=-1)
        return (weights * filtered_local_sim).sum(dim=-1)

    def spatial_smoothing(self, importance, grid_h, grid_w, kernel_size=3, sigma=1.0):
        B, N = importance.shape
        device = importance.device
        if grid_h * grid_w != N:
            return importance
        imp_img = importance.view(B, 1, grid_h, grid_w)
        x_coord = torch.arange(kernel_size, device=device, dtype=importance.dtype) - (kernel_size - 1) / 2
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.sqrt(x_grid ** 2 + y_grid ** 2)
        gaussian_kernel = torch.exp(-(xy_grid ** 2) / (2 * sigma ** 2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        pad = kernel_size // 2
        imp_padded = F.pad(imp_img, (pad, pad, pad, pad), mode='reflect')
        imp_smoothed = F.conv2d(imp_padded, gaussian_kernel)
        return imp_smoothed.view(B, N)

    def compute_importance(self, image_features, text_embeds, text_embeds_seq, grid_h, grid_w):
        global_sim, local_sim_all = self.sim_cross(text_embeds, text_embeds_seq, image_features)
        filtered_local_sim, top_entropy_vals = self.entropy_filter(local_sim_all)
        local_sim = self.local_aggregation(filtered_local_sim, top_entropy_vals)
        text_sim_all = self.alpha * global_sim + (1 - self.alpha) * local_sim
        text_sim = text_sim_all.mean(dim=-1)
        importance = (text_sim - text_sim.min(dim=-1, keepdim=True).values + 1e-6) / \
                     (text_sim.max(dim=-1, keepdim=True).values - text_sim.min(dim=-1, keepdim=True).values + 1e-6)
        importance = self.spatial_smoothing(importance, grid_h, grid_w)
        importance = importance ** self.beta
        return importance

    def greed_select(self, importance, sim_matrix, token_num):
        B, N = importance.shape
        device = importance.device
        T = min(token_num, N)
        select_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        current_max_sim = torch.zeros(B, N, device=device)
        selected_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        for t in range(T):
            cur_max_sim_expanded = current_max_sim.unsqueeze(1)
            new_coverage = torch.maximum(sim_matrix, cur_max_sim_expanded)
            gains_per_token = new_coverage - cur_max_sim_expanded
            weighted_gains = gains_per_token * importance.unsqueeze(1)
            total_gain = weighted_gains.sum(dim=2)
            total_gain[selected_mask] = -1e9
            best_idx = torch.argmax(total_gain, dim=1)
            select_idx[:, t] = best_idx
            batch_indices = torch.arange(B, device=device)
            chosen_sim_row = sim_matrix[batch_indices, best_idx, :]
            current_max_sim = torch.maximum(current_max_sim, chosen_sim_row)
            selected_mask[batch_indices, best_idx] = True

        return select_idx

    @torch.no_grad()
    def forward(
        self,
        image_features: torch.Tensor,
        text: str,
        grid_h: int = 13,
        grid_w: int = 13,
        vision_space_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prune visual tokens per frame using EADP.

        Args:
            image_features: (num_frames, tokens_per_frame, D) post-projector,
                            post-spatial-pool video features
            text: text prompt for importance computation
            grid_h: spatial grid height per frame (default 13 for LLaVA-Video)
            grid_w: spatial grid width per frame (default 13 for LLaVA-Video)
            vision_space_features: optional (num_frames, tokens_per_frame, D_vision)
                from the model's vision tower (same pooling). When set, importance
                and sim_visual use this in aligned space; else fallback to visual_proj
                or uniform importance.

        Returns:
            pruned_features: (num_frames, visual_token_num, D) pruned features
        """
        device = image_features.device
        num_frames, N, D = image_features.shape
        T = min(self.visual_token_num, N)

        if T >= N:
            return image_features

        use_vision_space = (
            vision_space_features is not None
            and vision_space_features.shape[0] == num_frames
            and vision_space_features.shape[1] == N
        )


        if use_vision_space or self.visual_proj is not None:
            text_embeds, text_embeds_seq = self.encode_text([text], device)
        else:
            text_embeds = text_embeds_seq = None

        pruned_list = []
        for f_idx in range(num_frames):
            frame_feats = image_features[f_idx].unsqueeze(0)  # (1, N, D)

            if use_vision_space:
                clip_feats = vision_space_features[f_idx].unsqueeze(0).float().to(device)
                clip_feats = clip_feats / clip_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            elif self.visual_proj is not None:
                clip_feats = self.visual_proj(frame_feats.float())
            else:
                # Fallback: no aligned features; use normalized post-projector for sim only
                clip_feats = frame_feats.float() / frame_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            sim_matrix = self.sim_visual(clip_feats)
            if text_embeds is not None and text_embeds_seq is not None:
                importance = self.compute_importance(
                    clip_feats, text_embeds, text_embeds_seq, grid_h, grid_w
                )
            else:
                importance = torch.ones(1, N, device=device, dtype=clip_feats.dtype)
                importance = self.spatial_smoothing(importance, grid_h, grid_w)
                importance = importance ** self.beta

            select_idx = self.greed_select(importance, sim_matrix, T)
            idx_sorted = select_idx[0].sort().values
            pruned_list.append(image_features[f_idx][idx_sorted])

        return torch.stack(pruned_list, dim=0)
