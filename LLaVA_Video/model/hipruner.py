"""
HiPrune for LLaVA-Video (approximation at post-projector level).

HiPrune (https://github.com/Danielement321/HiPrune) selects three token types:
- Anchor: high attention in middle (object-centric) layers
- Buffer: spatial neighbors of anchors
- Register: high attention in deep layer (global)

LLaVA-Video does not expose vision encoder attention in the eval path; pruning
runs after get_2dPool (post-projector, post-spatial-pool). We approximate:
- Anchor: tokens with highest L2 norm (saliency proxy for object-like regions)
- Buffer: spatial neighbors of anchor indices (±1, ±grid_w)
- Register: from the rest, tokens with highest cosine similarity to global mean
  (global context proxy)

Order: anchor -> buffer -> register. Hyperparameters follow HiPrune defaults
(alpha=0.1 for proportion of anchor+buffer; object_layer N/A here).
"""

import torch
import torch.nn as nn
from typing import Tuple


def _spatial_neighbors_1d(indices: torch.Tensor, grid_h: int, grid_w: int, num_tokens: int) -> torch.Tensor:
    """Expand 1d indices with 4-neighbors (left, right, up, down)."""
    row = indices // grid_w
    col = indices % grid_w
    # ±1 col, ±1 row
    candidates = torch.stack([
        indices,
        row * grid_w + (col - 1).clamp(0, grid_w - 1),
        row * grid_w + (col + 1).clamp(0, grid_w - 1),
        (row - 1).clamp(0, grid_h - 1) * grid_w + col,
        (row + 1).clamp(0, grid_h - 1) * grid_w + col,
    ], dim=0)
    candidates = candidates.flatten().unique(sorted=False)
    candidates = candidates[(candidates >= 0) & (candidates < num_tokens)]
    return candidates


class HiPrunerVideo(nn.Module):
    """
    HiPrune-style token selection for video: anchor (salient) + buffer (spatial)
    + register (global), applied per frame on post-projector features.
    """

    def __init__(
        self,
        visual_token_num: int = 64,
        alpha: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.visual_token_num = visual_token_num
        self.alpha = alpha

    @torch.no_grad()
    def forward(
        self,
        image_features: torch.Tensor,
        text: str = "",
        grid_h: int = 13,
        grid_w: int = 13,
    ) -> torch.Tensor:
        """
        Prune visual tokens per frame using HiPrune-style selection.

        Args:
            image_features: (num_frames, tokens_per_frame, D) post-projector,
                            post-spatial-pool video features
            text: unused (HiPrune is training-free and does not use text at inference)
            grid_h: spatial grid height per frame
            grid_w: spatial grid width per frame

        Returns:
            pruned_features: (num_frames, visual_token_num, D) pruned features
        """
        device = image_features.device
        num_frames, N, D = image_features.shape
        T = min(self.visual_token_num, N)

        if T >= N:
            return image_features

        # Proportion of anchor (+ buffer): HiPrune uses alpha, shallow_token_num = round(N' * alpha / 5)
        anchor_count = max(1, min(N, round(T * self.alpha / 5)))

        pruned_list = []
        for f_idx in range(num_frames):
            feats = image_features[f_idx].float()  # (N, D)

            # --- Anchor: top by L2 norm (object saliency proxy) ---
            norms = feats.norm(dim=-1)  # (N,)
            _, anchor_idx = torch.topk(norms, k=min(anchor_count, N), dim=0, largest=True)

            # --- Buffer: spatial neighbors of anchors ---
            buffer_idx = _spatial_neighbors_1d(anchor_idx, grid_h, grid_w, N)
            if buffer_idx.device != device:
                buffer_idx = buffer_idx.to(device)
            anchor_buffer_set = set(buffer_idx.tolist())

            # --- If we already have enough, take first T and sort ---
            if len(anchor_buffer_set) >= T:
                selected = torch.tensor(sorted(anchor_buffer_set)[:T], device=device, dtype=torch.long)
                pruned_list.append(image_features[f_idx][selected])
                continue

            # --- Register: from remaining, top by cosine sim to global mean ---
            remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
            remaining_mask[buffer_idx] = False
            remaining_idx = torch.where(remaining_mask)[0]

            global_mean = feats.mean(dim=0, keepdim=True)
            global_mean = global_mean / global_mean.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            feats_norm = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            sim_to_mean = (feats_norm @ global_mean.t()).squeeze(-1)  # (N,)

            register_needed = T - len(anchor_buffer_set)
            sim_remaining = sim_to_mean[remaining_idx]
            _, register_top = torch.topk(sim_remaining, k=min(register_needed, len(remaining_idx)), largest=True)
            register_idx = remaining_idx[register_top]

            selected = torch.cat([buffer_idx, register_idx], dim=0)
            selected = torch.unique(selected, sorted=True)
            if selected.numel() > T:
                selected = selected[:T]
            elif selected.numel() < T:
                # Fallback: fill with remaining by norm
                used = set(selected.tolist())
                fill_idx = [i for i in range(N) if i not in used]
                fill_idx = torch.tensor(fill_idx, device=device, dtype=torch.long)
                _, by_norm = torch.topk(norms[fill_idx], k=min(T - selected.numel(), fill_idx.numel()), largest=True)
                selected = torch.cat([selected, fill_idx[by_norm]], dim=0)
                selected = torch.unique(selected, sorted=True)[:T]

            pruned_list.append(image_features[f_idx][selected])

        return torch.stack(pruned_list, dim=0)
