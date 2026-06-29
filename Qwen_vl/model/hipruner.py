"""
HiPrune: Hierarchical Pruning for Qwen2.5-VL using vision attention.
Uses shallow-layer + deep-layer attention and spatial neighborhood expansion.
Ref: https://github.com/Danielement321/HiPrune
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class HiPruner(nn.Module):
    """
    HiPrune selection: shallow-layer attention (top-k + neighbor expand) + deep-layer (top-k) to fill.
    forward(image_embeds, attn_list, image_grid_thw) -> (pruned_embeds, pruned_split_sizes).
    """

    def __init__(
        self,
        object_layer: int = 4,
        alpha: float = 0.5,
        retain: float = 0.5,
        visual_token_num: Optional[int] = None,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.object_layer = object_layer
        self.alpha = alpha
        self.retain = retain
        self.visual_token_num = visual_token_num
        self.spatial_merge_size = spatial_merge_size

    @staticmethod
    def _n_image_tokens(grid_thw: torch.Tensor, spatial_merge_size: int = 2) -> List[int]:
        """Token count per image after spatial merge (t * (h/2) * (w/2) for 2x2 merge)."""
        sizes = []
        for i in range(grid_thw.shape[0]):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item()) // spatial_merge_size
            w = int(grid_thw[i, 2].item()) // spatial_merge_size
            sizes.append(t * h * w)
        return sizes

    @torch.no_grad()
    def _select_one_image(
        self,
        shallow_attn: torch.Tensor,
        deep_attn: torch.Tensor,
        n_image_tokens: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        HiPrune selection for a single image.
        Returns indices of tokens to retain (length visual_token_num).
        """
        if self.visual_token_num is not None:
            visual_token_num = min(self.visual_token_num, n_image_tokens)
        else:
            visual_token_num = max(1, round(n_image_tokens * self.retain))

        visual_token_num = min(visual_token_num, n_image_tokens)
        if visual_token_num >= n_image_tokens:
            return torch.arange(n_image_tokens, device=device)

        shallow_token_num = max(0, round((visual_token_num * self.alpha) / 5))

        select_mask = torch.zeros(n_image_tokens, dtype=torch.bool, device=device)

        if shallow_token_num > 0:
            shallow_token_indices = torch.topk(shallow_attn, k=shallow_token_num).indices
            # Spatial neighborhood: ±1, ±width (same as HiPrune official)
            neighbors = torch.cat(
                [
                    shallow_token_indices,
                    shallow_token_indices - 1,
                    shallow_token_indices + 1,
                    shallow_token_indices - width,
                    shallow_token_indices + width,
                ],
                dim=0,
            )
            shallow_token_indices = neighbors.clamp(0, n_image_tokens - 1).unique(sorted=False)
        else:
            shallow_token_indices = torch.empty(0, dtype=torch.long, device=device)

        select_mask[shallow_token_indices] = True
        deep_token_num = visual_token_num - select_mask.sum().item()
        deep_token_num = max(0, deep_token_num)

        if deep_token_num > 0:
            available_index_mask = select_mask.clone()
            deep_attn_masked = deep_attn.clone()
            deep_attn_masked[available_index_mask] = -float("inf")
            deep_token_indices = torch.topk(deep_attn_masked, k=deep_token_num).indices
            select_mask[deep_token_indices] = True

        retain_indices = select_mask.nonzero(as_tuple=True)[0]
        if retain_indices.shape[0] > visual_token_num:
            retain_indices = retain_indices[: visual_token_num]
        elif retain_indices.shape[0] < visual_token_num:
            # Pad with remaining by importance
            remaining = (~select_mask).nonzero(as_tuple=True)[0]
            if remaining.numel() > 0:
                k = visual_token_num - retain_indices.shape[0]
                add = torch.topk(deep_attn[remaining], k=min(k, remaining.numel())).indices
                retain_indices = torch.cat([retain_indices, remaining[add]], dim=0)
        return retain_indices

    @torch.no_grad()
    def forward(
        self,
        image_embeds: torch.Tensor,
        attn_list: List[torch.Tensor],
        image_grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            image_embeds: (total_tokens, D) merged visual features
            attn_list: list of (total_tokens,) per-layer importance from vision
            image_grid_thw: (num_images, 3) [t, h, w] per image

        Returns:
            pruned_embeds: (total_pruned_tokens, D)
            pruned_split_sizes: list of token counts per image
        """
        device = image_embeds.device
        split_sizes = self._n_image_tokens(image_grid_thw, self.spatial_merge_size)

        if len(attn_list) == 0:
            return image_embeds, split_sizes

        object_layer_idx = min(self.object_layer - 1, len(attn_list) - 1)
        shallow_attn_all = attn_list[object_layer_idx]
        deep_attn_all = attn_list[-1]

        pruned_list = []
        pruned_split_sizes = []

        offset = 0
        for i in range(image_grid_thw.shape[0]):
            n = split_sizes[i]
            shallow_attn = shallow_attn_all[offset : offset + n]
            deep_attn = deep_attn_all[offset : offset + n]
            offset += n

            width = int(image_grid_thw[i, 2].item()) // self.spatial_merge_size
            width = max(1, width)

            indices = self._select_one_image(
                shallow_attn, deep_attn, n, width, device
            )
            indices = torch.sort(indices).values
            start = offset - n
            pruned_list.append(image_embeds[start : start + n][indices])
            pruned_split_sizes.append(indices.shape[0])

        pruned_embeds = torch.cat(pruned_list, dim=0)
        return pruned_embeds, pruned_split_sizes
