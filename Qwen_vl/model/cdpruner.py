"""
CDPruner for Qwen-VL.

CDPruner: Pure LLM space; forward(imgs, text_embeds_llm, grid_thw).
"""

import torch
import torch.nn as nn
from typing import Tuple

class CDPruner(nn.Module):
    """
    CDPruner: Conditional DPP in LLM embedding space.
    forward(image_features, text_embeds_llm, grid_thw).
    """

    def __init__(
        self,
        visual_token_num: int = 128,
        visual_dim: int = 3584,
        spatial_merge_size: int = 2,
        device: str = "cuda",
    ):
        super().__init__()
        self.visual_token_num = visual_token_num
        self.visual_dim = visual_dim
        self.spatial_merge_size = spatial_merge_size

    @torch.no_grad()
    def forward(
        self,
        image_features: torch.Tensor,
        text_embeds_llm: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, list]:
        device = image_features.device
        spatial_merge_size = self.spatial_merge_size
        split_sizes = (grid_thw.prod(-1) // (spatial_merge_size ** 2)).tolist()
        image_features_list = torch.split(image_features, split_sizes, dim=0)

        pruned_features_list = []
        pruned_split_sizes = []

        for img_idx, img_feats in enumerate(image_features_list):
            N_i = img_feats.shape[0]
            token_num = min(self.visual_token_num, N_i)
            if token_num >= N_i:
                pruned_features_list.append(img_feats)
                pruned_split_sizes.append(N_i)
                continue

            img_feats_batch = img_feats.unsqueeze(0)
            B = 1

            image_normalized = img_feats_batch / img_feats_batch.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            image_normalized = image_normalized.float()
            similarity = torch.matmul(
                image_normalized, image_normalized.transpose(1, 2)
            )

            img_embeds_llm = img_feats_batch.float() / img_feats_batch.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
            text_emb = text_embeds_llm[img_idx : img_idx + 1].float()
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            relevance = torch.matmul(img_embeds_llm, text_emb.t()).squeeze(-1)
            relevance = (-relevance).float()
            relevance = (relevance - relevance.min(dim=-1, keepdim=True).values + 1e-6) / (
                relevance.max(dim=-1, keepdim=True).values
                - relevance.min(dim=-1, keepdim=True).values
                + 1e-6
            )

            kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
            N = N_i
            cis = torch.zeros((token_num, B, N), device=device, dtype=torch.float32)
            di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()
            select_idx = torch.empty((token_num, B), dtype=torch.long, device=device)

            for i in range(token_num):
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

            select_idx = torch.sort(select_idx.t()).values
            pruned_feats = img_feats[select_idx[0]]
            pruned_features_list.append(pruned_feats)
            pruned_split_sizes.append(pruned_feats.shape[0])

        pruned_features = torch.cat(pruned_features_list, dim=0)
        return pruned_features, pruned_split_sizes
