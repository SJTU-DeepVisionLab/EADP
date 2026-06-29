"""
DivPruner for Qwen2.5-VL

Diversity-based visual token pruning: iteratively select tokens that maximize
pairwise distance (1 - cosine similarity) to already selected tokens.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class DivPruner(nn.Module):
    """
    DivPruner: Diversity-based visual token pruning.
    Uses pairwise cosine similarity and greedy max-diversity selection.
    """

    def __init__(
        self,
        visual_token_num: int = 128,
        spatial_merge_size: int = 2,
        **kwargs,  # ignore extra args for API compatibility with CDPruner
    ):
        super().__init__()
        self.visual_token_num = visual_token_num
        self.spatial_merge_size = spatial_merge_size

    @staticmethod
    def pairwise_cosine_similarity(matrix: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine similarity: (N, D) -> (N, N)."""
        norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
        cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
        return cosine_similarity

    def divprune_core(
        self,
        visual_feature_vectors: torch.Tensor,
        image_feature_length: int,
        num_keep: int,
        cosine_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core DivPrune algorithm: greedy selection by diversity (distance).
        Uses a fixed number of tokens to keep (128 / 256 / 512), not threshold_ratio.

        Args:
            visual_feature_vectors: (N, D) visual features
            image_feature_length: N (number of tokens)
            num_keep: exact number of tokens to keep (e.g. 128, 256, 512)
            cosine_matrix: optional precomputed (N, N); if None, computed from features

        Returns:
            s: (num_keep,) indices of selected tokens
            cosine_matrix: (N, N) distance matrix (1 - cosine_sim) used for selection
        """
        threshold_terms = min(num_keep, image_feature_length)
        if threshold_terms <= 0:
            return torch.empty(0, dtype=torch.long, device=visual_feature_vectors.device), cosine_matrix

        if cosine_matrix is None:
            cosine_sim = self.pairwise_cosine_similarity(
                visual_feature_vectors)
            cosine_matrix = 1.0 - cosine_sim  # distance

        s = torch.empty(threshold_terms, dtype=torch.long,
                        device=visual_feature_vectors.device)
        for i in range(threshold_terms):
            if i == 0:
                m2 = cosine_matrix
            else:
                prev_idx = torch.index_select(
                    s, 0, torch.arange(0, i, device=cosine_matrix.device)
                )
                m2 = torch.index_select(cosine_matrix, 0, prev_idx)

            if i == 0:
                # for distance: second smallest (first is self)
                scores = torch.topk(m2, 2, dim=0, largest=False).values[1, :]
            else:
                scores = torch.min(m2, dim=0).values

            phrase_to_add_idx = torch.argmax(scores).item()
            s[i] = phrase_to_add_idx

        return s, cosine_matrix

    @torch.no_grad()
    def forward(
        self,
        image_features: torch.Tensor,
        text_embeds_llm: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, list]:
        """
        Prune visual tokens per image (same interface as CDPruner).

        Args:
            image_features: (total_tokens, D) concatenated visual features
            text_embeds_llm: (M, D) instruction embeddings in LLM space (e.g. mean of instruction token embeddings), D = visual_dim.
            grid_thw: (num_images, 3) tensor of [t, h, w] for each image

        Returns:
            pruned_features: (total_pruned_tokens, D)
            pruned_split_sizes: list of token counts per image
        """
        split_sizes = (grid_thw.prod(-1) // (self.spatial_merge_size ** 2)).tolist()
        image_features_list = torch.split(image_features, split_sizes, dim=0)

        pruned_features_list = []
        pruned_split_sizes = []

        for img_feats in image_features_list:
            N_i = img_feats.shape[0]
            token_num = min(self.visual_token_num, N_i)

            if token_num >= N_i:
                pruned_features_list.append(img_feats)
                pruned_split_sizes.append(N_i)
                continue

            # 直接使用保留 token 数目 128/256/512，不用 threshold_ratio
            s, _ = self.divprune_core(
                img_feats,
                N_i,
                num_keep=token_num,
                cosine_matrix=None,
            )
            # Sort indices to preserve spatial order (consistent with CDPruner)
            s = torch.sort(s).values
            pruned_feats = img_feats[s]
            pruned_features_list.append(pruned_feats)
            pruned_split_sizes.append(pruned_feats.shape[0])

        pruned_features = torch.cat(pruned_features_list, dim=0)
        return pruned_features, pruned_split_sizes
