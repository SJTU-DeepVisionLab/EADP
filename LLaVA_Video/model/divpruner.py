"""
DivPrune for LLaVA-Video
Diversity-based visual token pruning: greedily select tokens that are
maximally distant from already selected ones (pairwise cosine distance).

- Input: (num_frames, tokens_per_frame, D) post-projector features
- Per-frame independent pruning (no text conditioning, no CLIP)
- Uses only LLM-space features; no extra encoders
"""

import torch
import torch.nn as nn
from typing import Tuple


def pairwise_cosine_similarity(matrix: torch.Tensor) -> torch.Tensor:
    """(N, D) -> (N, N) cosine similarity."""
    norm_matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.mm(norm_matrix, norm_matrix.t())


def divprune_select(
    visual_feature_vectors: torch.Tensor,
    num_keep: int,
    distance_matrix: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Greedy diversity selection: at each step, add the token that is
    farthest from the current selected set (maximin diversity).

    Args:
        visual_feature_vectors: (N, D) unused if distance_matrix provided
        num_keep: number of tokens to keep
        distance_matrix: (N, N) distance (1 - cosine_sim), or None to compute
        device: torch device

    Returns:
        s: (num_keep,) indices of selected tokens
    """
    N = distance_matrix.shape[0]
    T = min(num_keep, N)
    s = torch.empty(T, dtype=torch.long, device=device)

    for i in range(T):
        if i == 0:
            m2 = distance_matrix  # (N, N)
            # Per column j: 2 smallest distances (0 is self). Take 2nd = min dist to others.
            scores = torch.topk(m2, 2, dim=0, largest=False).values[1, :]  # (N,)
        else:
            # Rows = selected indices, cols = all tokens
            m2 = torch.index_select(
                distance_matrix, 0,
                torch.index_select(s, 0, torch.arange(i, device=device))
            )  # (i, N)
            scores = torch.min(m2, dim=0).values  # (N,)

        phrase_to_add_idx = torch.argmax(scores).item()
        s[i] = phrase_to_add_idx

    return s


class DivPrunerVideo(nn.Module):
    """
    DivPrune for video: per-frame greedy diversity selection.

    Each frame: compute 1 - pairwise_cosine_similarity(features) as distance,
    then select visual_token_num tokens that are maximally diverse (farthest
    from each other in cosine distance).
    """

    def __init__(
        self,
        visual_token_num: int = 64,
        device: str = "cuda",
    ):
        super().__init__()
        self.visual_token_num = visual_token_num

    @torch.no_grad()
    def forward(
        self,
        image_features: torch.Tensor,
        text: str = "",
    ) -> torch.Tensor:
        """
        Prune visual tokens per frame using DivPrune (diversity only; text ignored).

        Args:
            image_features: (num_frames, tokens_per_frame, D) post-projector,
                            post-spatial-pool video features
            text: unused (DivPrune is text-agnostic)

        Returns:
            pruned_features: (num_frames, visual_token_num, D) pruned features
        """
        device = image_features.device
        num_frames, N, D = image_features.shape
        T = min(self.visual_token_num, N)

        if T >= N:
            return image_features

        pruned_list = []
        for f_idx in range(num_frames):
            frame_feats = image_features[f_idx]  # (N, D)
            frame_feats = frame_feats.float()

            # Distance matrix: 1 - cosine_similarity
            cos_sim = pairwise_cosine_similarity(frame_feats)
            distance_matrix = 1.0 - cos_sim

            select_idx = divprune_select(
                frame_feats, T, distance_matrix, device
            )
            idx_sorted = torch.sort(select_idx).values
            pruned_list.append(image_features[f_idx][idx_sorted])

        return torch.stack(pruned_list, dim=0)
