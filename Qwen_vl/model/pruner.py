"""
Visual Token Pruner for Qwen-VL.

VisualTokenPruner: Pure LLM space; forward(imgs, text_embeds_llm, text_embeds_seq_llm, grid_thw).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def _sim_visual_impl(image_features: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise visual similarity matrix.

    Args:
        image_features: (B, N, D) visual features

    Returns:
        sim_matrix: (B, N, N) similarity matrix in [0, 1]
    """
    v = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v = v.float()
    cos_sim = torch.matmul(v, v.transpose(1, 2))
    cos_sim = cos_sim.clamp(-1.0, 1.0)
    sim_matrix = 0.5 * (cos_sim + 1.0)
    return sim_matrix


def _entropy_filter_impl(
    local_sim_all: torch.Tensor,
    T: float = 100.0,
    entropy_keep_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter out high-entropy (uninformative) text tokens.

    Args:
        local_sim_all: (B, N, M, L)
        T: temperature for softmax
        entropy_keep_ratio: ratio of text tokens to keep

    Returns:
        filtered_local_sim: (B, N, M, num_keep)
        top_entropy_vals: (B, M, num_keep)
    """
    sim_logits = local_sim_all * T
    sim_probs = torch.softmax(sim_logits, dim=1)  # softmax over visual tokens

    epsilon = 1e-12
    text_entropy = -torch.sum(sim_probs * torch.log(sim_probs + epsilon), dim=1)  # (B, M, L)

    num_keep = max(1, int(text_entropy.shape[-1] * entropy_keep_ratio))
    top_entropy_vals, top_entropy_indices = torch.topk(
        text_entropy, k=num_keep, dim=-1, largest=False
    )  # (B, M, num_keep) - keep lowest entropy tokens

    B, N, M, L = local_sim_all.shape
    expanded_indices = top_entropy_indices.unsqueeze(1).expand(-1, N, -1, -1)
    filtered_local_sim = torch.gather(local_sim_all, dim=-1, index=expanded_indices)

    return filtered_local_sim, top_entropy_vals


def _local_aggregation_impl(
    filtered_local_sim: torch.Tensor,
    top_entropy_vals: torch.Tensor,
    M_temp: float = 0.01,
    strategy: str = "negative_entropy",
) -> torch.Tensor:
    """
    Aggregate local similarities using entropy-tempered softmax.

    Args:
        filtered_local_sim: (B, N, M, num_keep)
        top_entropy_vals: (B, M, num_keep)

    Returns:
        local_sim: (B, N, M)
    """
    if strategy == "entropy_tempered":
        H_expanded = top_entropy_vals.unsqueeze(1)
        H_safe = H_expanded + 1e-6
        logits = filtered_local_sim / (M_temp * H_safe)
        weights = torch.softmax(logits, dim=-1)
        local_sim = (weights * filtered_local_sim).sum(dim=-1)
    elif strategy == "negative_entropy":
        H_expanded = top_entropy_vals.unsqueeze(1)
        logits = -H_expanded / (M_temp + 1e-6)
        weights = torch.softmax(logits, dim=-1)
        local_sim = (weights * filtered_local_sim).sum(dim=-1)
    elif strategy == "top_k":
        k_local = min(5, filtered_local_sim.shape[-1])
        local_sim = filtered_local_sim.topk(dim=-1, k=k_local).values.mean(dim=-1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return local_sim


def _spatial_smoothing_impl(
    importance: torch.Tensor,
    grid_h: int,
    grid_w: int,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Apply Gaussian spatial smoothing using actual grid dimensions.

    Args:
        importance: (B, N) importance scores
        grid_h: height of the visual token grid (after merge)
        grid_w: width of the visual token grid (after merge)

    Returns:
        smoothed importance: (B, N)
    """
    B, N = importance.shape
    device = importance.device

    if grid_h * grid_w != N:
        # Fallback: skip smoothing if dimensions don't match
        return importance

    imp_img = importance.view(B, 1, grid_h, grid_w)

    # Create Gaussian kernel
    x_coord = torch.arange(kernel_size, device=device, dtype=importance.dtype) - (kernel_size - 1) / 2
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.sqrt(x_grid ** 2 + y_grid ** 2)
    gaussian_kernel = torch.exp(-(xy_grid ** 2) / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    # Apply conv2d with reflect padding
    pad = kernel_size // 2
    imp_padded = F.pad(imp_img, (pad, pad, pad, pad), mode='reflect')
    imp_smoothed = F.conv2d(imp_padded, gaussian_kernel)

    return imp_smoothed.view(B, N)


def _greed_select_impl(
    importance: torch.Tensor,
    sim_matrix: torch.Tensor,
    token_num: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Greedy submodular maximization (Facility Location) for token selection.

    Args:
        importance: (B, N) importance scores
        sim_matrix: (B, N, N) similarity matrix
        token_num: number of tokens to select

    Returns:
        select_idx: (B, T) selected token indices
        token_quotas: (B,) actual quotas per batch
    """
    B, N = importance.shape
    device = importance.device
    T = min(token_num, N)

    token_quotas = torch.full((B,), T, dtype=torch.long, device=device)

    select_idx = torch.zeros(B, T, dtype=torch.long, device=device)
    current_max_sim = torch.zeros(B, N, device=device)
    selected_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

    for t in range(T):
        cur_max_sim_expanded = current_max_sim.unsqueeze(1)  # (B, 1, N)
        new_coverage = torch.maximum(sim_matrix, cur_max_sim_expanded)
        gains_per_token = new_coverage - cur_max_sim_expanded
        weighted_gains = gains_per_token * importance.unsqueeze(1)
        total_gain = weighted_gains.sum(dim=2)  # (B, N)
        total_gain[selected_mask] = -1e9

        best_idx = torch.argmax(total_gain, dim=1)
        select_idx[:, t] = best_idx

        batch_indices = torch.arange(B, device=device)
        chosen_sim_row = sim_matrix[batch_indices, best_idx, :]
        current_max_sim = torch.maximum(current_max_sim, chosen_sim_row)
        selected_mask[batch_indices, best_idx] = True

    return select_idx, token_quotas


def _sim_cross(text_embeds, text_embeds_seq, image_embeds):
    image_embeds = image_embeds.float() / image_embeds.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    text_embeds = text_embeds.float() / text_embeds.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    text_embeds_seq = text_embeds_seq.float() / text_embeds_seq.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    global_sim = -torch.einsum('bnc,mc->bnm', image_embeds, text_embeds)
    local_sim_all = -torch.einsum('bnc,mlc->bnml', image_embeds, text_embeds_seq)
    return global_sim, local_sim_all


class VisualTokenPruner(nn.Module):
    """EADP: all in LLM space; forward(imgs, text_embeds_llm, text_embeds_seq_llm, grid_thw)."""

    def __init__(
        self,
        visual_token_num: int = 128,
        alpha: float = 0.5,
        beta: float = 1.0,
        visual_dim: int = 3584,
        spatial_merge_size: int = 2,
        device: str = "cuda",
    ):
        super().__init__()
        self.visual_token_num = visual_token_num
        self.alpha = alpha
        self.beta = beta
        self.visual_dim = visual_dim
        self.spatial_merge_size = spatial_merge_size

    def compute_importance(self, image_features, text_embeds, text_embeds_seq, grid_h, grid_w):
        global_sim, local_sim_all = _sim_cross(text_embeds, text_embeds_seq, image_features)
        filtered_local_sim, top_entropy_vals = _entropy_filter_impl(local_sim_all, T=100.0, entropy_keep_ratio=0.2)
        local_sim = _local_aggregation_impl(
            filtered_local_sim, top_entropy_vals, M_temp=0.01, strategy="negative_entropy"
        )
        text_sim_all = self.alpha * global_sim + (1 - self.alpha) * local_sim
        text_sim = text_sim_all.mean(dim=-1)
        importance = (text_sim - text_sim.min(dim=-1, keepdim=True).values + 1e-6) / \
                     (text_sim.max(dim=-1, keepdim=True).values - text_sim.min(dim=-1, keepdim=True).values + 1e-6)
        importance = _spatial_smoothing_impl(importance, grid_h, grid_w, kernel_size=3, sigma=1.0)
        importance = importance ** self.beta
        return importance

    @torch.no_grad()
    def forward(
        self,
        image_features: torch.Tensor,
        text_embeds_llm: torch.Tensor,
        text_embeds_seq_llm: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main pruning entry point (LLM-space text and visual).

        Args:
            image_features: (total_tokens, D) concatenated visual features (post-merger)
            text_embeds_llm: (M, D) global instruction embedding (mean of instruction tokens)
            text_embeds_seq_llm: (M, L, D) instruction token sequence from model.get_input_embeddings()
            grid_thw: (num_images, 3) tensor of [t, h, w] per image

        Returns:
            pruned_features: (total_pruned_tokens, D)
            pruned_split_sizes: list of ints
        """
        device = image_features.device
        spatial_merge_size = self.spatial_merge_size

        # Split features by image
        split_sizes = (grid_thw.prod(-1) // (spatial_merge_size ** 2)).tolist()
        image_features_list = torch.split(image_features, split_sizes, dim=0)

        pruned_features_list = []
        pruned_split_sizes = []

        for img_idx, img_feats in enumerate(image_features_list):
            # img_feats: (N_i, D) where N_i = (h_i / merge) * (w_i / merge)
            N_i = img_feats.shape[0]
            t_i, h_i, w_i = grid_thw[img_idx].tolist()
            grid_h = int(h_i) // spatial_merge_size
            grid_w = int(w_i) // spatial_merge_size

            # Determine token budget for this image
            token_num = min(self.visual_token_num, N_i)
            if token_num >= N_i:
                # No pruning needed
                pruned_features_list.append(img_feats)
                pruned_split_sizes.append(N_i)
                continue

            # Add batch dimension: (1, N_i, D)
            img_feats_batch = img_feats.unsqueeze(0)

            # All in LLM space: no projection
            sim_matrix = _sim_visual_impl(img_feats_batch)

            text_emb = text_embeds_llm[img_idx : img_idx + 1]  # (1, D)
            text_emb_seq = text_embeds_seq_llm[img_idx : img_idx + 1]  # (1, L, D)

            # Compute importance scores
            importance = self.compute_importance(
                img_feats_batch, text_emb, text_emb_seq, grid_h, grid_w
            )

            # Greedy selection
            select_idx, _ = _greed_select_impl(importance, sim_matrix, token_num)

            # Sort indices to maintain spatial order
            select_idx_sorted = select_idx[0].sort().values

            # Select pruned features
            pruned_feats = img_feats[select_idx_sorted]

            pruned_features_list.append(pruned_feats)
            pruned_split_sizes.append(pruned_feats.shape[0])

        pruned_features = torch.cat(pruned_features_list, dim=0)
        return pruned_features, pruned_split_sizes
