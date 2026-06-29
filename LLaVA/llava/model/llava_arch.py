#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
import os

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config, **kwargs):
        super(LlavaMetaModel, self).__init__(config, **kwargs)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def joint_reshape(self, B, N, C, image_features, image_embeds, split_sizes=None):
        
        # [Joint Pruning Logic]
        # If split_sizes are provided and uniform (e.g., all 5 for LLaVA-1.6),
        # we reshape to (Num_Images, Total_Tokens, C) to perform joint selection across all patches of an image.

        K = 1
        new_B = B
        new_N = N

        if split_sizes is not None and len(split_sizes) > 0:
            # Check if all images have the same number of patches
            if len(set(split_sizes)) == 1:
                K = split_sizes[0]
                # Only apply joint pruning if we have multiple patches per image (K > 1)
                # and the total batch size matches
                if K > 1 and B % K == 0:
                    self.is_joint = True
                    new_B = B // K
                    new_N = N * K
                    
                    # Reshape features and embeds
                    image_features = image_features.view(new_B, new_N, -1)
                    image_embeds = image_embeds.view(new_B, new_N, -1)

        return K, new_B, new_N, image_features, image_embeds

    def sim_visual(self, image_features):

        v = image_features / image_features.norm(dim=-1, keepdim=True)
        v = v.float()
        cos_sim = torch.matmul(v, v.transpose(1, 2))  # (B, N, N), range [-1, 1]
        cos_sim = cos_sim.clamp(-1.0, 1.0) # 防止数值误差
        # Sim(i, j) = 0.5 * (cos(i, j) + 1), 映射到 [0, 1] 区间，保证增益计算的稳定性
        sim_matrix = 0.5 * (cos_sim + 1.0) 

        return sim_matrix
    
    def sim_cross(self, text_embeds, text_embeds_seq, image_embeds, original_B=None, original_N=None):

        device = image_embeds.device
        text_embeds = text_embeds.to(device)
        text_embeds_seq = text_embeds_seq.to(device)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)
        text_embeds_seq = text_embeds_seq / text_embeds_seq.norm(dim=-1, keepdim=True)
        
        # Global: (B, N, C) x (M, C) -> (B, N, M)
        global_sim = -torch.einsum('bnc,mc->bnm', image_embeds, text_embeds)
        # Local: (B, N, C) x (M, L, C) -> (B, N, M, L)
        local_sim_all = -torch.einsum('bnc,mlc->bnml', image_embeds, text_embeds_seq)

        # Cache similarity matrices for visualization
        if getattr(self, 'enable_similarity_caching', False) and original_B is not None and original_N is not None:
            self.similarity_debug_cache = {
                'global_sim': global_sim.view(original_B, original_N, -1).detach().cpu(),
                'local_sim_all': local_sim_all.view(original_B, original_N, local_sim_all.shape[2], local_sim_all.shape[3]).detach().cpu(),
            }

        return global_sim, local_sim_all

    def entrpy_filter(self, local_sim_all, T = 100.0, entropy_keep_ratio = 0.5):
        """
        [Entropy Filtering]
        1. Compute entropy for each text token based on its similarity distribution over visual tokens.
           High entropy -> Diffuse similarity -> Irrelevant token (e.g. "the", "is", punctuation).
        2. Filter out top-K% high entropy text tokens.
        """

        sim_logits = local_sim_all * T # (B, N, M, L)
        
        # Softmax over N (visual tokens) to get p(visual | text)
        sim_probs = torch.softmax(sim_logits, dim=1) # (B, N, M, L)
        
        # Entropy H(text) = - sum_i p_i * log(p_i)
        # Add epsilon to avoid log(0)
        epsilon = 1e-12
        text_entropy = -torch.sum(sim_probs * torch.log(sim_probs + epsilon), dim=1) # (B, M, L)

        # Filter out the top K% highest entropy tokens.
        num_keep = int(text_entropy.shape[-1] * entropy_keep_ratio)
        if num_keep < 1: 
            num_keep = 1
        
        top_entropy_vals, top_entropy_indices = torch.topk(text_entropy, k=num_keep, dim=-1, largest=False) # (B, M, num_keep)
        
        # Gather the relevant local similarities
        # local_sim_all: (B, N, M, L)
        # indices: (B, M, num_keep) -> expand to (B, N, M, num_keep) for gathering

        B, N, M, L = local_sim_all.shape

        expanded_indices = top_entropy_indices.unsqueeze(1).expand(-1, N, -1, -1) # (B, N, M, num_keep)
        
        filtered_local_sim = torch.gather(local_sim_all, dim=-1, index=expanded_indices) # (B, N, M, num_keep)
        
        return filtered_local_sim, top_entropy_vals

    def local_aggregation(self, filtered_local_sim, top_entropy_vals = None, M_temp = 0.01, k_local = 5, strategy = "negative_entropy"):
        """
        [Local Aggregation]
        - Entropy tempered:
            1. For each visual token, aggregate its similarity to the KEPT text tokens.
            2. Strategy: Entropy-tempered Softmax
            w_{i,j} = softmax(s_{i,j} / (M * H_i))
        - negative_entropy:
            1. For each visual token, aggregate its similarity to the KEPT text tokens.
            2. Strategy: Negative Entropy
            w_{i,j} = softmax(-H_i / M_temp)
        - top_k:
            1. For each visual token, aggregate its similarity to the top-K% most similar text tokens.
            2. Strategy: Simple Mean
            w_{i,j} = 1/num_keep * sum_{k=1}^{num_keep} s_{i,j,k}
        
        filtered_local_sim: (B, N, M, num_keep)
        top_entropy_vals: (B, M, num_keep)
        """
        if strategy == "entropy_tempered":

            # Broadcast Entropy: (B, 1, M, num_keep)
            H_expanded = top_entropy_vals.unsqueeze(1)
            H_safe = H_expanded + 1e-6
            
            # Calculate Logits: S / (M * H)
            logits = filtered_local_sim / (M_temp * H_safe)
            
            # Calculate Weights: Softmax over the 'num_keep' dimension
            weights = torch.softmax(logits, dim=-1) # (B, N, M, num_keep)
            
            # Weighted Sum
            local_sim = (weights * filtered_local_sim).sum(dim=-1) # (B, N, M)

        elif strategy == "negative_entropy":
            # logits = -H/M_temp; lower entropy -> larger weight
            H_expanded = top_entropy_vals.unsqueeze(1)  # (B, 1, M, num_keep)
            logits = -H_expanded / (M_temp + 1e-6)
            weights = torch.softmax(logits, dim=-1)  # (B, N, M, num_keep) after broadcast
            local_sim = (weights * filtered_local_sim).sum(dim=-1)  # (B, N, M)

        elif strategy == "top_k":

            k_local = min(k_local, filtered_local_sim.shape[-1])
            # Top-K mean
            local_sim = filtered_local_sim.topk(dim=-1, k=k_local).values.mean(dim=-1) # (B, N, M)

        return local_sim

    def spatial_smoothing(self, importance, kernel_size = 3, sigma = 1.0):

        # Reshape to (B, 1, H, W) assuming square grid
        # Adapt for LLaVA-1.6 (AnyRes): N might be composed of multiple 24x24 patches

        B, N = importance.shape
        device = importance.device

        base_grid_side = 24
        base_patch_tokens = base_grid_side ** 2  # 576
                
        imp_img = None
        
        if N % base_patch_tokens == 0:
            # Case 1: Composite grid (LLaVA-1.6 AnyRes or LLaVA-1.5 standard)
            # Treat as multiple independent 24x24 patches
            num_patches = N // base_patch_tokens
            # Shape: (B * num_patches, 1, 24, 24)
            imp_img = importance.view(B * num_patches, 1, base_grid_side, base_grid_side)
        else:
            # Case 2: Fallback to single square grid if perfect square
            H_grid = int(N**0.5)
            if H_grid ** 2 == N:
                imp_img = importance.view(B, 1, H_grid, H_grid)
        
        if imp_img is not None:
            # Create Gaussian Kernel or Avg Kernel
            # Use Gaussian-like kernel for smooth connection

            x_coord = torch.arange(kernel_size) - (kernel_size - 1) / 2
            x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.sqrt(x_grid**2 + y_grid**2)
            gaussian_kernel = torch.exp(-(xy_grid**2) / (2*sigma**2))
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
            
            gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(device)
            
            # Apply Conv2d with padding to keep size
            # Use 'reflect' padding to avoid edge effects; pad = (kernel_size - 1) // 2
            pad = (kernel_size - 1) // 2
            imp_padded = torch.nn.functional.pad(imp_img, (pad, pad, pad, pad), mode='reflect')
            imp_smoothed = torch.nn.functional.conv2d(imp_padded, gaussian_kernel)
            
            importance = imp_smoothed.view(B, N)

        return importance

    def importance_clipping(self, importance, importance_keep_ratio):
        """
        Clip the importance scores to keep the top K% importance tokens.
        """

        N = importance.shape[-1]

        num_keep = int(N * importance_keep_ratio)
        if num_keep < 1: 
            num_keep = 1
        if num_keep < self.visual_token_num:
            num_keep = self.visual_token_num

        _, selected_idx = importance.topk(dim=-1, k=num_keep)
        selected_mask = torch.zeros_like(importance, dtype=torch.bool, device=importance.device)
        selected_mask.scatter_(dim=-1, index=selected_idx, value=1)

        importance = importance * selected_mask

        return importance

    def strategy_importance_allocation(self, importance, original_B, K, token_quotas):
        """
        [Scheme 1: Importance-Proportional Allocation]
        """
        if K < 2:
            return token_quotas.view(-1)

        B, N = importance.shape
        # importance is (Real_Batch * 5, N_patch)
        # Reshape to calculate sum per patch
        
        importance_view = importance.view(original_B, K, -1)
        patch_scores = importance_view.sum(dim=-1) # (Real_Batch, 5)
        
        # Calculate Total Budget per Image
        quotas_view = token_quotas.view(original_B, K)
        total_budgets = quotas_view.sum(dim=-1, keepdim=True) # (Real_Batch, 1)

        # Allocate
        patch_ratios = patch_scores / (patch_scores.sum(dim=-1, keepdim=True) + 1e-6)
        allocated = (patch_ratios * total_budgets).long()
        
        # Ensure minimum 1 token
        allocated = torch.maximum(allocated, torch.ones_like(allocated))
        
        # Flatten
        return allocated.view(-1)

    def strategy_global_guided_allocation(self, importance, original_B, K, token_quotas, base_patch_tokens=576):
        """
        [Scheme 2: Global-Guided Allocation]
        Use the Global Patch (Patch 0) to guide the allocation of Local Patches.
        """
        if K < 2:
            return token_quotas.view(-1)

        B, N = importance.shape
        device = importance.device
        
        # 1. 提取 Global Patch 的 Importance
        # importance 已经被 reshape 成了 (Real_Batch * 5, N_patch)
        # 我们需要将其 reshape 回 (Real_Batch, 5, N_patch)
        importance_view = importance.view(original_B, K, -1)
        global_importance = importance_view[:, 0, :] # (Real_Batch, N_patch)

        # 2. 将 Global Patch 划分为 2x2 网格，对应 4 个 Local Patches
        # 假设 N_patch = 24*24 = 576
        # Reshape to (Real_Batch, 24, 24)
        H_grid = int(base_patch_tokens**0.5) # 24
        global_map = global_importance.view(original_B, H_grid, H_grid)
        
        # Split into 4 quadrants
        mid = H_grid // 2
        # Order: TL, TR, BL, BR (matches LLaVA-1.6 crop order usually)
        # Quadrants importance sum
        q_TL = global_map[:, :mid, :mid].sum(dim=(1, 2))
        q_TR = global_map[:, :mid, mid:].sum(dim=(1, 2))
        q_BL = global_map[:, mid:, :mid].sum(dim=(1, 2))
        q_BR = global_map[:, mid:, mid:].sum(dim=(1, 2))
        
        quadrant_scores = torch.stack([q_TL, q_TR, q_BL, q_BR], dim=1) # (Real_Batch, 4)
        
        # 3. 分配 Local Patches (Indices 1-4) 的配额
        # Global Patch (Index 0) 保留固定数量或按其自身 Importance 分配
        # Global Patch 拿走 20% 预算，剩下 80% 由 4 个 Local Patches 分
        
        # Calculate Total Budget per Image
        quotas_view = token_quotas.view(original_B, K)
        total_budget = quotas_view.sum(dim=-1, keepdim=True) # (Real_Batch, 1) - using keepdim for broadcasting
        
        # Assuming uniform budget for now, but being safe
        # total_budget is (Real_Batch, 1)
        
        global_budget = (total_budget * 0.2).long()
        local_budget = total_budget - global_budget
        
        # Normalize local scores
        local_ratios = quadrant_scores / (quadrant_scores.sum(dim=-1, keepdim=True) + 1e-6)
        local_quotas = (local_ratios * local_budget).long()
        
        # 4. Fill token_quotas tensor
        # token_quotas is (Real_Batch * 5,)
        new_quotas = torch.zeros_like(token_quotas).view(original_B, K)
        
        new_quotas[:, 0] = global_budget.squeeze(-1)
        new_quotas[:, 1:] = local_quotas
        
        # Flatten back
        return new_quotas.view(-1)

    def greed_select(self, importance, sim_matrix, K=1, strategy='joint'):
        
        B, N = importance.shape
        device = importance.device

        # Default: Uniform Quota
        T_base = self.visual_token_num
        token_quotas = torch.full((B,), T_base, dtype=torch.long, device=device)

        if strategy == 'joint' and self.is_joint:
            # joint strategy: flatten 5 patches -> 1 big patch, select 5*N tokens
            T = T_base * K
            token_quotas = torch.full((B,), T, dtype=torch.long, device=device)
            
        elif strategy in ['importance_based', 'global_guided']:
            # Scheme 1 & 2
            if B % K == 0:
                original_B = B // K
                if strategy == 'importance_based':
                    token_quotas = self.strategy_importance_allocation(importance, original_B, K, token_quotas)
                elif strategy == 'global_guided':
                    token_quotas = self.strategy_global_guided_allocation(importance, original_B, K, token_quotas)

        # --- Vectorized Variable-K Selection ---
        max_T = token_quotas.max().item()
        
        select_idx = torch.zeros(B, max_T, dtype=torch.long, device=device)
        current_max_sim = torch.zeros(B, N, device=device) 
        selected_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        
        # We run the loop for max_T iterations.
        # Efficient Trick: Just select max_T for everyone.
        # Later, in create_output_masks, we will mask out the extras based on token_quotas.

        for t in range(max_T):
            # 1. 扩展维度
            cur_max_sim_expanded = current_max_sim.unsqueeze(1) # (B, 1, N)
            
            # 2. 计算 Coverage & Gain
            new_coverage = torch.maximum(sim_matrix, cur_max_sim_expanded)
            gains_per_token = new_coverage - cur_max_sim_expanded
            
            # 3. 加权求和
            weighted_gains = gains_per_token * importance.unsqueeze(1)
            total_gain = weighted_gains.sum(dim=2) # (B, N) candidates
            
            # 4. Mask 掉已选
            total_gain[selected_mask] = -1e9
            
            # 5. 贪心选择
            best_idx = torch.argmax(total_gain, dim=1) # (B,)
            
            # 6. 记录
            select_idx[:, t] = best_idx
            
            # 7. 更新状态
            batch_indices = torch.arange(B, device=device)
            chosen_sim_row = sim_matrix[batch_indices, best_idx, :] # (B, N)
            current_max_sim = torch.maximum(current_max_sim, chosen_sim_row)
            selected_mask[batch_indices, best_idx] = True

        return select_idx, token_quotas
    

    # [EADP]
    def encode_images(self, images, texts=None, split_sizes=None):

        visual_token_num = getattr(self, 'visual_token_num', 0)

        # Bypass pruning when visual_token_num <= 0 (baseline / no pruning)
        if visual_token_num <= 0:
            image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)
            B, N, _ = image_features.shape
            index_masks = torch.ones(
                B, N, dtype=torch.bool, device=image_features.device)
            return image_features, index_masks

        alpha = getattr(self, 'alpha', 0.5)
        beta = getattr(self, 'beta', 1.0)

        # -------------------------------------------------------------
        # (0) Extract image features + text features
        # -------------------------------------------------------------

        # 这里得到的 image_features 是 (B, N, D)
        # image_embeds 是用于计算相似度的 (B, N, C)
        image_features, image_embeds, text_embeds, text_embeds_seq = self.get_model().get_vision_tower()(images, texts=texts)
        B, N, C = image_features.shape
        M = text_embeds.shape[0]
        device = image_features.device
        # Projector 变换
        image_features = self.get_model().mm_projector(image_features)


        # Joint Logic
        self.is_joint = False
        original_B = B
        original_N = N
        
        # [Strategy Selection]
        # Options: 'joint', 'importance_based', 'global_guided'
        pruning_strategy = 'importance_based'

        # We need K regardless of strategy to know grouping
        K_joint, B_joint, N_joint, image_features_joint, image_embeds_joint = self.joint_reshape(B, N, C, image_features, image_embeds, split_sizes)
        K = K_joint

        if pruning_strategy == 'joint':
            # Use the reshaped (merged) features
            B, N = B_joint, N_joint
            image_features = image_features_joint
            image_embeds = image_embeds_joint
            # self.is_joint is already True from joint_reshape
        else:
            # Revert is_joint if it was set, as we process patches independently
            if self.is_joint:
                self.is_joint = False
            # B, N, image_features, image_embeds remain original (B*5, N)


        # -------------------------------------------------------------
        # (1) Compute Pairwise Visual Similarity
        # -------------------------------------------------------------

        sim_matrix = self.sim_visual(image_features)

        # -------------------------------------------------------------
        # (2) Compute Text-Visual Importance Scores
        # -------------------------------------------------------------

        # Note: sim_cross expects original_B, original_N for caching/logic.
        # If strategy='joint', B, N are modified. original_B, original_N preserve the raw shape.
        # If strategy!='joint', B=original_B, N=original_N (roughly, unless K=1).
        
        global_sim, local_sim_all = self.sim_cross(text_embeds, text_embeds_seq, image_embeds, original_B, original_N)

        # (2. 1) Entropy Filtering
        filtered_local_sim, top_entropy_vals = self.entrpy_filter(local_sim_all, T = 100.0, entropy_keep_ratio=0.2)

        # (2. 2) Local Aggregation
        local_sim = self.local_aggregation(filtered_local_sim=filtered_local_sim, top_entropy_vals=top_entropy_vals, M_temp=0.01, strategy="negative_entropy")

        # (2. 3) Fusion
        text_sim_all = alpha * global_sim + (1 - alpha) * local_sim # (B, N, M)
        text_sim = text_sim_all.mean(dim=-1) # (B, N)

        # (2. 4) Min-Max Normalize
        text_sim = text_sim.view(B_joint, N_joint) # 为了保持全局归一化，这里先恢复到 joint 形状

        importance = (text_sim - text_sim.min(dim=-1, keepdim=True).values + 1e-6) / \
                     (text_sim.max(dim=-1, keepdim=True).values - text_sim.min(dim=-1, keepdim=True).values + 1e-6)

        importance = importance.view(B, N)

        # (2. 5) Spatial Smoothing
        importance = self.spatial_smoothing(importance, kernel_size = 3, sigma = 1.0)

        # (2. 6) Power Transform
        importance = importance ** beta

        # -------------------------------------------------------------
        # (3) Submodular Maximization (Facility Location)
        # -------------------------------------------------------------

        select_idx, token_quotas = self.greed_select(importance, sim_matrix, K=K, strategy=pruning_strategy)

        # -------------------------------------------------------------
        # (4) Create Output Masks
        # -------------------------------------------------------------
        
        # select_idx: (B, max_T)
        # token_quotas: (B,)
        
        # Create validity mask based on quotas
        # shape: (B, max_T)
        device = select_idx.device
        range_mask = torch.arange(select_idx.shape[1], device=device).unsqueeze(0) < token_quotas.unsqueeze(1)
        src = range_mask.float()
        
        index_masks = torch.zeros(B, N, dtype=torch.float32, device=device)
        index_masks.scatter_(1, select_idx, src)
        index_masks = index_masks.bool()

        if self.is_joint:
            # Restore original shape (Total_Patches, Original_N)
            image_features = image_features.view(original_B, original_N, -1)
            index_masks = index_masks.view(original_B, original_N)

        return image_features, index_masks

    # Prune visual tokens according to index masks
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, texts=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Prune visual tokens
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            
            # Calculate split_sizes before encoding for joint pruning support
            split_sizes = [image.shape[0] for image in images]
            
            concat_images = torch.cat([image for image in images], dim=0)
            image_features, index_masks = self.encode_images(concat_images, texts=texts, split_sizes=split_sizes)
            
            image_features = torch.split(image_features, split_sizes, dim=0)
            index_masks = torch.split(index_masks, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            mm_patch_merge_type = mm_patch_merge_type.replace('_unpad', '')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                index_masks = [x.flatten(0, 1) for x in index_masks]
                image_features = [x[m] for x, m in zip(image_features, index_masks)]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, (image_feature, index_mask) in enumerate(zip(image_features, index_masks)):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        base_index_mask = index_mask[0]
                        index_mask = index_mask[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            index_mask = index_mask.view(num_patch_height, num_patch_width, height, width)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous().unsqueeze(0)
                            index_mask = index_mask.flatten(1, 2).flatten(2, 3)
                            index_mask = unpad_image(index_mask, image_sizes[image_idx])
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(*index_mask.shape[:-1], 1, dtype=torch.bool).to(index_mask.device)
                            ), dim=-1)
                            index_mask = index_mask.flatten(1, 2).squeeze(0)
                            image_feature = image_feature[index_mask]
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous()
                            index_mask = index_mask.flatten(0, 3)
                            image_feature = image_feature[index_mask]
                        base_image_feature = base_image_feature[base_index_mask]
                        image_feature = torch.cat((base_image_feature, image_feature))
                    else:
                        image_feature = image_feature[0]
                        index_mask = index_mask[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(1, dtype=torch.bool).to(index_mask.device)
                            ), dim=0)
                        image_feature = image_feature[index_mask]
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features, index_masks = self.encode_images(images, texts=texts)
            image_features = image_features[index_masks].unsqueeze(0)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_features[0].shape[0]

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
