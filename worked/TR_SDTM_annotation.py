import torch
from typing import Tuple, Callable
import math
from typing import Type, Dict, Any, Tuple, Callable
import torch.nn.functional as F
from diffusers.models.attention import _chunked_feed_forward, Attention
import random
import time
from tabulate import tabulate
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
import os
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


def do_nothing(x: torch.Tensor, mode:str=None):
    return x

# 适配所有系统的torch.gather操作
def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

def SSM(
    metric: torch.Tensor,
    reduce_num: int = 0,
    threshold: float = 0,
    window_size: Tuple[int, int] = (4,4),
    no_rand: bool = False,
    generator: torch.Generator = None,
    tore_info: Dict = None
) -> Tuple[Callable, Callable]:
    if reduce_num <= 0:
        return do_nothing, do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        '''
        ws_h, ws_w: window的高和宽, 以token为单位
        stride_h, stride_w: window的垂直和水平步长, 以token为单位,默认为窗口大小，即不重叠
        num_token_window:window内的token数量,即窗口大小
        metric: [B, N, D], B是batch size, N是token数量, D是特征维度, metric是用来计算相似度的特征表示, metric其实就是input
        base_grid_H, base_grid_W: 将整个输入视为二维网格, base_grid_H是网格的高度, 以token为单位, 这里假设输入是正方形的, 所以base_grid_W=base_grid_H
        '''
        ws_h, ws_w = int(window_size[0]), int(window_size[1])
        stride_h, stride_w = ws_h, ws_w
        num_token_window = stride_h * stride_w
        assert num_token_window > 1, "window_size must produce at least 2 tokens (K>1)."
        
        B, N, D = metric.size()
        base_grid_H = int(math.sqrt(N))
        base_grid_W = base_grid_H
        assert base_grid_H * base_grid_W == N and base_grid_H % ws_h == 0 and base_grid_W % ws_w == 0

        # 改变输入形状, 先变成正方形的网格，并为卷积做准备[B, H, W, D] -> [B, D, H, W], 这里D就是channel
        metric = metric.view(B, base_grid_H, base_grid_W, D).permute(0, 3, 1, 2)
    
        # [B, D, H, W] -> [B, D, H//ws_h, ws_h, W//ws_w, ws_w] -> [B, H//ws_h, W//ws_w, D, ws_h, ws_w]
        metric = metric.view(B, D, base_grid_H // ws_h, ws_h, base_grid_W // ws_w, ws_w).permute(0, 2, 4, 1, 3, 5)
        '''
        b:Batch size
        gh: 垂直方向的window数量
        gw: 水平方向的window数量
        c: 特征维度, 即channel数
        ps_h: window的高度
        ps_w: window的宽度
        tensor_flattened: [B, gh, gw, c, ps_h*ps_w], 将每个window内的token展平以便后续计算相似度
        tensor_1, tensor_2: [B, gh, gw, c, ps_h*ps_w, 1] 和 [B, gh, gw, c, 1, ps_h*ps_w], 用于计算窗口内token之间的两两相似度
        sims: [B, gh, gw, ps_h*ps_w, ps_h*ps_w], 窗口内token之间的两两余弦相似度矩阵
        similarity_map: [B, gh, gw]->[B, gh*gw], 所有window的相似度得分, 用于后续选择相似的window进行merge
        ssmscore_map: [B, gh*gw], 展平后的相似度得分, 方便后续topk操作选择相似的window进行merge
        tore_info: 这个参数是为了支持SDTM而设计的, 里面可以存储一些状态信息和参数, 比如每个token被选为independent的次数, 以及一些超参数等(token reduction)
        li: last_independent的缩写, 记录了每个token被选为independent的次数(int类型)
        li_f: 将li转换为float类型以便计算优先级
        '''
        b, gh, gw, c, ps_h, ps_w = metric.shape

        # Flatten mxm window for pairwise operations
        tensor_flattened = metric.reshape(b, gh, gw, c, -1)
    
        # Expand dims for pairwise operations
        tensor_1 = tensor_flattened.unsqueeze(-1)
        tensor_2 = tensor_flattened.unsqueeze(-2)

        # Compute cosine similarities
        sims = F.cosine_similarity(tensor_1, tensor_2, dim=3)

        # Average similarities (including self-similarity)
        similarity_map = sims.sum(-1).sum(-1) / ((ps_h * ps_w) * (ps_h * ps_w))
        # Variance score map: variance越小，分数越高(越应merge)
        variance_raw = sims.var(dim=(-1, -2), unbiased=False)

        similarity_map = similarity_map.unsqueeze(1).reshape(similarity_map.size(0), -1)
        variance_raw = variance_raw.unsqueeze(1).reshape(variance_raw.size(0), -1)

        eps = 1e-6
        v_min = variance_raw.amin(dim=1, keepdim=True)
        v_max = variance_raw.amax(dim=1, keepdim=True)
        variance_map = 1.0 - (variance_raw - v_min) / (v_max - v_min + eps)
            
        # ---- Frequency priority score integration ----
        indiv_priority_flat = torch.zeros_like(similarity_map)
        if tore_info is not None and "states" in tore_info and tore_info["states"].get("last_independent") is not None:
            li = tore_info["states"]["last_independent"]  # [B, N]
            if li.shape[1] == base_grid_H * base_grid_W:
                eps = 1e-6
                li_f = li.to(similarity_map.dtype)
                mean_li = li_f.mean(dim=1, keepdim=True) + eps
                indiv_priority = li_f / mean_li  # [B, N]
                indiv_priority_grid = indiv_priority.view(B, base_grid_H, base_grid_W)
                indiv_priority_windows = (
                    indiv_priority_grid
                    .view(B, gh, ws_h, gw, ws_w)
                    .permute(0, 1, 3, 2, 4)
                    .reshape(B, gh, gw, ws_h * ws_w)
                    .mean(-1)
                )  # [B, gh, gw], 得到每个window的合并优先级
                indiv_priority_flat = indiv_priority_windows.view(B, gh * gw)
        a_s = tore_info.get("args", {}).get("a_s", 0.0) if tore_info is not None else 0.0
        ssmscore_map = similarity_map + a_s * indiv_priority_flat + variance_map
        # ----------------------------------------------
        '''
        ssmscore_map:[B, gh*gw], 每个window的相似度得分(包含了frequency priority的加成)
        reduce_num: 需要merge的window数量, 可以通过相似度得分和threshold动态决定, 也可以直接指定一个固定数量
        '''
        # --- creating the mergable and unmergable super patches
        tensor = torch.arange(base_grid_H * base_grid_W, device=metric.device).reshape(base_grid_H, base_grid_W)

        # Repeat the tensor to create a batch of size 2
        tensor = tensor.unsqueeze(0).repeat(B, 1, 1)
        
        # Apply unfold operation on last two dimensions to create the sliding window
        windowed_tensor = tensor.unfold(1, ws_h, stride_h).unfold(2, ws_w, stride_w)

        # Reshape the tensor to the desired shape
        windowed_tensor = windowed_tensor.reshape(B, -1, num_token_window)
        num_windows = windowed_tensor.shape[1]
        K = num_token_window
        assert K >= 4, "window_size建议为4x4(16 tokens)以启用多级融合。"

        # 窗口级特征: [B, num_windows, K, c]，用于把src分配给最近dst
        window_feat = tensor_flattened.permute(0, 1, 2, 4, 3).reshape(B, num_windows, K, c)

        # 复杂度阈值(variance_map分数: 越大越低复杂度)
        low_thr = 0.66
        high_thr = 0.33
        if tore_info is not None:
            args_cfg = tore_info.get("args", {})
            low_thr = float(args_cfg.get("low_complexity_threshold", low_thr))
            high_thr = float(args_cfg.get("high_complexity_threshold", high_thr))
        # 保证low_thr >= high_thr
        low_thr, high_thr = max(low_thr, high_thr), min(low_thr, high_thr)

        # class: 0=低复杂度(16->1), 1=中复杂度(16->2), 2=高复杂度(16->4)  -1=不merge(unm)
        selected_class = torch.full((B, num_windows), -1, dtype=torch.long, device=metric.device)

        # dst_keep per class: 0->1, 1->2, 2->4; src_num per class: K-1, K-2, K-4
        # For vectorized construction we handle each (class, keep_n) tier separately.
        # To guarantee identical sequence lengths across the batch, we compute the
        # *minimum* window count for each tier across batch items and use that fixed count.

        if reduce_num is None:
            # (2.1) adaptive版本: 按ssmscore阈值选window，再按variance_map阈值分档
            cand_mask = ssmscore_map >= float(threshold)
            low_mask  = cand_mask & (variance_map >= low_thr)
            mid_mask  = cand_mask & (variance_map < low_thr) & (variance_map >= high_thr)
            high_mask = cand_mask & (variance_map < high_thr)

            # 取各档在batch内的最小数量以保证长度一致（向量化 topk）
            keep_low  = int(low_mask.sum(dim=1).min().item())
            keep_mid  = int(mid_mask.sum(dim=1).min().item())
            keep_high = int(high_mask.sum(dim=1).min().item())

            # 使用 score * mask - 大负数 对不合格项压制，再 topk 取 keep_xxx 个，纯 GPU 操作
            NEG_INF = -1e9
            if keep_low > 0:
                score_low = ssmscore_map.masked_fill(~low_mask, NEG_INF)
                chosen_low = score_low.topk(keep_low, dim=-1).indices      # [B, keep_low]
                selected_class.scatter_(1, chosen_low,
                    torch.zeros(B, keep_low, dtype=torch.long, device=metric.device))
            if keep_mid > 0:
                score_mid = ssmscore_map.masked_fill(~mid_mask, NEG_INF)
                chosen_mid = score_mid.topk(keep_mid, dim=-1).indices
                selected_class.scatter_(1, chosen_mid,
                    torch.ones(B, keep_mid, dtype=torch.long, device=metric.device))
            if keep_high > 0:
                score_high = ssmscore_map.masked_fill(~high_mask, NEG_INF)
                chosen_high = score_high.topk(keep_high, dim=-1).indices
                selected_class.scatter_(1, chosen_high,
                    torch.full((B, keep_high), 2, dtype=torch.long, device=metric.device))
        else:
            # (2.2) 固定参数版本: 指定reduce_num，按variance_map的相对大小对选出window分25%/25%/50%三档
            r = min(int(reduce_num), num_windows)
            if r > 0:
                _, sim_super_patch_idxs = ssmscore_map.topk(r, dim=-1)   # [B, r]
                low_n  = int(r * 0.25)
                mid_n  = int(r * 0.25)
                high_n = r - low_n - mid_n

                # 取每个batch对应选出窗口的variance分数，按降序排列（高分=低复杂度在前）
                sel_var = variance_map.gather(1, sim_super_patch_idxs)    # [B, r]
                order   = sel_var.argsort(dim=-1, descending=True)        # [B, r]
                sorted_idxs = sim_super_patch_idxs.gather(1, order)       # [B, r] 已按复杂度排序

                if low_n > 0:
                    selected_class.scatter_(1, sorted_idxs[:, :low_n],
                        torch.zeros(B, low_n, dtype=torch.long, device=metric.device))
                if mid_n > 0:
                    selected_class.scatter_(1, sorted_idxs[:, low_n:low_n + mid_n],
                        torch.ones(B, mid_n, dtype=torch.long, device=metric.device))
                if high_n > 0:
                    selected_class.scatter_(1, sorted_idxs[:, low_n + mid_n:],
                        torch.full((B, high_n), 2, dtype=torch.long, device=metric.device))

        # -----------------------------------------------------------------------
        # 向量化构建 unm/src/dst/merge_idx，消除 Python for 循环
        # dst_keep per class: {0:1, 1:2, 2:4}
        # 对每一档 (cls_id, keep_n) 独立处理，最终拼接
        # -----------------------------------------------------------------------
        # 预先为每个窗口生成随机 dst 位置（全部 GPU 上完成，无 Python 循环）
        # 为 keep_n=1 生成 1 个随机位置；keep_n=2 生成 2 个；keep_n=4 生成 4 个
        # 使用 rand().argsort() 代替 randperm，可在 batch+window 维度一次生成

        # 预计算归一化窗口特征，用于 keep_n>1 时的 src->dst 分配
        wf_norm = window_feat / (window_feat.norm(dim=-1, keepdim=True) + 1e-6)  # [B, W, K, c]

        # 全局随机排列: [B, num_windows, K]，每个窗口内 K 个位置的随机顺序
        if no_rand:
            rand_order = torch.arange(K, device=metric.device).view(1, 1, K).expand(B, num_windows, K)
        else:
            rand_noise = torch.rand(B, num_windows, K, device=metric.device, generator=generator)
            rand_order = rand_noise.argsort(dim=-1)   # [B, num_windows, K]

        # windowed_tensor: [B, num_windows, K] — token id 表
        # 按 rand_order 重排 token ids，前 keep_n 个是 dst，其余是 src
        shuffled_tokens = windowed_tensor.gather(2, rand_order)  # [B, num_windows, K]

        all_unm_idx_list = []
        all_src_idx_list = []
        all_dst_idx_list = []
        all_merge_idx_list = []

        dst_cum = 0  # dst_idx 在全局 dst 列表中的累计偏移（用于 merge_idx 的值）

        for cls_id, keep_n in [(0, 1), (1, 2), (2, 4)]:
            src_n = K - keep_n
            cls_mask = (selected_class == cls_id)          # [B, num_windows], bool
            # 保证 batch 内每档窗口数相同（已由上面构建逻辑保证），取最小以防万一
            cls_cnt_per_batch = cls_mask.sum(dim=1)        # [B]
            n_cls = int(cls_cnt_per_batch.min().item())
            if n_cls == 0:
                all_src_idx_list.append(None)
                all_dst_idx_list.append(None)
                all_merge_idx_list.append(None)
                continue

            # 取各 batch 中该类的前 n_cls 个窗口（按窗口 id 升序，稳定）
            # score 替换成 cls_mask 的 float，用 topk 取 n_cls 个窗口索引
            cls_win_idx = cls_mask.float().topk(n_cls, dim=-1).indices   # [B, n_cls]

            # 取这些窗口的 shuffled token: [B, n_cls, K]
            tokens_cls = shuffled_tokens.gather(
                1, cls_win_idx.unsqueeze(-1).expand(B, n_cls, K))

            # dst: 前 keep_n 列；src: 后 src_n 列
            dst_tokens = tokens_cls[:, :, :keep_n]    # [B, n_cls, keep_n]
            src_tokens = tokens_cls[:, :, keep_n:]    # [B, n_cls, src_n]

            # merge_idx: 将每个 src 分配到最近的 dst（相对于当前档 dst 的局部偏移）
            if keep_n == 1:
                # 唯一 dst，所有 src 都分配到 0
                merge_local = torch.zeros(B, n_cls, src_n, dtype=torch.long, device=metric.device)
            else:
                # [B, n_cls, src_n, c] 与 [B, n_cls, keep_n, c] 的余弦相似度
                # wf_norm 按 cls_win_idx 取出对应窗口
                wf_cls = wf_norm.gather(
                    1, cls_win_idx.unsqueeze(-1).unsqueeze(-1).expand(B, n_cls, K, c))  # [B, n_cls, K, c]
                # 按 rand_order 重排 wf_cls 使之与 shuffled_tokens 对齐
                rand_cls = rand_order.gather(
                    1, cls_win_idx.unsqueeze(-1).expand(B, n_cls, K))   # [B, n_cls, K]
                wf_cls = wf_cls.gather(
                    2, rand_cls.unsqueeze(-1).expand(B, n_cls, K, c))   # [B, n_cls, K, c]
                src_feat = wf_cls[:, :, keep_n:, :]   # [B, n_cls, src_n, c]
                dst_feat = wf_cls[:, :, :keep_n, :]   # [B, n_cls, keep_n, c]
                # sim: [B, n_cls, src_n, keep_n]
                sim_sd = torch.einsum('bwsc,bwdc->bwsd', src_feat, dst_feat)
                merge_local = sim_sd.argmax(dim=-1)    # [B, n_cls, src_n]

            # merge_idx 的值要指向全局 dst 列表中的位置
            # 全局偏移 = dst_cum + 窗口在当前档内的序号 * keep_n + 局部 dst 位置
            win_base = torch.arange(n_cls, device=metric.device).view(1, n_cls, 1) * keep_n  # [1, n_cls, 1]
            merge_global = dst_cum + win_base + merge_local    # [B, n_cls, src_n]

            # 展平 src/dst/merge
            dst_flat   = dst_tokens.reshape(B, n_cls * keep_n).unsqueeze(-1)    # [B, n_cls*keep_n, 1]
            src_flat   = src_tokens.reshape(B, n_cls * src_n).unsqueeze(-1)     # [B, n_cls*src_n, 1]
            merge_flat = merge_global.reshape(B, n_cls * src_n).unsqueeze(-1)   # [B, n_cls*src_n, 1]

            all_src_idx_list.append(src_flat)
            all_dst_idx_list.append(dst_flat)
            all_merge_idx_list.append(merge_flat)

            dst_cum += n_cls * keep_n

        # unm: 所有 selected_class == -1 的窗口的全部 K 个 token
        # 同样向量化：取 unm 窗口数的最小值
        unm_mask = (selected_class == -1)    # [B, num_windows]
        unm_cnt  = unm_mask.sum(dim=1)
        n_unm    = int(unm_cnt.min().item())
        if n_unm > 0:
            unm_win_idx = unm_mask.float().topk(n_unm, dim=-1).indices    # [B, n_unm]
            unm_tokens  = windowed_tensor.gather(
                1, unm_win_idx.unsqueeze(-1).expand(B, n_unm, K))         # [B, n_unm, K]
            unm_idx = unm_tokens.reshape(B, n_unm * K).unsqueeze(-1)      # [B, n_unm*K, 1]
        else:
            unm_idx = torch.zeros(B, 0, 1, device=metric.device, dtype=torch.long)

        # 拼接各档 src/dst/merge
        valid_src   = [x for x in all_src_idx_list   if x is not None]
        valid_dst   = [x for x in all_dst_idx_list   if x is not None]
        valid_merge = [x for x in all_merge_idx_list if x is not None]

        src_idx   = torch.cat(valid_src,   dim=1) if valid_src   else torch.zeros(B, 0, 1, device=metric.device, dtype=torch.long)
        dst_idx   = torch.cat(valid_dst,   dim=1) if valid_dst   else torch.zeros(B, 0, 1, device=metric.device, dtype=torch.long)
        merge_idx = torch.cat(valid_merge, dim=1) if valid_merge else torch.zeros(B, 0, 1, device=metric.device, dtype=torch.long)

        unm_len = unm_idx.shape[1]
        src_len = src_idx.shape[1]
        dst_len = dst_idx.shape[1]

        dim_index = src_len

        independent_idx = None
        unindependent_idx = None

        def update_last_independent(independent_indices: torch.Tensor):
            if tore_info["states"].get("last_independent") is None:
                tore_info["states"]["last_independent"] = torch.zeros(B, N, device=independent_indices.device, dtype=torch.int32)
            last_ind = tore_info["states"]["last_independent"]
            last_ind.add_(1)
            zeros_src = torch.zeros_like(independent_indices, dtype=last_ind.dtype)
            last_ind.scatter_(1, independent_indices, zeros_src)

        if tore_info and tore_info.get("args", {}).get("pseudo_merge", False):
            independent_idx = torch.cat([unm_idx.squeeze(-1), dst_idx.squeeze(-1)], dim=-1)
            unindependent_idx = src_idx.squeeze(-1)
        else:
            independent_idx = unm_idx.squeeze(-1)
            unindependent_idx = torch.cat([src_idx.squeeze(-1), dst_idx.squeeze(-1)], dim=-1)
        update_last_independent(independent_idx)

        '''
        将window内要merge的token(src)融合进基石dst里
        n: batch size
        t1: 当前输入的token数量, 即N
        c: 特征维度, 即channel数
        src: [B, reduce_num*dim_index, c], 被选中进行merge的window内被选为src的token特征, 通过torch.gather操作得到
        dst: [B, reduce_num, c], 被选中进行merge的window内被选为dst的token特征, 通过torch.gather操作得到
        unm: [B, (num_window - reduce_num)*num_token_window, c], 没有被选中进行merge的token特征, 通过torch.gather操作得到
        dst: [B, reduce_num*dim_index, c], 融合src后的dst特征, 通过scatter_reduce操作得到, 将src的特征按照merge_idx指定的位置融合到dst上, mode指定了融合的方式(比如mean, sum等)
        '''
        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # TODO: num_token_window can be undefined
        
            n, t1, c = x.shape
            # src = x.gather(dim=-2, index=src_idx.expand(n, r*dim_index, c))
            # dst = x.gather(dim=-2, index=dst_idx.expand(n, r, c))
            # unm = x.gather(dim=-2, index=unm_idx.expand(n, t1 - (r*num_token_window), c))
            src = gather(x, dim=-2, index=src_idx.expand(n, src_len, c))
            dst = gather(x, dim=-2, index=dst_idx.expand(n, dst_len, c))
            unm = gather(x, dim=-2, index=unm_idx.expand(n, unm_len, c))
            if src_len > 0:
                dst = dst.scatter_reduce(-2, merge_idx.expand(n, src_len, c), src, reduce=mode)
            x = torch.cat([unm, dst], dim=1)

            return x
        '''
        直接丢弃要merge的token, 只保留不需要merge的window里的token(unm)和应该merge的window里的基石token(dst)
        所以叫prune, 而不是merge
        '''
        def mprune(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # TODO: num_token_window can be undefined
            n, t1, c = x.shape

            dst = gather(x, dim=-2, index=dst_idx.expand(n, dst_len, c))
            unm = gather(x, dim=-2, index=unm_idx.expand(n, unm_len, c))
            x = torch.cat([unm, dst], dim=1)

            return x

        '''
        将压缩结果还原为原布局(保持分辨率一致)
        x: 完成了merge或者prune的输入特征, 包含unm和dst两部分
        tu: 没有被merge的token数量, 即unm的token数量
        mcw: merge confidence weight, 在进行merge时可以选择性地融合被merge的token(src)和基石token(dst)的特征, 而不是完全用src覆盖dst, mcw控制了融合的权重, 例如mcw=0.5表示src和dst各占一半
        l: 当前所在的层数, 可以用来从tore_info里获取不同层的缓存特征进行融合
        '''
        def unmerge(x: torch.Tensor) -> torch.Tensor:
            # Determine cache_name directly from phase state; align with features dict keys
           

            unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
            _, tu, c = unm.shape
            # Reconstruct src from dst buckets (value copy, not true inverse)
            src = gather(dst, dim=-2, index=merge_idx.expand(B, src_len, c)) if src_len > 0 else dst[:, :0, :]

            # Optional fusion with cached features for dst/src tokens
            # unm: keep as-is; dst/src: fuse computed with cached by mcw
            try:
                cache_name = tore_info["states"].get("unmerge_phase", None)
                if cache_name not in ("attn_output", "attn_output2", "mlp_output"):
                    pass
                mcw = float(tore_info.get("args", {}).get("mcw", 1.0))
                l = tore_info.get("states", {}).get("layer_current", -1)
                key = f"l{l}"
                cache_dict = tore_info.get("features", {}).get(cache_name, None)
                cache_full = cache_dict.get(key) if isinstance(cache_dict, dict) else None
                if cache_full is not None:
                    cache_full = cache_full.to(device=x.device, dtype=x.dtype)
                    # gather cached dst/src by original indices
                    cached_dst = gather(cache_full, dim=1, index=dst_idx.expand(B, dst_len, c))
                    cached_src = gather(cache_full, dim=1, index=src_idx.expand(B, src_len, c))
                    # fuse
                    dst = mcw * dst + (1.0 - mcw) * cached_dst
                    src = mcw * src + (1.0 - mcw) * cached_src
            except Exception:
                pass  # best-effort fusion; fallback to computed values

            # Combine back to the original shape
            out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
            # NOTE: src_idx is (src in x), dst_idx is (dst in x), unm_idx is (unm in x)
            out.scatter_(dim=-2, index=dst_idx.expand(B, dst_len, c), src=dst)
            out.scatter_(dim=-2, index=unm_idx.expand(B, tu, c), src=unm)
            if src_len > 0:
                out.scatter_(dim=-2, index=src_idx.expand(B, src_len, c), src=src)
            return out

    return merge, mprune, unmerge

# NOTE: Fake IDM
# Since xFormers does not support the explicit output of attention maps, FIDM removes the dependency on attention maps.
# Instead, within each window, we select the tokens with the highest frequency priority scores as the "attentive group", while the remaining tokens are assigned to the "inattentive group".
# This design enables effective separation without relying on attention outputs.
def FIDM(
        metric: torch.Tensor,
        reduce_num: int = 0,
        window_size: Tuple[int, int] = (2,2),
        no_rand: bool = False,
        generator: torch.Generator = None,
        tore_info: Dict = None
        ) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, C = metric.shape

    if reduce_num <= 0:
        return do_nothing, do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        sx, sy = int(window_size[0]), int(window_size[1])

        h = int(math.sqrt(N))
        w = h
        assert h * w == N and h % sy == 0 and w % sx == 0
        hsy, wsx =  h // sy, w // sx

        # Decide dst inside each (sy, sx) window: prefer max last_independent if available else random
        use_li = (
            tore_info is not None and "states" in tore_info and
            tore_info["states"].get("last_independent") is not None and
            tore_info["states"]["last_independent"].shape[1] == N
        )

        if use_li:
            li = tore_info["states"]["last_independent"]  # [B, N]
            li_grid = li.view(B, h, w)
            # Reshape to windows: [B, hsy, sy, wsx, sx] -> permute to [B, hsy, wsx, sy, sx]
            li_windows = li_grid.view(B, hsy, sy, wsx, sx).permute(0, 1, 3, 2, 4)
            # Flatten each window tokens: [B, hsy, wsx, sy*sx]
            li_flat_win = li_windows.reshape(B, hsy, wsx, sy * sx)
            # Argmax per window: returns index in [0, sy*sx)
            dst_pos = li_flat_win.argmax(dim=-1, keepdim=True)  # [B, hsy, wsx, 1]
        else:
            # Random fallback (per batch) to maintain original behavior when no last_independent
            if no_rand:
                dst_pos = torch.zeros(B, hsy, wsx, 1, device=metric.device, dtype=torch.int64)
            else:
                if generator is not None:
                    dst_pos = torch.randint(sy * sx, (B, hsy, wsx, 1), device=metric.device, generator=generator)
                else:
                    dst_pos = torch.randint(sy * sx, (B, hsy, wsx, 1), device=metric.device)

        # Build sentinel buffer: -1 at dst position, 0 elsewhere
        idx_buffer_view = torch.zeros(B, hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        neg_one = -torch.ones_like(dst_pos, dtype=idx_buffer_view.dtype)
        idx_buffer_view.scatter_(dim=3, index=dst_pos, src=neg_one)

        # Reshape to spatial (with same ordering as original single-batch code)
        # Original single batch: (hsy, wsx, sy, sx).transpose(1,2) -> (hsy, sy, wsx, sx)
        # Multi-batch adaption:
        idx_buffer_view = idx_buffer_view.view(B, hsy, wsx, sy, sx).transpose(2, 3).reshape(B, hsy * sy, wsx * sx)

        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(B, h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:, :(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # Argsort per batch to obtain dst|src partition indices
        rand_idx = idx_buffer.reshape(B, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        reduce_num = min(a.shape[1], reduce_num)
        reduce_num = reduce_num // 16 * 16 # ensure multiple of 16

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., reduce_num:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :reduce_num, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        # Expand original dst/src index tensors along batch for downstream mapping
        a_idx_b = a_idx.expand(B, a_idx.shape[1], 1)  # [B, N_src, 1]
        b_idx_b = b_idx.expand(B, b_idx.shape[1], 1)  # [B, num_dst, 1]

        # Store raw token indices (no channel expansion yet)
        dst_in_x_index = b_idx_b                       # [B, num_dst, 1]
        unm_in_x_index = gather(a_idx_b, dim=1, index=unm_idx)  # [B, a_len-reduce_num, 1]
        src_in_x_index = gather(a_idx_b, dim=1, index=src_idx)  # [B, reduce_num, 1]

        independent_idx = None
        unindependent_idx = None

        def update_last_independent(independent_indices: torch.Tensor):
            if tore_info["states"].get("last_independent") is None:
                tore_info["states"]["last_independent"] = torch.zeros(B, N, device=independent_indices.device, dtype=torch.int32)
            last_ind = tore_info["states"]["last_independent"]
            last_ind.add_(1)
            zeros_src = torch.zeros_like(independent_indices, dtype=last_ind.dtype)
            last_ind.scatter_(1, independent_indices, zeros_src)

        if tore_info and tore_info.get("args", {}).get("pseudo_merge", False):
            independent_idx = torch.cat([unm_in_x_index.squeeze(-1), dst_in_x_index.squeeze(-1)], dim=-1)
            unindependent_idx = src_in_x_index.squeeze(-1)
        else:
            independent_idx = unm_in_x_index.squeeze(-1)
            unindependent_idx = torch.cat([src_in_x_index.squeeze(-1), dst_in_x_index.squeeze(-1)], dim=-1)
        update_last_independent(independent_idx)

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            a, dst = split(x)
            n, t1, c = a.shape
            
            unm = gather(a, dim=-2, index=unm_idx.expand(n, t1 - reduce_num, c))
            src = gather(a, dim=-2, index=src_idx.expand(n, reduce_num, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, reduce_num, c), src, reduce=mode)

            return torch.cat([unm, dst], dim=1)

        def mprune(x: torch.Tensor) -> torch.Tensor:
            a, dst = split(x)
            n, t1, c = a.shape
            
            unm = gather(a, dim=-2, index=unm_idx.expand(n, t1 - reduce_num, c))

            return torch.cat([unm, dst], dim=1)
        
        def unmerge(x: torch.Tensor) -> torch.Tensor:
            unm_len = unm_idx.shape[1]
            unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
            _, _, c = unm.shape

            src = gather(dst, dim=-2, index=dst_idx.expand(B, reduce_num, c))

            # Optional fusion with cached features for dst/src tokens (unm remains as-is)
            try:
                # Determine cache_name directly from phase state; align with features dict keys
                cache_name = tore_info.get("states", {}).get("unmerge_phase", None)
                if cache_name not in ("attn_output", "attn_output2", "mlp_output"):
                    pass
                mcw = float(tore_info.get("args", {}).get("mcw", 1.0))
                l = tore_info.get("states", {}).get("layer_current", -1)
                key = f"l{l}"
                cache_dict = tore_info.get("features", {}).get(cache_name, None)
                cache_full = cache_dict.get(key) if isinstance(cache_dict, dict) else None
                if cache_full is not None:
                    cache_full = cache_full.to(device=x.device, dtype=x.dtype)
                    # Gather cached values by original positions
                    cached_dst = gather(cache_full, dim=1, index=dst_in_x_index.expand(B, dst_in_x_index.shape[1], c))
                    cached_src = gather(cache_full, dim=1, index=src_in_x_index.expand(B, src_in_x_index.shape[1], c))
                    # fuse
                    dst = mcw * dst + (1.0 - mcw) * cached_dst
                    src = mcw * src + (1.0 - mcw) * cached_src
            except Exception:
                pass  # best-effort fusion; fallback to computed values

            # Combine back to the original shape
            out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
            # NOTE: a_idx is (a in x) b_idx is (dst in x), 
            # NOTE: dst_idx is (src in dst), unm_idx is (unm in a), (src_idx) is (src in a)

            out.scatter_(dim=-2, index=dst_in_x_index.expand(B, dst_in_x_index.shape[1], c), src=dst)
            out.scatter_(dim=-2, index=unm_in_x_index.expand(B, unm_in_x_index.shape[1], c), src=unm)
            out.scatter_(dim=-2, index=src_in_x_index.expand(B, src_in_x_index.shape[1], c), src=src)
            return out
    
    return merge, mprune, unmerge

# 不用引入具体的class，即可判断一个对象是否是某个类的实例（或子类的实例），更甚可以引入猴子补丁
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False

def init_generator(device: torch.device, fallback: torch.Generator=None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback

def compute_ratio(tore_info: Dict[str, Any]) -> float:
    """Compute ratio_current using values stored in `tore_info`.

    The function extracts the following fields from tore_info:
      - ratio, deviation
      - step_current, step_count
      - layer_current, layer_count
      - protect_steps_frequency, protect_layers_frequency

    If a protect frequency is negative (e.g. -1) it is treated as disabled.
    Returns 0.0 when current step or layer is protected; otherwise returns linear
    interpolation from ratio+deviation (step=0) to ratio-deviation (step=step_count-1).
    """
    args = tore_info.get("args", {})
    states = tore_info.get("states", {})

    ratio = args.get("ratio", 0.5)
    deviation = args.get("deviation", 0.2)
    step_current = states.get("step_current", 0)
    step_count = states.get("step_count", 1)
    layer_current = states.get("layer_current", 0)
    layer_count = states.get("layer_count", 1)
    # NOTE: protect_steps_frequency no longer affects ratio scheduling. It is now
    # handled inside the transformer block forward pass to completely bypass
    # merge/unmerge logic for protected steps instead of forcing ratio=0 here.
    protect_steps_frequency = args.get("protect_steps_frequency", None)  # kept for backward compat, unused below
    protect_layers_frequency = args.get("protect_layers_frequency", None)

    def is_protected(idx, total, freq):
        # Treat None or negative frequency as disabled
        if freq is None or freq < 0:
            return False
        # frequency == 0 is invalid -> treat as disabled
        if freq == 0:
            return False
        if idx % freq == 0:
            return True
        if idx == max(total - 1, 0):
            return True
        return False

    # Only layer protection can still zero out ratio here; step protection is handled in block forward.
    if is_protected(layer_current, layer_count, protect_layers_frequency):
        tore_info["states"]["last_independent"] = None
        return 0.0

    if step_count > 1:
        progress = step_current / (step_count - 1)
    else:
        progress = 0.0

    # Use a cosine-shaped descent to go from (ratio + deviation) -> (ratio - deviation).
    # We map progress in [0,1] to alpha = cos(progress * pi/2), which goes 1 -> 0.
    # Then interpolate: value = (ratio - deviation) + 2*deviation*alpha
    alpha = math.cos(progress * math.pi / 2)
    ratio_current = (ratio - deviation) + (2.0 * deviation) * alpha

    return float(ratio_current)

def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a SDTM Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model.transformer if hasattr(model, "transformer") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tore_info"):
            for hook in module._tore_info["hooks"]:
                hook.remove()
            module._tore_info["hooks"].clear()

        if module.__class__.__name__ == "SDTMBlock":
            module.__class__ = module._parent
    
    return model
'''
m_a:merge_attn, 表示是否在注意力模块进行token merge
m_m:merge_mlp, 表示是否在MLP模块进行token merge
u_a:unmerge_attn, 表示是否在注意力模块进行token unmerge
u_m:unmerge_mlp, 表示是否在MLP模块进行token unmerge
'''
def compute_merge(x: torch.Tensor, tore_info: Dict[str, Any]) -> Tuple[Callable, ...]:

    w = int(math.sqrt(x.shape[1]))
    h = w
    assert w * h == x.shape[1], "Input must be square"

    # 使用模块级 compute_ratio
    ratio_current = compute_ratio(tore_info)
    tore_info["states"]["ratio_current"] = ratio_current

    reduce_num = int(x.shape[1] * ratio_current)
    if reduce_num <= 0:
        return do_nothing, do_nothing, do_nothing, do_nothing

    # Re-init the generator if it hasn't already been initialized or device has changed.
    if tore_info["args"]["generator"] is None:
        tore_info["args"]["generator"] = init_generator(x.device)
    elif tore_info["args"]["generator"].device != x.device:
        tore_info["args"]["generator"] = init_generator(x.device, fallback=tore_info["args"]["generator"])

    # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
    # batch, which causes artifacts with use_rand, so force it to be off.
    use_rand = False if x.shape[0] % 2 == 1 else tore_info["args"]["use_rand"]

    # Choose strategy based on switch_step: SSM when step_current <= switch_step, otherwise IDM
    step_current = tore_info["states"].get("step_current", 0)
    switch_step = tore_info["args"].get("switch_step", 20)
    if step_current <= switch_step:
        adaptive_ssm = tore_info["args"].get("adaptive_ssm", False)
        ssm_threshold = float(tore_info["args"].get("ssm_threshold", 0.0))
        m, pm, u  = SSM(
            metric=x,
            reduce_num=None if adaptive_ssm else reduce_num,
            threshold=ssm_threshold,
            window_size=(tore_info["args"]["sx"], tore_info["args"]["sy"]),
            no_rand=not use_rand,
            generator=tore_info["args"]["generator"],
            tore_info=tore_info,
        )
    else:
        m, pm, u  = FIDM(metric=x, reduce_num=reduce_num, window_size=(tore_info["args"]["sx"], tore_info["args"]["sy"]),  
                        no_rand=not use_rand, generator=tore_info["args"]["generator"], tore_info=tore_info)

    if tore_info["args"]["pseudo_merge"]:
        m_a, u_a = (pm, u) if tore_info["states"]["merge_attn"]==True else (do_nothing, do_nothing)
        m_m, u_m = (pm, u) if tore_info["states"]["merge_mlp"]==True  else (do_nothing, do_nothing)
    else:
        m_a, u_a = (m, u) if tore_info["states"]["merge_attn"]==True else (do_nothing, do_nothing)
        m_m, u_m = (m, u) if tore_info["states"]["merge_mlp"]==True  else (do_nothing, do_nothing)

    return m_a, m_m, u_a, u_m

'''
每次生成前，重置并准备 SDTM 的全局状态。
'''
def make_SDTM_pipe(pipe_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class StableDiffusion3Pipeline_SDTM(pipe_class):
        # Save for unpatching later
        _parent = pipe_class

        def __call__(self, *args, **kwargs):
            self._tore_info["states"]["step_count"] = kwargs['num_inference_steps']
            self._tore_info["states"]["step_iter"] = list(range(kwargs['num_inference_steps']))
            self._tore_info["states"]["last_independent"] = None
            output = super().__call__(*args, **kwargs)
            return output

    return StableDiffusion3Pipeline_SDTM
'''
每个 step 决定策略开关
'''
def make_SDTM_model(model_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    
    class SD3Transformer2DModel_SDTM(model_class):
        _parent = model_class

        def forward(self, *args, **kwargs):
            self._tore_info["states"]["layer_count"] = self.config.num_layers
            # pop next step; wrapper installation is handled once in apply_SDTM to avoid per-forward overhead
            self._tore_info["states"]["step_current"] = self._tore_info["states"]["step_iter"].pop(0)
            self._tore_info["states"]["layer_iter"] = list(range(self.config.num_layers))
            if self._tore_info["states"]["step_current"] <= self._tore_info["args"]["switch_step"]:
                self._tore_info["states"]["merge_attn"] = True
                self._tore_info["states"]["merge_mlp"] = True
            else:
                self._tore_info["states"]["merge_attn"] = False
                self._tore_info["states"]["merge_mlp"] = True
            output = super().forward(*args, **kwargs)
            return output

    return SD3Transformer2DModel_SDTM
'''
每层真正执行 merge/unmerge 或跳过
'''
def make_SDTM_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class JointTransformerBlock_SDTM(block_class):
        
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        ):

            self._tore_info["states"]["layer_current"] = self._tore_info["states"]["layer_iter"].pop(0)

            joint_attention_kwargs = joint_attention_kwargs or {}
            if self.use_dual_attention:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                    hidden_states, emb=temb
                )
            else:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

            if self.context_pre_only:
                norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
            else:
                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                    encoder_hidden_states, emb=temb
                )

            # Determine if current step is protected (skip merge/unmerge entirely)
            args = self._tore_info.get("args", {})
            states = self._tore_info.get("states", {})
            protect_freq = args.get("protect_steps_frequency", None)
            step_current = states.get("step_current", 0)

            def _is_protected_step(idx, freq):
                if freq is None or freq < 0 or freq == 0:
                    return False
                # protect steps that are multiples of freq and the final step handled at pipeline level
                return idx % freq == 0

            protected_step = _is_protected_step(step_current, protect_freq)

            # helper: store intermediate features by (step, layer), device configurable
            def _store_feature(name: str, tensor: torch.Tensor):
                try:
                    feat = self._tore_info.setdefault("features", {})
                    if not isinstance(feat.get(name), dict):
                        feat[name] = {}
                    l = self._tore_info.get("states", {}).get("layer_current", -1)
                    key = f"l{l}"
                    feat[name][key] = tensor.detach()
                except Exception:
                    # best-effort; do not break the forward pass if logging fails
                    pass

            if protected_step:
                attn_output, context_attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_encoder_hidden_states,
                    **joint_attention_kwargs,
                )
                attn_output = gate_msa.unsqueeze(1) * attn_output
                _store_feature("attn_output", attn_output)
                
                hidden_states = hidden_states + attn_output

                if self.use_dual_attention:
                    attn_output2 = self.attn2(hidden_states=norm_hidden_states2, **joint_attention_kwargs)
                    attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
                    _store_feature("attn_output2", attn_output2)
                    hidden_states = hidden_states + attn_output2

                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                if self._chunk_size is not None:
                    ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
                else:
                    ff_output = self.ff(norm_hidden_states)
                ff_output = gate_mlp.unsqueeze(1) * ff_output
                _store_feature("mlp_output", ff_output)
                hidden_states = hidden_states + ff_output
            else:
                #! Step 1_1: Compute_Merge
                m_a, m_m, u_a, u_m = compute_merge(norm_hidden_states, self._tore_info)
                if self.use_dual_attention:
                    m_a2, _, u_a2, _ = compute_merge(norm_hidden_states2, self._tore_info)

                #! Step 1_2_1: Merge_Attn
                norm_hidden_states = m_a(norm_hidden_states)
                attn_output, context_attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_encoder_hidden_states,
                    **joint_attention_kwargs,
                )
                attn_output = gate_msa.unsqueeze(1) * attn_output
                #! Step 1_2_2: UnMerge_Attn
                self._tore_info["states"]["unmerge_phase"] = "attn_output"
                attn_output = u_a(attn_output)
                if self._tore_info["args"]["cache_each_step"]==True: _store_feature("attn_output", attn_output)
                hidden_states = hidden_states + attn_output

                if self.use_dual_attention:
                    #! Step 1_2_3: Merge_DualAttn
                    norm_hidden_states2 = m_a2(norm_hidden_states2)
                    attn_output2 = self.attn2(hidden_states=norm_hidden_states2, **joint_attention_kwargs)
                    attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
                    #! Step 1_2_4: UnMerge_DualAttn
                    self._tore_info["states"]["unmerge_phase"] = "attn_output2"
                    attn_output2 = u_a2(attn_output2)
                    if self._tore_info["args"]["cache_each_step"]==True: _store_feature("attn_output2", attn_output2)
                    hidden_states = hidden_states + attn_output2

                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                #! Step 1_3_1: Merge_MLP
                norm_hidden_states = m_m(norm_hidden_states)
                if self._chunk_size is not None:
                    ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
                else:
                    ff_output = self.ff(norm_hidden_states)
                ff_output = gate_mlp.unsqueeze(1) * ff_output
                #! Step 1_3_2: UnMerge_MLP
                self._tore_info["states"]["unmerge_phase"] = "mlp_output"
                ff_output = u_m(ff_output)
                if self._tore_info["args"]["cache_each_step"]==True:  _store_feature("mlp_output", ff_output)
                hidden_states = hidden_states + ff_output

                self._tore_info["states"]["unmerge_phase"] = None
            # Process attention outputs for the `encoder_hidden_states`.
            if self.context_pre_only:
                encoder_hidden_states = None
            else:
                context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
                encoder_hidden_states = encoder_hidden_states + context_attn_output

                norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
                norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                if self._chunk_size is not None:
                    # "feed_forward_chunk_size" can be used to save memory
                    context_ff_output = _chunked_feed_forward(
                        self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                    )
                else:
                    context_ff_output = self.ff_context(norm_encoder_hidden_states)
                encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
                
            return encoder_hidden_states, hidden_states
    
    return JointTransformerBlock_SDTM

def apply_SDTM(
    pipe: torch.nn.Module,
    ratio: float = 0.5,
    deviation: float = 0.2,
    switch_step: int = 20,
    use_rand: bool = True,
    sx: int = 4,
    sy: int = 4,
    a_s: float = 0.05,
    a_d: float = 0.05,
    a_p: float = 2,
    adaptive_ssm: bool = False,
    ssm_threshold: float = 0.0,
    low_complexity_threshold: float = 0.66,
    high_complexity_threshold: float = 0.33,
    pseudo_merge: bool = False,
    mcw: float = 0.2,
    protect_steps_frequency: int = None,
    protect_layers_frequency: int = None,
    merge_attn: bool = False,
    merge_mlp: bool = False,
):

    # Make sure the module is not currently patched
    remove_patch(pipe)
    make_pipe_fn = make_SDTM_pipe
    pipe.__class__ = make_pipe_fn(pipe.__class__)
    pipe._tore_info = {
        "type": "SDTM",
        "args": {
            "ratio": ratio,
            "deviation": deviation,
            "switch_step": switch_step,
            "use_rand": use_rand,
            "sx": sx,
            "sy": sy,
            "a_s": a_s,
            "a_d": a_d,
            "a_p": a_p,
            "adaptive_ssm": adaptive_ssm,
            "ssm_threshold": ssm_threshold,
            "low_complexity_threshold": low_complexity_threshold,
            "high_complexity_threshold": high_complexity_threshold,
            "pseudo_merge": pseudo_merge,
            "mcw": mcw,
            "protect_steps_frequency": protect_steps_frequency,
            "protect_layers_frequency": protect_layers_frequency,
            "generator": None,
            "cache_each_step": False,
        },
        "features": {
            "attn_output": None,
            "attn_output2": None,
            "mlp_output": None,
        },
        "states": {
            "last_independent": None,
            "ratio_current": ratio,
            "step_count": None,
            "step_iter": None,
            "step_current": None,
            "layer_count": None,
            "layer_iter": None,
            "layer_current": None,
            "merge_attn": merge_attn,
            "merge_mlp": merge_mlp,
            "unmerge_phase": None,
        }
    }

    model = pipe.transformer
    make_model_fn = make_SDTM_model
    model.__class__ = make_model_fn(model.__class__)
    model._tore_info = pipe._tore_info
    for _, module in model.named_modules():
        if isinstance_str(module, "JointTransformerBlock"):
            make_block_fn = make_SDTM_block
            module.__class__ = make_block_fn(module.__class__)
            module._tore_info = pipe._tore_info
            # Disable dual attention on patched blocks to simplify behavior
            try:
                module.use_dual_attention = False
            except Exception:
                pass
    # Note: attention map collection is handled by _call_attn_with_get_scores so no global monkeypatch is needed.
    return pipe
