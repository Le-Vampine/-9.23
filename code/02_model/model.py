# -*- coding: utf-8 -*-
"""
MRF-Net: Missing-Robust Fusion Network
  掩码感知编码 -> 跨模态重建 -> 代理令牌门控 -> 共享/特有分解 -> 动态专家融合 -> 不确定性双头
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ("text", "audio", "vision")


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        n_even = pe[:, 0::2].shape[1]
        div = torch.exp(torch.arange(0, n_even, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ModalityEncoder(nn.Module):
    """掩码感知编码器：显式把"这一段是瞎的"喂给网络。"""

    def __init__(self, d_in, hidden, nhead, layers, dropout, down=1):
        super().__init__()
        # 逐时间步线性投影（比大核卷积省 5 倍算力）+ 轻量局部混合
        self.inp = nn.Sequential(nn.Linear(d_in + 2, hidden), nn.GELU())
        self.local = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.pos = PositionalEncoding(hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, nhead, dim_feedforward=hidden * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.down = None
        if down > 1:
            self.down = nn.Conv1d(hidden, hidden, kernel_size=down, stride=down)

    def forward(self, x, miss, pad):
        m = miss.float().unsqueeze(-1)               # 1=缺失（显式告知网络该段不可用）
        v = 1.0 - m
        h = self.inp(torch.cat([x * v, m, v], dim=-1))
        h = h + F.gelu(self.local(h.transpose(1, 2))).transpose(1, 2)
        h = self.enc(self.pos(h), src_key_padding_mask=pad)
        return h                      # 原生序列长度

    def downsample(self, h, miss, pad):
        if self.down is None:
            return h, miss, pad
        h2 = self.down(h.transpose(1, 2)).transpose(1, 2)
        k, s = self.down.kernel_size[0], self.down.stride[0]
        md = F.max_pool1d(miss.unsqueeze(1).float(), k, s).squeeze(1).bool()
        pd = F.max_pool1d(pad.unsqueeze(1).float(), k, s).squeeze(1).bool()
        return h2, md, pd


class Expert(nn.Module):
    """跨模态 + 跨时间自注意力专家。"""

    def __init__(self, hidden, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden))

    def forward(self, tok, pad):
        a, _ = self.attn(tok, tok, tok, key_padding_mask=pad)
        x = self.norm1(tok + a)
        x = self.norm2(x + self.ffn(x))
        return x


class MRFNet(nn.Module):
    def __init__(self, dims, hidden=128, nhead=4, enc_layers=2, n_experts=4,
                 dropout=0.1, downs=None, target_len=50, mag_levels=0,
                 ms_pool=False, ms_windows=(5, 25, 50), use_pol=False):
        super().__init__()
        downs = downs or {"text": 1, "audio": 1, "vision": 1}
        self.dims = dims
        self.hidden = hidden
        self.target_len = target_len
        self.mag_levels = int(mag_levels)
        self.ms_pool = bool(ms_pool)
        self.ms_windows = tuple(int(w) for w in ms_windows)
        self.use_pol = bool(use_pol)
        self.mods = list(MODALITIES)

        self.enc = nn.ModuleDict({
            m: ModalityEncoder(dims[m], hidden, nhead, enc_layers, dropout, downs[m])
            for m in self.mods})

        # --- 跨模态重建（缺失位置靠"造"）---
        self.cross = nn.ModuleDict({m: nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True) for m in self.mods})
        self.dec = nn.ModuleDict({m: nn.Linear(hidden, dims[m]) for m in self.mods})

        # --- 代理令牌 ---
        self.proxy = nn.ParameterDict({m: nn.Parameter(torch.randn(hidden) * 0.02) for m in self.mods})
        self.beta = nn.ModuleDict({m: nn.Linear(hidden + 1, 1) for m in self.mods})

        # --- 共享 / 特有分解 ---
        self.share = nn.ModuleDict({m: nn.Linear(hidden, hidden) for m in self.mods})
        self.spec = nn.ModuleDict({m: nn.Linear(hidden, hidden) for m in self.mods})
        self.mix = nn.ModuleDict({m: nn.Linear(hidden * 2, hidden) for m in self.mods})

        # --- 动态专家融合 ---
        self.mod_emb = nn.Embedding(3, hidden)
        self.experts = nn.ModuleList([Expert(hidden, nhead, dropout) for _ in range(n_experts)])
        self.gate = nn.Sequential(nn.Linear(hidden + 7, hidden), nn.GELU(), nn.Linear(hidden, n_experts))
        self.pool_score = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))

        # --- 输出头 ---
        self.head_mu = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.head_lv = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.head_cls = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

        # --- 强度分级辅助头（CORAL 有序回归：K-1 个累计 logit）---
        # 动机（实测）：61.7% 的样本 |y|<=2/3（弱标注）而 ACC 仅 0.486；|y|=1/3 层
        # ACC 更低至 0.361 —— 模型分不清“完全中性”与“轻微极性”。该头把 |y| 离散为
        # K 个等级（步长 1/3）做有序回归，强制模型在零点附近建立分辨率。
        if self.mag_levels > 1:
            self.mag_w = nn.Linear(hidden, 1, bias=False)
            self.mag_b = nn.Parameter(torch.linspace(-1.5, 1.5, self.mag_levels - 1))

        # --- 多尺度时间池化（补足单层注意力池化“过于均匀”的问题）---
        # 实测：帧注意力熵 3.86 / 上限 3.91 ≈ 均匀，说明注意力没聚焦到关键片段。
        # 改为“多窗口掩码均值池化 + 线性投影”，并与注意力池化残差相加。
        if self.ms_pool:
            T = 3 * int(target_len)          # 三模态拼接后的 token 长度
            nseg = sum(max(1, T // w) for w in self.ms_windows)
            self._ms_nseg = nseg
            self._ms_T = T
            self.ms_proj = nn.Linear(nseg * hidden, hidden)

        # --- 二分类极性辅助头（对齐文献 Acc-2 的做法）---
        if self.use_pol:
            self.head_pol = nn.Linear(hidden, 1)

    # ------------------------------------------------------------------
    def _masked_win_pool(self, h, pad):
        """多窗口掩码均值池化：把时间轴切成若干段，段内对非填充位置求均值。

        h:(B,T,H)  pad:(B,T) bool(True=填充)  返回 (B, nseg*H)
        """
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        valid = (~pad).float().unsqueeze(-1)                 # (B,T,1)
        T = h.size(1)
        outs = []
        for w in self.ms_windows:
            nb = max(1, T // w)
            seg = torch.arange(T, device=h.device) * nb // T   # 把 T 均分到 nb 段
            for b in range(nb):
                m = (seg == b).float().unsqueeze(0).unsqueeze(-1)   # (1,T,1)
                s = (h * m * valid).sum(dim=1)
                n = (m * valid).sum(dim=1).clamp(min=1.0)
                outs.append(s / n)
        return torch.cat(outs, dim=-1)

    @staticmethod
    def _availability(rho):
        """rho:(B,3) 缺失率 -> (B,7) 可用性特征。"""
        n_vis = (rho < 0.999).float().sum(dim=1, keepdim=True)
        return torch.cat([rho, (rho < 0.999).float(), n_vis / 3.0], dim=1)

    def forward(self, x, miss, pad, return_aux=False):
        """x/miss/pad: dict 模态 -> (B,L,d)/(B,L)/(B,L)"""
        h, rec, share, spec = {}, {}, {}, {}
        for m in self.mods:
            h[m] = self.enc[m](x[m], miss[m], pad[m])

        # --- 1) 跨模态重建（原生长度上计算，损失只取缺失位置）---
        for m in self.mods:
            others = [h[o] for o in self.mods if o != m]
            kv = torch.cat(others, dim=1)
            kv_pad = torch.cat([pad[o] for o in self.mods if o != m], dim=1)
            a, _ = self.cross[m](h[m], kv, kv, key_padding_mask=kv_pad)
            h_hat = h[m] + a * miss[m].float().unsqueeze(-1)
            rec[m] = self.dec[m](h_hat)

        # --- 2) 降采样到 target_len 统一时间尺度 ---
        hd, md, pd_ = {}, {}, {}
        for m in self.mods:
            hd[m], md[m], pd_[m] = self.enc[m].downsample(h[m], miss[m], pad[m])

        # --- 3) 代理令牌门控 + 分解 ---
        rho = torch.stack([md[m].float().mean(dim=1) for m in self.mods], dim=1)   # (B,3)
        beta_val = {}
        for m in self.mods:
            hm = hd[m]
            inp = torch.cat([hm.mean(dim=1), rho[:, self.mods.index(m)].unsqueeze(-1)], dim=-1)
            b = torch.sigmoid(self.beta[m](inp)).unsqueeze(1)                      # (B,1,1)
            beta_val[m] = b.detach().view(-1)
            hm = (1 - b) * hm + b * self.proxy[m].view(1, 1, -1)
            share[m] = self.share[m](hm)
            spec[m] = self.spec[m](hm)
            hd[m] = self.mix[m](torch.cat([share[m], spec[m]], dim=-1))

        # --- 4) 动态专家融合 ---
        tokens = torch.cat([hd[m] for m in self.mods], dim=1)
        tok_pad = torch.cat([pd_[m] for m in self.mods], dim=1)
        mid = torch.arange(3, device=tokens.device).repeat_interleave(tokens.size(1) // 3)
        tokens = tokens + self.mod_emb(mid).unsqueeze(0)

        av = self._availability(rho)
        cls_pool = tokens.mean(dim=1)
        g = torch.softmax(self.gate(torch.cat([cls_pool, av], dim=-1)), dim=-1)     # (B,E)

        outs = torch.stack([e(tokens, tok_pad) for e in self.experts], dim=1)       # (B,E,T,H)
        fused = (g.unsqueeze(-1).unsqueeze(-1) * outs).sum(dim=1)                  # (B,T,H)

        score = self.pool_score(fused).masked_fill(tok_pad.unsqueeze(-1), -1e9)
        alpha = torch.softmax(score, dim=1)                                        # (B,T,1)
        z = (alpha * fused).sum(dim=1)
        if self.ms_pool:
            pooled = self._masked_win_pool(fused, tok_pad)
            z = z + self.ms_proj(pooled)

        mu = self.head_mu(z).squeeze(-1)
        logvar = self.head_lv(z).squeeze(-1).clamp(-6.0, 4.0)
        logits = self.head_cls(z)

        if not return_aux:
            return mu, logvar, logits
        aux = dict(rec=rec, share=share, spec=spec, gate=g, alpha=alpha,
                   rho=rho, hd=hd, pad_ds=pd_, miss_ds=md, z=z, mu=mu, logits=logits,
                   beta=beta_val)
        if self.mag_levels > 1:
            aux["logits_mag"] = self.mag_w(z) + self.mag_b.view(1, -1)
        if self.use_pol:
            aux["pol"] = self.head_pol(z).squeeze(-1)
        return mu, logvar, logits, aux


class PlainFusion(nn.Module):
    """A0/A1 基线：**朴素拼接 + MLP** 融合（无掩码感知编码/重建/代理令牌/共享-特有分解/动态专家）。

    用于论文消融中的“常规固定权重融合模型”对照：
      * ``use_indicator=False`` → **A0**（完全不做缺失处理的常规模型）
      * ``use_indicator=True``  → **A1**（仅额外告知“哪段缺失”）

    接口与 :class:`MRFNet` 完全一致（``forward`` 返回 ``mu/logvar/logits[+aux]``），
    因此可直接复用 train/runtime/infer 的全部代码。
    """

    def __init__(self, dims, hidden=128, nhead=4, enc_layers=2, n_experts=4,
                 dropout=0.1, downs=None, target_len=50, use_indicator=True):
        super().__init__()
        self.mods = list(MODALITIES)
        self.use_indicator = bool(use_indicator)
        d_in = sum(int(dims[m]) for m in self.mods)
        if self.use_indicator:
            d_in += 2 * len(self.mods)          # 每模态：缺失率 + 可见率
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU())
        self.head_mu = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.head_lv = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.head_cls = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, x, miss, pad, return_aux=False):
        feats = []
        for m in self.mods:
            v = (~pad[m]).float().unsqueeze(-1)                     # (B,L,1)
            pooled = (x[m] * v).sum(dim=1) / v.sum(dim=1).clamp(min=1.0)   # 有效位均值池化
            feats.append(pooled)
            if self.use_indicator:
                mval = (miss[m] & ~pad[m]).float().mean(dim=1, keepdim=True)
                feats += [mval, 1.0 - mval]
        h = self.net(torch.cat(feats, dim=-1))
        mu = self.head_mu(h).squeeze(-1)
        logvar = self.head_lv(h).squeeze(-1).clamp(-6.0, 4.0)
        logits = self.head_cls(h)
        if not return_aux:
            return mu, logvar, logits
        return mu, logvar, logits, dict(z=h)
