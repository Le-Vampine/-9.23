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
                 dropout=0.1, downs=None, target_len=50):
        super().__init__()
        downs = downs or {"text": 1, "audio": 1, "vision": 1}
        self.dims = dims
        self.hidden = hidden
        self.target_len = target_len
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

    # ------------------------------------------------------------------
    @staticmethod
    def _availability(rho):
        """rho:(B,3) 缺失率 -> (B,6) 可用性特征。"""
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
        for m in self.mods:
            hm = hd[m]
            inp = torch.cat([hm.mean(dim=1), rho[:, self.mods.index(m)].unsqueeze(-1)], dim=-1)
            b = torch.sigmoid(self.beta[m](inp)).unsqueeze(1)                      # (B,1,1)
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

        mu = self.head_mu(z).squeeze(-1)
        logvar = self.head_lv(z).squeeze(-1).clamp(-6.0, 4.0)
        logits = self.head_cls(z)

        if not return_aux:
            return mu, logvar, logits
        aux = dict(rec=rec, share=share, spec=spec, gate=g, alpha=alpha,
                   rho=rho, hd=hd, pad_ds=pd_, miss_ds=md, z=z, mu=mu, logits=logits)
        return mu, logvar, logits, aux
