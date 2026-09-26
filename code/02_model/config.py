# -*- coding: utf-8 -*-
"""
问题2 全局配置。

所有脚本（train / eval / infer）统一从本文件取配置，保证"可复现"要求不打折。
"""
import argparse
import os
from dataclasses import dataclass, field
from typing import List

# 工作区根目录（本文件位于 <工作区>/code/02_model/），用于给出与机器无关的默认路径
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))


@dataclass
class Config:
    # ---------------- 数据 ----------------
    data_dir: str = os.path.join(_ROOT, "附件2")   # 存放 aligned_50.pkl / unaligned_50.pkl 的目录
    version: str = "aligned"             # aligned | unaligned  <-- 全程必须一致
    target_len: int = 50                 # 统一最大序列位置数 K（与附件2 一致）
    min_zero_len: int = 2                # 连续多少个全零步才判为"缺失"（题目缺失为连续区间）

    # ---------------- 模型 ----------------
    hidden: int = 128
    nhead: int = 4
    enc_layers: int = 2
    n_experts: int = 4
    dropout: float = 0.1

    # ---------------- 训练 ----------------
    epochs_teacher: int = 30
    epochs_student: int = 50
    batch_size: int = 32
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    patience: int = 12          # 早停耐心轮数（MAE 会震荡，过小会过早停止）
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])

    # ---------------- 缺失模拟（训练课程） ----------------
    # 依据附件3 实测分布标定（规范产物：missing_profile.py → paper_tables/missing_profile.md）：
    #   audio 17/30（56.7%）样本含缺失、vision 18/30（60.0%）；
    #   缺失率 mean 0.101/0.107、p50 0.097/0.118、p90 0.224、max 0.400；
    #   区间为短碎片（段长均值 2.5 步、max 4 步）；位置 **头 0.46 / 中 0.27 / 尾 0.27**；
    #   口径 D1：连续 ≥2 帧全零，分母 = [0, 末个非零帧]（首部零帧计入缺失）。
    #   → 第三阶段 rho 上限设为 0.25（覆盖实测 p90=0.224），并保留少量高压样本
    curric: List = field(default_factory=lambda: [
        (1, 8, 0.00, 0.10, "single"),     # (起始epoch, 结束epoch, rho_min, rho_max, 缺失模态数模式)
        (9, 20, 0.04, 0.20, "mixed"),
        (21, 10 ** 9, 0.04, 0.25, "mixed"),
    ])
    # ⚠️ 该先验为中部分主（中 4/9），与 D1 实测（头 0.46 / 中 0.27 / 尾 0.27）**不匹配**。
    #    因性能路线已终止（缺失分布对齐实验 R1 证明无收益、AV 局部缺失退化 ≤0.7%，
    #    见 runs/exp_r1 与 runs/exp_m1），修改会作废已训练模型且预期收益为零，
    #    故保留现状并在论文「局限」中如实披露。
    pos_modes: List[str] = field(default_factory=lambda: [
        "middle", "middle", "middle", "middle", "head", "head", "head", "tail", "random"])
    n_intervals: List[int] = field(default_factory=lambda: [1, 3])   # 每模态拆成 1~3 段短碎片
    crop_aug: bool = True        # 训练时随机时间裁剪增强（补足首尾缺失场景）
    crop_min_keep: float = 0.7   # 裁剪后至少保留的比例
    mod_dropout: float = 0.0     # 整模态缺失增强的概率（训练期，0=关闭；v3 建议 0.15）
    ema: bool = False            # 学生权重指数移动平均（评估/保存用 EMA 权重）
    ema_decay: float = 0.995     # EMA 衰减系数（训练轮数少时用 0.99~0.995）
    save_half: bool = True       # checkpoint 以 float16 保存（附件 50MB 预算）

    # ---------------- 消融开关（论文第 4 节用；均不改变默认行为） ----------------
    ablate_mask_indicator: bool = False   # A2：训练与推理都不把"哪段缺失"喂给模型
    ablate_rec: bool = False              # A3：去掉跳模态重建损失（lam_rec=lam_rec0=0）
    ablate_curric: bool = False           # A9：去掉课程学习（固定缺失率区间）
    ablate_inject: bool = False           # A0：训练期完全不注入缺失（干净数据训练）
    model_kind: str = "mrf"                # mrf=MRF-Net；plain=朴素拼接+MLP（A0/A1 基线）

    # ---------------- 损失权重 ----------------
    lam_cls: float = 1.0
    reg_beta: float = 0.5       # SmoothL1 的 beta（0.5 时对 MAE 友好）
    lam_var: float = 0.1        # NLL 不确定性辅助项权重（过大会拖累 MAE）
    noise_sigma: float = 0.0    # 训练时输入高斯噪声强度（正则化，防过拟合）
    label_smoothing: float = 0.0  # 分类标签平滑
    cls_balanced: bool = True   # 按训练集类频倒数加权 CE（评测用 Macro-F1，中性类易被忽略）
    lam_rec: float = 1.0        # 缺失位置重建（自监督，核心）
    lam_rec0: float = 0.5       # 可见位置重建（保持表示保真）
    lam_sim: float = 0.5        # 共享表示相似
    lam_diff: float = 0.5       # 共享/特有正交
    lam_bal: float = 0.01       # 专家负载均衡
    lam_kd0: float = 0.3        # 蒸馏初始权重（随 epoch 指数衰减）
    lam_cons: float = 0.0       # 极性–强度一致性正则权重（标签 sign(y)≡class，实测 100% 一致）
    y_weight_alpha: float = 0.0  # 回归损失按 |y| 加权的强度（0=不加权）
    y_weight_gamma: float = 0.5  # 加权指数：w = 1 + alpha·|y|^gamma
    # --- D 系列：面向弱标注样本（|y| 小 / 中性类）---
    soft_tau: float = 0.0        # 强度软目标温度 tau（>0 才用；越小越接近硬标签）
    soft_mix: float = 0.0        # 硬 CE 与软标签 KL 的混合比（0=纯硬 CE）
    lam_emd: float = 0.0         # 有序 EMD² 项权重（惩罚跨类错误 > 邻类错误）
    weak_alpha: float = 0.0      # 弱标注样本加权强度：w = 1 + α·(1-|y|/3)
    lam_neusup: float = 0.0      # D3'：中性抑制正则（代价敏感，专治极性样本被判成中性）
    # --- D4：强度分级辅助头（CORAL 有序回归）---
    mag_levels: int = 0          # |y| 离散档位数（10 = 步长 1/3 到 3.0；0/1 = 关闭）
    lam_mag: float = 0.0         # 强度分级辅助损失权重
    bf16: bool = False           # CPU bfloat16 autocast（AMD Zen4/5 支持 AVX512-BF16，可加速矩阵乘）
    # --- 缺失率分布对齐测试集（附件3 实测 mean 0.183 / p90 0.333 / max 0.478）---
    rho_mix: List = None         # [(lo,hi,w),...]；非 None 时覆盖课程的 [rho_min,rho_max]
    rho_mix_epoch: int = 9       # 从第几轮开始启用 rho_mix（9=课程进入 mixed 阶段时）
    # --- 多尺度时间池化 & 二分类极性头 ---
    ms_pool: bool = False        # 多窗口掩码均值池化（缓解帧注意力“均匀化”）
    ms_windows: List = field(default_factory=lambda: [5, 25, 50])
    lam_pol: float = 0.0         # 二分类极性辅助头权重（1[y<0] 的 BCE，对齐 Acc-2）
    kd_decay_epochs: float = 15.0
    kd_warmup: int = 3          # 前若干 epoch 只用任务损失，让学生先站稳
    kd_tau: float = 2.0

    # ---------------- 输出与设备 ----------------
    out_dir: str = os.path.join(_ROOT, "runs", "q2")
    device: str = "auto"        # auto | cpu | cuda
    num_workers: int = 0
    limit: int = 0              # >0 时只用前 N 条样本（冒烟测试用）
    sweep: int = 0              # 1 时额外跑缺失因素扫描实验

    def __post_init__(self):
        self.pkl_path = os.path.join(
            self.data_dir, "%s_50.pkl" % self.version)
        self.auto_found = getattr(self, "auto_found", False)
        if not os.path.isfile(self.pkl_path):
            self.autodetect()
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        os.makedirs(self.out_dir, exist_ok=True)

    def autodetect(self):
        """在工程目录（及其上一级）中搜索 <version>_50.pkl，跳过 data_dummy。"""
        import glob
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        name = "%s_50.pkl" % self.version
        for root in (here, os.path.dirname(here)):
            hits = [h for h in glob.glob(os.path.join(root, "**", name), recursive=True)
                    if "data_dummy" not in h.replace("\\", "/")]
            if hits:
                hits.sort(key=len)
                self.pkl_path = hits[0]
                self.data_dir = os.path.dirname(hits[0])
                self.auto_found = True
                return self.pkl_path
        return None

    @property
    def down_factor_map(self):
        """各模态从原生长度降采样到 target_len 的步幅。"""
        if self.version == "unaligned":
            return {"text": 1, "audio": 10, "vision": 10}
        return {"text": 1, "audio": 1, "vision": 1}


def parse_args(argv=None):
    p = argparse.ArgumentParser("MRF-Net for Problem 2")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--version", type=str, default=None, choices=["aligned", "unaligned"])
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--lr_backbone", type=float, default=None)
    p.add_argument("--lr_head", type=float, default=None)
    p.add_argument("--noise_sigma", type=float, default=None)
    p.add_argument("--reg_beta", type=float, default=None,
                   help="SmoothL1 的 beta；越小越接近 L1（MAE 的最优损失）")
    p.add_argument("--ema", action="store_true", help="启用学生权重 EMA")
    p.add_argument("--ema_decay", type=float, default=None)
    p.add_argument("--lam_cons", type=float, default=None,
                   help="极性–强度一致性正则权重（建议 0.3~1.0）")
    p.add_argument("--y_weight", type=float, default=None,
                   help="回归损失按 |y| 加权的 alpha（建议 0.3~1.0）")
    p.add_argument("--soft_tau", type=float, default=None,
                   help="D2：强度软目标温度 tau（建议 0.2~0.5）")
    p.add_argument("--soft_mix", type=float, default=None,
                   help="D2：软标签与硬 CE 的混合比（建议 0.3~0.6）")
    p.add_argument("--lam_emd", type=float, default=None,
                   help="D2：有序 EMD² 项权重（建议 0.2~0.5）")
    p.add_argument("--weak_weight", type=float, default=None,
                   help="D3：弱标注样本加权 alpha，w=1+α(1-|y|/3)（建议 0.3~1.0）")
    p.add_argument("--lam_neusup", type=float, default=None,
                   help="D3'：中性抑制正则权重（建议 0.3~1.0）")
    p.add_argument("--mag_levels", type=int, default=None,
                   help="D4：强度分级辅助头的档位数（10=步长1/3；0=关闭）")
    p.add_argument("--lam_mag", type=float, default=None,
                   help="D4：强度分级 CORAL 损失权重（建议 0.3~1.0）")
    p.add_argument("--bf16", action="store_true",
                   help="CPU bfloat16 autocast（需硬件支持 AVX512-BF16）")
    p.add_argument("--rho_mix", type=str, default=None,
                   help="按测试集实测分布采样缺失率，格式 'w:lo:hi,...'，"
                        "例 '0.75:0.02:0.22,0.25:0.22:0.50'（附件3 适配）")
    p.add_argument("--ms_pool", action="store_true",
                   help="启用多尺度时间池化（多窗口掩码均值）")
    p.add_argument("--ms_windows", type=str, default=None,
                   help="多尺度窗口大小，例 '5,25,50'")
    p.add_argument("--lam_pol", type=float, default=None,
                   help="二分类极性辅助头权重（建议 0.3~1.0）")
    p.add_argument("--mod_dropout", type=float, default=None)
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--n_experts", type=int, default=None)
    p.add_argument("--enc_layers", type=int, default=None,
                   help="Transformer 编码层数（超参搜索维度）")
    p.add_argument("--lam_kd0", type=float, default=None,
                   help="蒸馏初始权重（随 epoch 指数衰减）；0 = 关闭蒸馏")
    p.add_argument("--nhead", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs_teacher", type=int, default=None)
    p.add_argument("--epochs_student", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sweep", type=int, default=None)
    p.add_argument("--no_distill", action="store_true")
    p.add_argument("--ablate_mask", action="store_true", help="A2：不给模型缺失指示")
    p.add_argument("--ablate_rec", action="store_true", help="A3：去掉跨模态重建损失")
    p.add_argument("--ablate_curric", action="store_true", help="A9：去掉缺失课程学习")
    p.add_argument("--ablate_inject", action="store_true", help="A0：训练期不注入缺失")
    p.add_argument("--no_crop", action="store_true", help="A0：关闭时间裁剪增强（干净训练）")
    p.add_argument("--model", default=None, choices=["mrf", "plain"],
                   help="mrf=MRF-Net（默认）；plain=朴素拼接+MLP（A0/A1 基线）")
    return p.parse_args(argv)


def get_config(argv=None):
    a = parse_args(argv)
    cfg = Config()
    for k in ["data_dir", "version", "out_dir", "device", "hidden",
              "batch_size", "epochs_teacher", "epochs_student", "limit", "sweep",
              "dropout", "weight_decay", "lr_backbone", "lr_head",
              "noise_sigma", "mod_dropout", "label_smoothing", "patience", "n_experts",
              "enc_layers", "lam_kd0", "nhead",
              "reg_beta", "ema_decay", "lam_cons", "y_weight",
              "soft_tau", "soft_mix", "lam_emd", "lam_neusup",
              "mag_levels", "lam_mag", "lam_pol"]:
        v = getattr(a, k)
        if v is not None:
            setattr(cfg, k, v)
    if a.seed is not None:
        cfg.seeds = [a.seed]
    cfg.use_distill = not a.no_distill
    cfg.ablate_mask_indicator = bool(a.ablate_mask)
    cfg.ablate_rec = bool(a.ablate_rec)
    cfg.ablate_curric = bool(a.ablate_curric)
    cfg.ablate_inject = bool(a.ablate_inject)
    cfg.ema = bool(a.ema)
    if a.y_weight is not None:
        cfg.y_weight_alpha = float(a.y_weight)
    if a.weak_weight is not None:
        cfg.weak_alpha = float(a.weak_weight)
    cfg.bf16 = bool(a.bf16)
    cfg.ms_pool = bool(a.ms_pool)
    if a.ms_windows:
        cfg.ms_windows = [int(x) for x in str(a.ms_windows).split(",")]
    if a.rho_mix:
        segs = []
        for part in str(a.rho_mix).split(","):
            w, lo, hi = part.split(":")
            segs.append((float(lo), float(hi), float(w)))
        cfg.rho_mix = segs
    if a.model:
        cfg.model_kind = a.model
    if a.no_crop:
        cfg.crop_aug = False
    if cfg.ablate_rec:                 # A3：关掉重建（缺失位与可见位一起关）
        cfg.lam_rec = 0.0
        cfg.lam_rec0 = 0.0
    if cfg.ablate_curric:              # A9：固定缺失率区间，取消三阶段课程
        cfg.curric = [(1, 10 ** 9, 0.04, 0.25, "mixed")]
    cfg.__post_init__()
    return cfg
