# -*- coding: utf-8 -*-
"""
问题2 全局配置。

所有脚本（train / eval / infer）统一从本文件取配置，保证"可复现"要求不打折。
"""
import argparse
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ---------------- 数据 ----------------
    data_dir: str = r"E:\数学建模\data"   # 存放 aligned_50.pkl / unaligned_50.pkl 的目录
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
    # 依据附件3 实测分布标定（calibrate_missing.py）：
    #   audio 56.7% 样本含缺失、vision 60%；缺失率 mean≈0.10、p90≈0.22、max=0.40；
    #   区间为短碎片（约 2 步）；位置 头 0.37 / 中 0.56 / 尾 0.07
    #   → 第三阶段 rho 上限设为 0.25（覆盖测试集 p95），并保留少量高压样本
    curric: List = field(default_factory=lambda: [
        (1, 8, 0.00, 0.10, "single"),     # (起始epoch, 结束epoch, rho_min, rho_max, 缺失模态数模式)
        (9, 20, 0.04, 0.20, "mixed"),
        (21, 10 ** 9, 0.04, 0.25, "mixed"),
    ])
    # 位置权重近似实测分布：中 4/9、头 3/9、尾 1/9、随机 1/9
    pos_modes: List[str] = field(default_factory=lambda: [
        "middle", "middle", "middle", "middle", "head", "head", "head", "tail", "random"])
    n_intervals: List[int] = field(default_factory=lambda: [1, 3])   # 每模态拆成 1~3 段短碎片
    crop_aug: bool = True        # 训练时随机时间裁剪增强（补足首尾缺失场景）
    crop_min_keep: float = 0.7   # 裁剪后至少保留的比例
    save_half: bool = True       # checkpoint 以 float16 保存（附件 50MB 预算）

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
    kd_decay_epochs: float = 15.0
    kd_warmup: int = 3          # 前若干 epoch 只用任务损失，让学生先站稳
    kd_tau: float = 2.0

    # ---------------- 输出与设备 ----------------
    out_dir: str = r"E:\数学建模\runs\q2"
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
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--n_experts", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs_teacher", type=int, default=None)
    p.add_argument("--epochs_student", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sweep", type=int, default=None)
    p.add_argument("--no_distill", action="store_true")
    return p.parse_args(argv)


def get_config(argv=None):
    a = parse_args(argv)
    cfg = Config()
    for k in ["data_dir", "version", "out_dir", "device", "hidden",
              "batch_size", "epochs_teacher", "epochs_student", "limit", "sweep",
              "dropout", "weight_decay", "lr_backbone", "lr_head",
              "noise_sigma", "label_smoothing", "patience", "n_experts"]:
        v = getattr(a, k)
        if v is not None:
            setattr(cfg, k, v)
    if a.seed is not None:
        cfg.seeds = [a.seed]
    cfg.use_distill = not a.no_distill
    cfg.__post_init__()
    return cfg
