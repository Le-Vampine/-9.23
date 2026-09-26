# -*- coding: utf-8 -*-
"""损失函数与评价指标。"""
import numpy as np
import torch
import torch.nn.functional as F

MODALITIES = ("text", "audio", "vision")


# --------------------------------------------------------------------------
# 损失
# --------------------------------------------------------------------------
def reg_nll(mu, logvar, y):
    """高斯负对数似然（仅作不确定性辅助项）。"""
    return 0.5 * (torch.exp(-logvar) * (y - mu) ** 2 + logvar).mean()


def reg_main(mu, y, beta=0.5):
    """主回归损失：SmoothL1，与官方评价指标 MAE 对齐。"""
    return F.smooth_l1_loss(mu, y, beta=beta)


def cls_ce(logits, y, weight=None, label_smoothing=0.0):
    return F.cross_entropy(logits, y, weight=weight, label_smoothing=label_smoothing)


# --------------------------------------------------------------------------
# D 系列：针对"弱标注样本"（|y| 小 / 中性类）的标签分布与有序损失
#   实测（附件2）：|y| 以 1/3 为步长、最大 3.0，且 类 = sign(均值强度)。
#   因此 |y| 本身就是"标注一致性"的直接度量：|y|=1/3 相当于标注者 2:1 分裂。
#   三类混淆中 84% 只是邻类错，说明需要**序信息**而不是更多容量。
# --------------------------------------------------------------------------
def intensity_soft_target(y_reg, tau=0.3):
    """用连续强度 |y| 构造三分类**软目标**（标签分布 / 置信度感知平滑）。

        q_极性 = tanh(|y|/tau),  q_中性 = 1 - q_极性,   y=0 时 q = 纯中性

    tau 越小越接近硬标签；tau→∞ 退化为"只区分中性/非中性"。
    """
    y = y_reg
    conf = torch.tanh(torch.abs(y) / max(float(tau), 1e-6)).clamp(0.0, 1.0)
    q = torch.zeros(y.shape[0], 3, dtype=y.dtype, device=y.device)
    pos = y > 0
    neg = y < 0
    q[:, 1] = 1.0 - conf
    q[pos, 2] = conf[pos]
    q[neg, 0] = conf[neg]
    q[~(pos | neg), 1] = 1.0
    return q / q.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def emd2_loss(logits, q):
    """EMD²（有序）惩罚：累计分布的平方差，使"跨类错误"代价 > "邻类错误"。"""
    p = torch.softmax(logits, dim=-1)
    return ((torch.cumsum(p, dim=-1) - torch.cumsum(q, dim=-1)) ** 2).sum(-1).mean()


def neu_suppress_loss(logits, y_cls, lam=0.0):
    """中性抑制正则（代价敏感学习）：惩罚"真值为极性、却把概率压在 neu 上"。

    实测误差结构：valid 上 299 个错误里 **167 个（56%）是把极性样本判成中性**
    （预测为 neu 的 269 个样本里只有 102 个正确，neu precision 仅 0.36）。
    该正则直接把 p_neu 从极性样本上挤走，把"中性↔极性"错误转化为"邻类极性"错误。
    """
    if not lam or lam <= 0:
        return None
    p = torch.softmax(logits, dim=-1)
    m = y_cls != 1
    if m.sum() < 1:
        return None
    return p[m, 1].mean()


# --------------------------------------------------------------------------
# D4：强度分级辅助头（CORAL 有序回归）
#   动机（实测）：61.7% 的样本 |y|<=2/3，其 ACC 仅 0.486；|y|=1/3 层更低到 0.361。
#   误差 84% 是邻类错 —— 模型分不清"完全中性"与"轻微极性"。把 |y| 按 1/3 步长
#   离散为 K 个等级做有序回归，等于直接监督“强度分辨率”，而不是让分类头自己猜。
# --------------------------------------------------------------------------
def mag_level(y_reg, n_levels=10):
    """|y| -> 等级 k ∈ [0, n_levels-1]（|y| 的离散档位，步长 1/3 → n_levels=10）。"""
    k = torch.round(torch.abs(y_reg) * 3.0).long()
    return k.clamp(0, int(n_levels) - 1)


def coral_loss(logits_mag, y_reg, n_levels=10):
    """CORAL：共享投影 + 有序偏置，对 K-1 个累计事件 P(k > j) 做二分类。

    logits_mag: (B, n_levels-1)
    """
    k = mag_level(y_reg, n_levels)
    levels = torch.arange(logits_mag.size(1), device=k.device).unsqueeze(0)
    target = (k.unsqueeze(1) > levels).float()
    return F.binary_cross_entropy_with_logits(logits_mag, target)


def mag_consistency(mu, logits_mag, n_levels=10):
    """辅助一致性：累计概率得到的期望等级应与 |mu| 对得上（弱监督，可选）。"""
    p = torch.sigmoid(logits_mag)                     # P(k > j)
    exp_k = p.sum(dim=1)                              # E[k] = Σ_j P(k>j)
    return F.smooth_l1_loss(exp_k / 3.0, torch.abs(mu).detach(), beta=0.2)


def cls_D_loss(logits, y_cls, y_reg, cls_weight=None, label_smoothing=0.0,
               soft_tau=0.0, soft_mix=0.0, lam_emd=0.0, weak_alpha=0.0,
               lam_neusup=0.0):
    """D 系列合成分类损失（针对弱标注样本 / 中性类）。

    D2  soft_mix>0 ：硬 CE 与强度软目标 KL 的凸组合
    D2  lam_emd>0  ：有序 EMD² 项（内建序结构；实测 84% 错误只是邻类错）
    D3  weak_alpha>0：按 w = 1 + α·(1-|y|/3) 给**弱标注样本**加权
    D3' lam_neusup>0：中性抑制正则（代价敏感，专治"极性样本被判成中性"）
    """
    hard = F.cross_entropy(logits, y_cls, weight=cls_weight,
                           reduction="none", label_smoothing=label_smoothing)
    q = None
    if soft_mix and soft_mix > 0:
        q = intensity_soft_target(y_reg, tau=soft_tau if soft_tau > 0 else 0.3)
        soft = -(q * F.log_softmax(logits, dim=-1)).sum(-1)
        per = (1.0 - float(soft_mix)) * hard + float(soft_mix) * soft
    else:
        per = hard
    if weak_alpha and weak_alpha > 0:
        w = 1.0 + float(weak_alpha) * (1.0 - (torch.abs(y_reg) / 3.0).clamp(0, 1))
        per = per * w
    loss = per.mean()
    if lam_emd and lam_emd > 0:
        if q is None:
            q = intensity_soft_target(y_reg, tau=soft_tau if soft_tau > 0 else 0.3)
        loss = loss + float(lam_emd) * emd2_loss(logits, q)
    if lam_neusup and lam_neusup > 0:
        ls = neu_suppress_loss(logits, y_cls, lam_neusup)
        if ls is not None:
            loss = loss + float(lam_neusup) * ls
    return loss



def consistency_loss(mu, logits):
    """极性–强度一致性。

    实验事实：附件2 的 sign(强度) 与三分类标签一致率为 **1.0000**（n=4850），
    即“极性 ≡ 强度符号”是硬约束。因此 tanh(μ) 与分类头的期望极性 E[cls]=P(pos)−P(neg)
    应当一致；该正则把回归头与分类头对齐。
    """
    p = torch.softmax(logits, dim=-1)
    e = p[:, 2] - p[:, 0]                    # ∈[-1,1]
    return F.mse_loss(torch.tanh(mu), e)


def weighted_smooth_l1(mu, y, beta=0.5, alpha=0.0, gamma=0.5):
    """按 |y| 加权的 SmoothL1：把训练重心移向误差主体（强情感样本）。

    alpha=0 时退化为普通 SmoothL1（与 MAE 目标对齐）。
    """
    loss = F.smooth_l1_loss(mu, y, beta=beta, reduction="none")
    if alpha and alpha > 0:
        w = 1.0 + float(alpha) * torch.abs(y).pow(float(gamma))
        loss = loss * w
    return loss.mean()


def masked_l1(pred, target, sel):
    """sel:(B,L) bool，只对选中位置求 L1。"""
    if sel.sum() < 1:
        return pred.new_zeros(())
    p = pred[sel]
    t = target[sel]
    return (p - t).abs().mean()


def recon_loss(rec, x, miss, pad, use_missing=True):
    """重建损失：use_missing=True 时只算缺失位置，否则只算可见位置。"""
    total, n = rec[list(rec.keys())[0]].new_zeros(()), 0
    for m in MODALITIES:
        sel = (miss[m] & ~pad[m]) if use_missing else (~miss[m] & ~pad[m])
        if sel.sum() < 1:
            continue
        total = total + masked_l1(rec[m], x[m], sel)
        n += 1
    return total / max(n, 1)


def masked_mean(h, valid):
    w = valid.float().unsqueeze(-1)
    return (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)


def sim_loss(share, pad):
    """共享表示相似性：把各模态"模态不变"部分拉到一起。"""
    pooled = {m: masked_mean(share[m], ~pad[m]) for m in MODALITIES}
    loss, n = pooled["text"].new_zeros(()), 0
    keys = list(pooled.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            loss = loss + F.mse_loss(pooled[keys[i]], pooled[keys[j]])
            n += 1
    return loss / max(n, 1)


def diff_loss(share, spec, pad):
    """共享/特有正交：用余弦平方，避免尺度影响。"""
    loss, n = None, 0
    for m in MODALITIES:
        a, b = share[m], spec[m]
        cos = F.cosine_similarity(a, b, dim=-1)
        w = (~pad[m]).float()
        v = ((cos ** 2) * w).sum() / w.sum().clamp(min=1.0)
        loss = v if loss is None else loss + v
        n += 1
    return loss / max(n, 1)


def balance_loss(gate):
    """专家负载均衡（熵形式），防止专家坍缩。"""
    gbar = gate.mean(dim=0)
    E = gate.size(1)
    return -(gbar * torch.log(gbar + 1e-8)).sum() * E


def total_loss(out, batch, cfg, lam_kd=0.0, teacher_out=None):
    """组装总损失；返回 (loss, 明细 dict)。"""
    mu, logvar, logits = out["mu"], out["logvar"], out["logits"]
    x, miss, pad = batch["x"], batch["miss"], batch["pad"]

    l_reg_main = weighted_smooth_l1(mu, batch["y_reg"], beta=getattr(cfg, "reg_beta", 0.5),
                                    alpha=getattr(cfg, "y_weight_alpha", 0.0),
                                    gamma=getattr(cfg, "y_weight_gamma", 0.5))
    l_reg_var = reg_nll(mu, logvar, batch["y_reg"])
    l_reg = l_reg_main + getattr(cfg, "lam_var", 0.1) * l_reg_var
    cw = getattr(cfg, "cls_weight", None)
    if cw is not None:
        cw = torch.tensor(cw, dtype=logits.dtype, device=logits.device)
    soft_mix = float(getattr(cfg, "soft_mix", 0.0) or 0.0)
    lam_emd = float(getattr(cfg, "lam_emd", 0.0) or 0.0)
    weak_alpha = float(getattr(cfg, "weak_alpha", 0.0) or 0.0)
    lam_neusup = float(getattr(cfg, "lam_neusup", 0.0) or 0.0)
    if soft_mix > 0 or lam_emd > 0 or weak_alpha > 0 or lam_neusup > 0:
        l_cls = cls_D_loss(logits, batch["y_cls"], batch["y_reg"], cw,
                           label_smoothing=getattr(cfg, "label_smoothing", 0.0),
                           soft_tau=float(getattr(cfg, "soft_tau", 0.0) or 0.0),
                           soft_mix=soft_mix, lam_emd=lam_emd,
                           weak_alpha=weak_alpha, lam_neusup=lam_neusup)
    else:
        l_cls = cls_ce(logits, batch["y_cls"], cw, label_smoothing=getattr(cfg, "label_smoothing", 0.0))
    zero = l_reg.new_zeros(())
    has = out.__contains__
    # D4：强度分级辅助头（CORAL 有序回归）
    lam_mag = float(getattr(cfg, "lam_mag", 0.0) or 0.0)
    l_mag = (coral_loss(out["logits_mag"], batch["y_reg"],
                        n_levels=int(getattr(cfg, "mag_levels", 10) or 10))
             if (lam_mag > 0 and has("logits_mag")) else zero)
    # 二分类极性辅助头：直接监督“是否负面”，与 Acc-2 的判定目标一致
    lam_pol = float(getattr(cfg, "lam_pol", 0.0) or 0.0)
    l_pol = zero
    if lam_pol > 0 and has("pol"):
        tneg = (batch["y_reg"] < 0).float()
        pw = ((1.0 - tneg).sum() / tneg.sum().clamp(min=1.0)).detach()
        l_pol = F.binary_cross_entropy_with_logits(out["pol"], tneg,
                                                   pos_weight=pw)
    # 模块可缺省：A0/A1 朴素基线没有重建/分解/专家，对应项自动跳过（置 0）
    l_rec = recon_loss(out["rec"], x, miss, pad, use_missing=True) if has("rec") else zero
    l_rec0 = recon_loss(out["rec"], x, miss, pad, use_missing=False) if has("rec") else zero
    l_sim = (sim_loss(out["share"], out["pad_ds"])
             if (has("share") and has("pad_ds")) else zero)
    l_diff = (diff_loss(out["share"], out["spec"], out["pad_ds"])
              if (has("share") and has("spec") and has("pad_ds")) else zero)
    l_bal = balance_loss(out["gate"]) if has("gate") else zero
    lam_cons = float(getattr(cfg, "lam_cons", 0.0))
    l_cons = consistency_loss(mu, logits) if lam_cons > 0 else zero

    loss = (l_reg + cfg.lam_cls * l_cls + cfg.lam_rec * l_rec + cfg.lam_rec0 * l_rec0
            + cfg.lam_sim * l_sim + cfg.lam_diff * l_diff + cfg.lam_bal * l_bal
            + lam_cons * l_cons + lam_mag * l_mag + lam_pol * l_pol)

    detail = dict(reg=float(l_reg.detach()), reg_main=float(l_reg_main.detach()),
                  reg_var=float(l_reg_var.detach()), cls=float(l_cls.detach()),
                  cons=float(l_cons.detach()), mag=float(l_mag.detach()),
                  pol=float(l_pol.detach()),
                  rec=float(l_rec.detach()),
                  rec0=float(l_rec0.detach()), sim=float(l_sim.detach()),
                  diff=float(l_diff.detach()), bal=float(l_bal.detach()))

    if teacher_out is not None and lam_kd > 0:
        tau = cfg.kd_tau
        l_kd_logit = F.kl_div(F.log_softmax(logits / tau, dim=-1),
                              F.softmax(teacher_out["logits"] / tau, dim=-1),
                              reduction="batchmean") * tau * tau
        l_kd_reg = F.mse_loss(mu, teacher_out["mu"])
        # 关系蒸馏：样本间相似度矩阵（CMAD 相关性感知的简化实现）
        st = F.normalize(out["z"], dim=-1)
        tt = F.normalize(teacher_out["z"], dim=-1)
        sim_s, sim_t = st @ st.t(), tt @ tt.t()
        l_kd_rel = F.mse_loss(sim_s, sim_t)
        l_kd = l_kd_logit + l_kd_reg + 0.1 * l_kd_rel     # 关系蒸馏降权，避免量级压过任务损失
        loss = loss + lam_kd * l_kd
        detail.update(kd=float(l_kd), kd_logit=float(l_kd_logit),
                      kd_reg=float(l_kd_reg), kd_rel=float(l_kd_rel))
    return loss, detail


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def _to_np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def polar_from_score(score, theta):
    """连续强度 -> 三分类（题目定义：<0 负, =0 中, >0 正；theta 容差在 valid 上标定）。"""
    out = np.ones_like(score, dtype=np.int64)
    out[score < -theta] = 0
    out[score > theta] = 2
    return out


def tune_threshold(score, label_cls, grid=None):
    """在验证集上按 Macro-F1 选择决策阈值。"""
    from sklearn.metrics import f1_score
    if grid is None:
        grid = np.arange(0.0, 2.01, 0.05)
    best, best_theta = -1.0, 0.0
    for th in grid:
        pred = polar_from_score(score, th)
        f1 = f1_score(label_cls, pred, average="macro", zero_division=0)
        if f1 > best:
            best, best_theta = f1, float(th)
    return best_theta, best


def compute_metrics(y_true_reg, y_pred_reg, y_true_cls, theta=0.0, y_pred_cls=None):
    from sklearn.metrics import accuracy_score, f1_score
    from scipy.stats import pearsonr
    yt = _to_np(y_true_reg).astype(np.float64)
    yp = _to_np(y_pred_reg).astype(np.float64)
    tc = _to_np(y_true_cls).astype(np.int64)

    mae = float(np.abs(yt - yp).mean())
    try:
        corr = float(pearsonr(yt, yp)[0])
    except Exception:
        corr = float("nan")
    # CCC
    mt, mp = yt.mean(), yp.mean()
    vt, vp = yt.var(), yp.var()
    cov = ((yt - mt) * (yp - mp)).mean()
    ccc = float(2 * cov / (vt + vp + (mt - mp) ** 2 + 1e-12))

    # 极性判别有两套口径：
    #   (A) θ 阈值口径：|ŷ| ≤ θ 判为 Neutral（θ 在 valid 上按 Macro-F1 选出）
    #       —— 与提交 CSV 的标签完全一致，故作为**主指标**
    #   (B) 分类头 argmax 口径：直接取分类头三类的最大值
    pc_th = polar_from_score(yp, theta).astype(np.int64)
    res = dict(mae=mae, pearson=corr, ccc=ccc,
               acc=float(accuracy_score(tc, pc_th)),
               f1=float(f1_score(tc, pc_th, average="macro", zero_division=0)),
               theta=float(theta))
    f1s = f1_score(tc, pc_th, average=None, zero_division=0, labels=[0, 1, 2])
    res["f1_neg"], res["f1_neu"], res["f1_pos"] = [float(x) for x in f1s]
    if y_pred_cls is not None:
        pc_h = _to_np(y_pred_cls).astype(np.int64)
        res["acc_head"] = float(accuracy_score(tc, pc_h))
        res["f1_head"] = float(f1_score(tc, pc_h, average="macro", zero_division=0))
    res["n"] = int(len(yt))
    return res


def format_metrics(res, tag=""):
    s = ("%s MAE=%.4f  Pearson=%.4f  CCC=%.4f  |  ACC(θ)=%.4f  MacroF1(θ)=%.4f"
         % (tag, res["mae"], res["pearson"], res["ccc"], res["acc"], res["f1"]))
    if "acc_head" in res:
        s += "  |  ACC(分类头)=%.4f  MacroF1(分类头)=%.4f" % (res["acc_head"], res["f1_head"])
    s += ("  |  F1 neg/neu/pos=%.3f/%.3f/%.3f  θ=%.2f  n=%d"
          % (res["f1_neg"], res["f1_neu"], res["f1_pos"], res["theta"], res["n"]))
    return s
