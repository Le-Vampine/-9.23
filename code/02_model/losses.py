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

    l_reg_main = reg_main(mu, batch["y_reg"], beta=getattr(cfg, "reg_beta", 0.5))
    l_reg_var = reg_nll(mu, logvar, batch["y_reg"])
    l_reg = l_reg_main + getattr(cfg, "lam_var", 0.1) * l_reg_var
    cw = getattr(cfg, "cls_weight", None)
    if cw is not None:
        cw = torch.tensor(cw, dtype=logits.dtype, device=logits.device)
    l_cls = cls_ce(logits, batch["y_cls"], cw, label_smoothing=getattr(cfg, "label_smoothing", 0.0))
    l_rec = recon_loss(out["rec"], x, miss, pad, use_missing=True)
    l_rec0 = recon_loss(out["rec"], x, miss, pad, use_missing=False)
    l_sim = sim_loss(out["share"], out["pad_ds"])       # 注意：share/spec 已降采样到统一长度
    l_diff = diff_loss(out["share"], out["spec"], out["pad_ds"])
    l_bal = balance_loss(out["gate"])

    loss = (l_reg + cfg.lam_cls * l_cls + cfg.lam_rec * l_rec + cfg.lam_rec0 * l_rec0
            + cfg.lam_sim * l_sim + cfg.lam_diff * l_diff + cfg.lam_bal * l_bal)

    detail = dict(reg=float(l_reg.detach()), reg_main=float(l_reg_main.detach()),
                  reg_var=float(l_reg_var.detach()), cls=float(l_cls.detach()),
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
