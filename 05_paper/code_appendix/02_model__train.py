# -*- coding: utf-8 -*-
"""
问题2 训练主程序：教师（全模态） -> 学生（课程学习 + 缺失注入 + 蒸馏） -> 评价 -> 缺失因素扫描

用法示例：
  python train.py --data_dir E:\\数学建模\\data --version aligned
  python train.py --version unaligned --epochs_teacher 30 --epochs_student 50 --sweep 1
  python train.py --limit 64 --epochs_teacher 2 --epochs_student 2 --seed 42   # 冒烟测试
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import get_config                                  # noqa: E402
from data_utils import load_all, MoseiDataset, collate, describe  # noqa: E402
from missing_sim import (inject_missing, batch_missing_ratios,   # noqa: E402
                         make_scenario_masks, crop_batch,
                         inject_modality_dropout)
from model import MRFNet, PlainFusion                             # noqa: E402
from losses import (total_loss, compute_metrics, tune_threshold,  # noqa: E402
                    format_metrics)

MODALITIES = ("text", "audio", "vision")


# --------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(split, cfg, shuffle, augment=False):
    ds = MoseiDataset(split, augment=augment)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, collate_fn=collate,
                      drop_last=False)


def prep(batch, device):
    x = {m: batch[m].to(device) for m in MODALITIES}
    miss = {m: batch["miss_" + m].to(device).bool() for m in MODALITIES}
    pad = {m: batch["pad_" + m].to(device) for m in MODALITIES}
    for m in MODALITIES:
        pad[m] = ensure_valid(pad[m])          # 防全遮 NaN
    return dict(x=x, miss=miss, pad=pad,
                y_reg=batch["y_reg"].to(device),
                y_cls=batch["y_cls"].to(device),
                idx=batch["idx"].to(device))


def ensure_valid(pad):
    """保证每条样本每个模态至少有一个非填充位（否则注意力 key 全被屏蔽 -> NaN）。"""
    allpad = pad.all(dim=1, keepdim=True)
    return torch.where(allpad, torch.zeros_like(pad), pad)


def mask_feed(b, cfg, unaware=False):
    """模型看到的"缺失指示"。

    unaware=True（推理期 A0 基线）或 cfg.ablate_mask_indicator=True（A2 消融，训练期也不给）
    时，把指示位置零：特征仍在缺失位被置零，但模型不知道"那段是缺失的"。
    """
    if unaware or getattr(cfg, "ablate_mask_indicator", False):
        return {m: torch.zeros_like(b["miss"][m]) for m in MODALITIES}
    return b["miss"]


def stack_masks(b):
    M = torch.stack([b["miss"][m] for m in MODALITIES], dim=-1)
    P = torch.stack([b["pad"][m] for m in MODALITIES], dim=-1)
    return M, P


def unstack_masks(b, M):
    """把缺失/填充掩码写回，并把不可用位置的特征置零（屏蔽进模型的无效值）。"""
    for i, m in enumerate(MODALITIES):
        b["miss"][m] = M[..., i]
        bad = (b["miss"][m] | b["pad"][m]).float().unsqueeze(-1)
        b["x"][m] = b["x"][m] * (1.0 - bad)


def curriculum(epoch, cfg):
    """课程学习的缺失注入参数。"""
    for e0, e1, r0, r1, mode in cfg.curric:
        if e0 <= epoch <= e1:
            return (r0, r1, mode)
    return (0.2, 0.5, "mixed")


# --------------------------------------------------------------------------
def run_epoch(model, loader, cfg, device, optimizer=None, teacher=None,
              lam_kd=0.0, inject_cfg=None, rng=None, ema=None, ema_decay=0.995,
              epoch=None):
    train = optimizer is not None
    model.train(train)
    if teacher is not None:
        teacher.eval()
    if rng is None:
        rng = np.random.default_rng()
    tot, cnt, details = 0.0, 0, {}
    for batch in loader:
        b = prep(batch, device)
        x_full = {m: b["x"][m].clone() for m in MODALITIES}
        if train and getattr(cfg, "crop_aug", False):
            b = crop_batch(b, rng, min_keep=cfg.crop_min_keep)
        if train and getattr(cfg, "noise_sigma", 0.0) > 0:
            sig = float(cfg.noise_sigma)
            for m in MODALITIES:
                b["x"][m] = b["x"][m] + torch.randn_like(b["x"][m]) * sig
        M, P = stack_masks(b)
        if inject_cfg is not None:
            mix = None
            if getattr(cfg, "rho_mix", None):
                ep = 10 ** 9 if epoch is None else int(epoch)
                if ep >= int(getattr(cfg, "rho_mix_epoch", 1) or 1):
                    mix = cfg.rho_mix
            M2, _ = inject_missing(M, P, inject_cfg[0], inject_cfg[1],
                                   inject_cfg[2], tuple(cfg.pos_modes), rng=rng,
                                   n_intervals=tuple(getattr(cfg, "n_intervals", (1, 3))),
                                   rho_mix=mix)
            M = M2
        if train and getattr(cfg, "mod_dropout", 0.0) > 0:
            M = inject_modality_dropout(M, P, cfg.mod_dropout, rng=rng, n_mods=1)
        unstack_masks(b, M)

        with torch.set_grad_enabled(train):
            ac = autocast_ctx(cfg)
            if ac is None:
                mu, logvar, logits, aux = model(b["x"], mask_feed(b, cfg), b["pad"], return_aux=True)
            else:
                with ac:
                    mu, logvar, logits, aux = model(b["x"], mask_feed(b, cfg), b["pad"], return_aux=True)
                mu, logvar, logits, aux = _to_fp32(mu), _to_fp32(logvar), _to_fp32(logits), _to_fp32(aux)
            out = dict(aux)
            out.update(mu=mu, logvar=logvar, logits=logits)

            t_out = None
            if teacher is not None and lam_kd > 0:
                with torch.no_grad():
                    zero = {m: torch.zeros_like(b["miss"][m]) for m in MODALITIES}
                    if ac is None:
                        t_mu, _, t_logits, t_aux = teacher(x_full, zero, b["pad"], return_aux=True)
                    else:
                        with ac:
                            t_mu, _, t_logits, t_aux = teacher(x_full, zero, b["pad"], return_aux=True)
                        t_mu, t_logits = t_mu.float(), t_logits.float()
                    t_out = dict(mu=t_mu, logits=t_logits, z=t_aux["z"].float())

            loss, det = total_loss(out, b, cfg, lam_kd=lam_kd, teacher_out=t_out)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if ema is not None:                     # 权重指数移动平均
                with torch.no_grad():
                    sd = model.state_dict()
                    for k in ema:
                        v = sd[k]
                        if v.dtype.is_floating_point:
                            ema[k].mul_(ema_decay).add_(v.detach().float(), alpha=1.0 - ema_decay)
                        else:
                            ema[k].copy_(v.detach())

        tot += float(loss.detach())
        cnt += 1
        for k, v in det.items():
            details[k] = details.get(k, 0.0) + v
    return tot / max(cnt, 1), {k: v / max(cnt, 1) for k, v in details.items()}


@torch.no_grad()
def predict(model, loader, cfg, device, split=None, scenario=None, unaware=False):
    model.eval()
    ac = autocast_ctx(cfg)
    S, CL, YR, YC, ID, RH = [], [], [], [], [], []
    for batch in loader:
        b = prep(batch, device)
        M, P = stack_masks(b)
        if scenario is not None:
            mtype, pos, rho = scenario
            M = make_scenario_masks(M, P, mtype, pos, rho, seed=1234)
        unstack_masks(b, M)
        feed = mask_feed(b, cfg, unaware=unaware)
        if ac is None:
            mu, _, logits = model(b["x"], feed, b["pad"])
        else:
            with ac:
                mu, _, logits = model(b["x"], feed, b["pad"])
            mu, logits = mu.float(), logits.float()
        S.append(mu.cpu().numpy())
        CL.append(logits.cpu().numpy())
        YR.append(b["y_reg"].cpu().numpy())
        YC.append(b["y_cls"].cpu().numpy())
        RH.append(batch_missing_ratios(M, P).cpu().numpy())
        if split is not None:
            idx = b["idx"].cpu().numpy()
            ID.extend([split["ids"][i] for i in idx])
    out = dict(score=np.concatenate(S), logits=np.concatenate(CL),
               y_reg=np.concatenate(YR), y_cls=np.concatenate(YC),
               rho=np.concatenate(RH))
    out["ids"] = np.array(ID) if ID else None
    return out


def report(pred, theta, tag=""):
    res = compute_metrics(pred["y_reg"], pred["score"], pred["y_cls"],
                          theta=theta, y_pred_cls=pred["logits"].argmax(-1))
    print(format_metrics(res, tag=tag))
    return res


# --------------------------------------------------------------------------
def build_model(cfg, dims):
    kind = getattr(cfg, "model_kind", "mrf")
    if kind == "plain":
        # A0/A1 基线：朴素拼接 + MLP；--ablate_mask 时连“缺失指示”这一输入通道也不给
        return PlainFusion(dims=dims, hidden=cfg.hidden, dropout=cfg.dropout,
                           target_len=cfg.target_len,
                           use_indicator=not getattr(cfg, "ablate_mask_indicator", False)).to(cfg.device)
    return MRFNet(dims=dims, hidden=cfg.hidden, nhead=cfg.nhead,
                  enc_layers=cfg.enc_layers, n_experts=cfg.n_experts,
                  dropout=cfg.dropout, downs=cfg.down_factor_map,
                  target_len=cfg.target_len,
                  mag_levels=int(getattr(cfg, "mag_levels", 0) or 0),
                  ms_pool=bool(getattr(cfg, "ms_pool", False)),
                  ms_windows=tuple(getattr(cfg, "ms_windows", (5, 25, 50))),
                  use_pol=float(getattr(cfg, "lam_pol", 0.0) or 0.0) > 0).to(cfg.device)


def autocast_ctx(cfg):
    """CPU bfloat16 autocast 上下文（本机 AMD Zen5 支持 AVX512-BF16）。

    注意：CPU 上 PyTorch 的“半精度加速”是 **bfloat16** 而不是 float16
    （AVX512-FP16 仅 Intel Sapphire Rapids 有；AMD 无 fp16 原生通路）。
    bf16 与 fp32 动态范围相同（8 位指数），无需 GradScaler。
    """
    if not getattr(cfg, "bf16", False):
        return None
    try:
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    except Exception as e:                                   # pragma: no cover
        print("[WARN] bf16 autocast 不可用：%s" % e)
        return None


def _to_fp32(obj):
    """把模型输出递归转回 float32，保证自定义损失在 fp32 下计算（可预测、无精度噪声）。"""
    if torch.is_tensor(obj):
        return obj.float() if obj.dtype.is_floating_point else obj
    if isinstance(obj, dict):
        return {k: _to_fp32(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_fp32(v) for v in obj)
    return obj


def run_experiment(cfg, seed, data):
    print("\n" + "=" * 78)
    print("[SEED %d] version=%s device=%s hidden=%d epochs(t/s)=%d/%d distill=%s"
          % (seed, cfg.version, cfg.device, cfg.hidden,
             cfg.epochs_teacher, cfg.epochs_student, getattr(cfg, "use_distill", True)))
    print("=" * 78)
    set_seed(seed)
    tr, va, te = data
    if cfg.cls_balanced:
        cnt = np.bincount(np.asarray(tr["labels_cls"]).astype(np.int64), minlength=3).astype(np.float64)
        w = cnt.sum() / (3.0 * np.maximum(cnt, 1.0))
        w = np.clip(w, 0.2, 10.0)                 # 防止某类缺失时权重爆炸
        cfg.cls_weight = [float(x) for x in w]
        print("[CLS] 类别计数=%s  权重=%s" % (cnt.astype(int).tolist(),
                                          [round(x, 3) for x in cfg.cls_weight]))
    dims = {m: tr["D"][m] for m in MODALITIES}
    print("dims = %s" % dims)
    train_loader = make_loader(tr, cfg, shuffle=True, augment=True)
    valid_loader = make_loader(va, cfg, shuffle=False)
    test_loader = make_loader(te, cfg, shuffle=False)

    tag = "s%d" % seed
    hist = []

    # ---------------- 1) 教师：全模态 ----------------
    teacher = build_model(cfg, dims)
    opt = torch.optim.AdamW([
        {"params": [p for n, p in teacher.named_parameters() if "head_" not in n], "lr": cfg.lr_backbone},
        {"params": [p for n, p in teacher.named_parameters() if "head_" in n], "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs_teacher))
    best_mae, best_state, bad = 1e9, None, 0
    for ep in range(1, cfg.epochs_teacher + 1):
        t0 = time.time()
        l, det = run_epoch(teacher, train_loader, cfg, cfg.device, optimizer=opt, inject_cfg=None)
        sched.step()
        pv = predict(teacher, valid_loader, cfg, cfg.device, split=va)
        mae = float(np.abs(pv["y_reg"] - pv["score"]).mean())
        print("[T ep%02d] loss=%.4f mae=%.4f rec=%.4f (%.1fs)"
              % (ep, l, mae, det.get("rec", 0.0), time.time() - t0))
        hist.append(dict(phase="teacher", epoch=ep, loss=l, valid_mae=mae, **det))
        if mae < best_mae - 1e-4:
            best_mae, bad = mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print("[T] early stop @ep%d" % ep)
                break
    if best_state is not None:
        teacher.load_state_dict(best_state)
    print("[T] best valid MAE = %.4f" % best_mae)

    # ---------------- 2) 学生：缺失鲁棒（课程学习 + 蒸馏） ----------------
    student = build_model(cfg, dims)
    opt = torch.optim.AdamW([
        {"params": [p for n, p in student.named_parameters() if "head_" not in n], "lr": cfg.lr_backbone},
        {"params": [p for n, p in student.named_parameters() if "head_" in n], "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs_student))
    use_kd = getattr(cfg, "use_distill", True)
    ema = None
    if getattr(cfg, "ema", False):
        ema = {k: v.detach().clone().float() for k, v in student.state_dict().items()}
        print("[EMA] 启用权重滑动平均，decay=%.4f" % getattr(cfg, "ema_decay", 0.995))
    best_mae, best_state, bad = 1e9, None, 0
    for ep in range(1, cfg.epochs_student + 1):
        t0 = time.time()
        r0, r1, mode = curriculum(ep, cfg)
        if getattr(cfg, "ablate_inject", False):        # A0：干净数据训练，不注入缺失
            inject_cfg = None
        else:
            inject_cfg = (r0, r1, mode)
        if use_kd and ep > cfg.kd_warmup:
            lam_kd = cfg.lam_kd0 * float(np.exp(-(ep - 1 - cfg.kd_warmup) / cfg.kd_decay_epochs))
        else:
            lam_kd = 0.0
        l, det = run_epoch(student, train_loader, cfg, cfg.device, optimizer=opt,
                           teacher=teacher, lam_kd=lam_kd, inject_cfg=inject_cfg,
                           rng=np.random.default_rng(seed * 1000 + ep),
                           ema=ema, ema_decay=float(getattr(cfg, "ema_decay", 0.995)),
                           epoch=ep)
        sched.step()
        if ema is not None:
            _bk = {k: v.detach().clone() for k, v in student.state_dict().items()}
            dt = next(student.parameters()).dtype
            student.load_state_dict({k: v.to(dt) for k, v in ema.items()})
            pv = predict(student, valid_loader, cfg, cfg.device, split=va)
            student.load_state_dict(_bk)
        else:
            pv = predict(student, valid_loader, cfg, cfg.device, split=va)
        mae = float(np.abs(pv["y_reg"] - pv["score"]).mean())
        print("[S ep%02d] loss=%.4f mae=%.4f rec=%.4f kd=%.4f | rho~[%.2f,%.2f] %s (%.1fs)"
              % (ep, l, mae, det.get("rec", 0.0), det.get("kd", 0.0), r0, r1, mode, time.time() - t0))
        hist.append(dict(phase="student", epoch=ep, loss=l, valid_mae=mae,
                         rho_min=r0, rho_max=r1, lam_kd=lam_kd, **det))
        if mae < best_mae - 1e-4:
            best_mae, bad = mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print("[S] early stop @ep%d" % ep)
                break
    if best_state is not None:
        student.load_state_dict(best_state)
    print("[S] best valid MAE = %.4f%s" % (best_mae, "（EMA 权重）" if ema is not None else ""))

    # ---------------- 3) 阈值选择 + 评价 ----------------
    pv = predict(student, valid_loader, cfg, cfg.device, split=va)
    theta, f1v = tune_threshold(pv["score"], pv["y_cls"])
    print("[TH] theta=%.2f (valid MacroF1=%.4f)" % (theta, f1v))
    res = {}
    res["valid_clean"] = report(pv, theta, tag="[VALID clean]")
    pt = predict(student, test_loader, cfg, cfg.device, split=te)
    res["test_clean"] = report(pt, theta, tag="[TEST  clean]")
    res["theta"] = theta

    # ---------------- 4) 缺失因素扫描（可选，用于第 4 节规律分析） ----------------
    if cfg.sweep:
        rows = []
        pv0 = pv
        for mtype in ["text", "audio", "vision", "text+audio", "text+vision", "audio+vision"]:
            for pos in ["head", "middle", "tail", "random"]:
                for rho in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    pa = predict(student, valid_loader, cfg, cfg.device, split=va,
                                 scenario=(mtype, pos, rho), unaware=False)
                    pu = predict(student, valid_loader, cfg, cfg.device, split=va,
                                 scenario=(mtype, pos, rho), unaware=True)
                    ra = report(pa, theta)
                    ru = report(pu, theta)
                    rows.append(dict(missing_type=mtype, position=pos, rho=rho,
                                     aware_mae=ra["mae"], aware_f1=ra["f1"], aware_acc=ra["acc"],
                                     aware_corr=ra["pearson"],
                                     unaware_mae=ru["mae"], unaware_f1=ru["f1"],
                                     unaware_acc=ru["acc"], unaware_corr=ru["pearson"]))
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(cfg.out_dir, "sweep_valid_%s.csv" % tag), index=False)
        print("[SWEEP] 已写入 %s" % os.path.join(cfg.out_dir, "sweep_valid_%s.csv" % tag))

        # 代表性场景在 test 上的"缺失泛化"验证（附件2 test 有标签，可量化）
        trows = []
        for mtype, pos, rho in [("text", "middle", 0.3), ("audio", "random", 0.3),
                                ("vision", "random", 0.3), ("text+audio", "random", 0.3),
                                ("audio+vision", "middle", 0.3), ("text+audio+vision", "random", 0.3)]:
            pa = predict(student, test_loader, cfg, cfg.device, split=te,
                         scenario=(mtype, pos, rho), unaware=False)
            pu = predict(student, test_loader, cfg, cfg.device, split=te,
                         scenario=(mtype, pos, rho), unaware=True)
            ra, ru = report(pa, theta), report(pu, theta)
            trows.append(dict(missing_type=mtype, position=pos, rho=rho,
                              aware_mae=ra["mae"], aware_f1=ra["f1"], aware_acc=ra["acc"],
                              unaware_mae=ru["mae"], unaware_f1=ru["f1"], unaware_acc=ru["acc"]))
        pd.DataFrame(trows).to_csv(os.path.join(cfg.out_dir, "sweep_test_%s.csv" % tag), index=False)
        print("[SWEEP] 已写入 %s" % os.path.join(cfg.out_dir, "sweep_test_%s.csv" % tag))
        res["sweep"] = rows

    # ---------------- 5) 保存 ----------------
    ckpt = os.path.join(cfg.out_dir, "student_%s.pt" % tag)
    state = student.state_dict()
    if getattr(cfg, "save_half", False):
        state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
    torch.save(dict(state=state, dims=dims, theta=theta,
                    stats=tr["stats"], cfg=cfg.__dict__, model_kind=getattr(cfg, "model_kind", "mrf"),
                    mag_levels=int(getattr(cfg, "mag_levels", 0) or 0),
                    ms_pool=bool(getattr(cfg, "ms_pool", False)),
                    ms_windows=list(getattr(cfg, "ms_windows", (5, 25, 50))),
                    use_pol=float(getattr(cfg, "lam_pol", 0.0) or 0.0) > 0,
                    version=cfg.version, seed=seed, half=bool(getattr(cfg, "save_half", False))), ckpt)
    print("[SAVE] %s (%.2f MB)" % (ckpt, os.path.getsize(ckpt) / 1e6))
    import pandas as pd
    pd.DataFrame(hist).to_csv(os.path.join(cfg.out_dir, "history_%s.csv" % tag), index=False)
    with open(os.path.join(cfg.out_dir, "metrics_%s.json" % tag), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in res.items() if k != "sweep"}, f,
                  ensure_ascii=False, indent=2)
    return res


def main():
    cfg = get_config()
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    print("[CFG] %s" % json.dumps({k: v for k, v in cfg.__dict__.items()
                                   if isinstance(v, (int, float, str, bool))},
                                  ensure_ascii=False, indent=2))
    print("[DATA] 读取 %s" % cfg.pkl_path)
    tr, va, te = load_all(cfg)
    for name, s in (("train", tr), ("valid", va), ("test", te)):
        print(describe(s, name))

    all_res = []
    for seed in cfg.seeds:
        all_res.append(run_experiment(cfg, seed, (tr, va, te)))

    if len(all_res) > 1:
        print("\n" + "=" * 78)
        print("[SUMMARY] %d seeds: mean +/- std" % len(all_res))
        keys = ["mae", "pearson", "ccc", "acc", "f1"]
        for split in ("valid_clean", "test_clean"):
            line = split + ": "
            for k in keys:
                v = np.array([r[split][k] for r in all_res])
                line += "%s=%.4f+/-%.4f  " % (k, v.mean(), v.std())
            print(line)
        with open(os.path.join(cfg.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump([{k: v for k, v in r.items() if k != "sweep"} for r in all_res],
                      f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
