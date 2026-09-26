# -*- coding: utf-8 -*-
"""只读：从 checkpoint 里反查主模型与各候选的真实超参（论文超参表的数据来源）。

用途
----
  checkpoint 由 train.py 以 dict 形式保存，其中含配置字段。
  本脚本把它们打印出来，作为论文"关键参数表"的可溯源依据，
  同时核验"论文描述的方法"与"实际部署模型"是否一致。

严格只读：不训练、不改文件、不写盘（除非 --csv）。
"""
import argparse
import io
import json
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 论文关键参数表的候选字段（按含义分组，便于直接抄进表格）
GROUPS = [
    ("数据与序列", ["version", "target_len", "min_zero_len"]),
    ("网络结构", ["hidden", "nhead", "enc_layers", "n_experts", "dropout",
                  "ms_pool", "ms_windows", "mag_levels", "use_pol"]),
    ("训练方案", ["epochs_teacher", "epochs_student", "batch_size",
                  "lr_backbone", "lr_head", "weight_decay", "warmup_epochs",
                  "grad_clip", "patience", "ema", "ema_decay", "bf16"]),
    ("缺失注入课程", ["curric", "pos_modes", "n_intervals", "crop_aug",
                      "crop_min_keep", "mod_dropout", "rho_mix", "rho_mix_epoch"]),
    ("损失权重", ["lam_cls", "lam_var", "reg_beta", "lam_rec", "lam_rec0",
                  "lam_sim", "lam_diff", "lam_bal", "lam_kd0", "kd_tau",
                  "kd_warmup", "kd_decay_epochs", "lam_emd", "lam_mag",
                  "lam_neusup", "lam_cons", "lam_pol", "weak_alpha",
                  "noise_sigma", "label_smoothing", "cls_balanced"]),
    ("消融开关", ["ablate_mask_indicator", "ablate_rec", "ablate_curric",
                  "ablate_inject", "model_kind"]),
]


def load_cfg(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck if isinstance(ck, dict) else {}


def flat(ck):
    """把 config 字段拉平到一层。"""
    cfg = dict(ck.get("config", {}) or {})
    for k, v in ck.items():
        if k != "config":
            cfg.setdefault(k, v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=[
        "runs/ens_top2/student_s42.pt",
        "runs/ens_top2/student_s43.pt",
        "runs/tune/c00_baseline/student_s42.pt",
        "runs/tune/c06_kd05/student_s42.pt",
        "runs/q2v3/student_s42.pt",
    ])
    ap.add_argument("--csv", default=None, help="可选：把参数表写到此 CSV")
    a = ap.parse_args()

    cfgs = {}
    for rel in a.ckpts:
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.isfile(p):
            print("[MISS] %s" % rel)
            continue
        try:
            cfg = flat(load_cfg(p))
        except Exception as e:  # noqa: BLE001
            print("[ERR ] %s -> %s" % (rel, e))
            continue
        cfgs[rel] = cfg
        print("[OK  ] %s   (%d 个字段)" % (rel, len(cfg)))

    for rel, cfg in cfgs.items():
        print("\n" + "=" * 74)
        print(rel)
        print("=" * 74)
        for gname, keys in GROUPS:
            print("-- %s" % gname)
            for k in keys:
                if k in cfg:
                    v = cfg[k]
                    if isinstance(v, float):
                        v = ("%.6g" % v)
                    print("   %-22s = %s" % (k, v))

    if a.csv:
        keys = [k for _, ks in GROUPS for k in ks]
        rows = ["ckpt," + ",".join(keys)]
        for rel, cfg in cfgs.items():
            vals = []
            for k in keys:
                v = cfg.get(k, "")
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, ensure_ascii=False).replace(",", ";")
                elif isinstance(v, str):
                    v = v.replace(",", ";")
                vals.append(str(v))
            rows.append(rel + "," + ",".join(vals))
        with open(os.path.join(ROOT, a.csv), "w", encoding="utf-8-sig", newline="") as f:
            f.write("\n".join(rows) + "\n")
        print("\n[OK] 已写出 %s" % a.csv)


if __name__ == "__main__":
    main()
