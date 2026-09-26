# 问题2 主模型（MRF-Net）最终配方 —— 论文「关键参数表」的溯源依据

> 本文件是论文表 `tab:q2hparam` 的**唯一数据来源**，所有取值可直接对照代码与训练命令核对。
> 生成方式：只读核查 checkpoint、训练脚本与配置默认值，**未做任何推测性填写**。

## 1. 主模型的确定（哪一份产物是"主模型"）

| 项 | 值 | 证据 |
|---|---|---|
| 主模型目录 | `runs/ens_top2` | `runs/ens_top2/report_p0.json` → `meta.ckpts = ["student_s42.pt","student_s43.pt"]` |
| 集成规模 | 2 个 checkpoint，logit 平均 | 同上，`meta.n_ckpt = 2` |
| 决策阈值 | θ = 0.325（仅用 valid 拟合） | `meta.theta_ckpt = 0.325`，与 `submission/pred_att3.csv` 的 `theta` 列一致 |
| 提交文件 | `submission/pred_att3.csv` = `pred_att3_ens_top2.csv` | `D:\_promote_sub.ps1`（把 `*_ens_top2.csv` 提升为正式名） |
| 成员来源 | `runs/ens_e1` 的 seed 42 / 43 | 三份 `student_s4*.pt` 的大小与时间戳逐一对应 |

`runs/ens_e1` 的三种子按验证集 Macro-F1 为 0.6203 / 0.6051 / 0.5787（见 `code/02_model/ens_combo.py` 文档串），
剔除最弱的 seed 44 后集成的验证集指标更优，故最终采用 top-2 集成。

## 2. 训练命令（权威来源：`D:\_p9_seeds.ps1` 的 `$base`）

```powershell
python -u 02_model\train.py --version aligned --hidden 96 --n_experts 2 `
  --dropout 0.35 --weight_decay 0.0005 --lr_backbone 0.00005 --lr_head 0.0005 `
  --noise_sigma 0.15 --mod_dropout 0.15 --label_smoothing 0.1 --ablate_rec `
  --reg_beta 0.15 --ema --ema_decay 0.99 --batch_size 64 `
  --epochs_teacher 12 --epochs_student 14 --patience 8 `
  --mag_levels 10 --lam_mag 0.5 --lam_emd 0.3 `
  --weak_weight 0.5 --lam_neusup 0.25 --seed 42 `
  --out_dir <工作区>\runs\ens_e1
```

同一 `$base` 亦见 `code/02_model/tune_hyperparams.py` 的 `BASE` 常量（注释标明"与 `_p11_msp.ps1` 的 `$base` 完全一致"），
两处逐项比对一致，可作为交叉验证。

## 3. 未在命令行出现、取自 `code/02_model/config.py` 默认值的项

| 参数 | 取值 | 出处 |
|---|---|---|
| `target_len` | 50 | `config.py` 数据段 |
| `min_zero_len` | 2 | 同上 |
| `nhead` | 4 | 模型段默认 |
| `enc_layers` | 2 | 模型段默认 |
| `grad_clip` | 1.0 | 训练段默认；`train.py` 第 161 行调用 |
| 优化器 | AdamW + CosineAnnealingLR | `train.py` 第 292–296、321–325 行 |
| `lam_cls` | 1.0 | 损失权重段默认 |
| `lam_var` | 0.1 | 同上 |
| `lam_sim` / `lam_diff` | 0.5 / 0.5 | 同上 |
| `lam_bal` | 0.01 | 同上 |
| `lam_kd0` | 0.3 | 同上（初始值，随 epoch 指数衰减） |
| `kd_tau` / `kd_warmup` / `kd_decay_epochs` | 2.0 / 3 / 15 | 同上 |
| `crop_aug` / `crop_min_keep` | True / 0.7 | 缺失模拟段默认 |
| `curric` | (1–8, ρ∈[0,0.10], 单模态)、(9–20, ρ∈[0.04,0.20], 混合)、(21–, ρ∈[0.04,0.25], 混合) | 同上 |
| `pos_modes` | 中 4/9、头 3/9、尾 1/9、随机 1/9 | 同上 |
| `n_intervals` | 每模态 1~3 段 | 同上 |

## 4. ⚠️ 必须如实入文的两点

1. **跨模态重建在最终配方中被关闭**（`--ablate_rec`，即 `lam_rec = lam_rec0 = 0`）。
   论文式 `eq:rec` 与总损失 `eq:total` 中的 $\lambda_2\mathcal{L}_{\mathrm{rec}}$ 描述的是**搜索起点**的配置，
   不是最终部署的配置。消融表 A3（去掉重建）的误差低于其参考配置，与该设计决定一致。
2. **消融实验的参考配置（`runs/q2v3`）不是主模型**：q2v3 为 hidden 128 / 专家 4 / 教师 20 / 学生 30 轮 / 重建开启；
   主模型为 hidden 96 / 专家 2 / 教师 12 / 学生 14 轮 / 重建关闭。两者相差若干项同时变化的改动，
   故消融表须显式标注其参考配置，不得与主报结果混读。

## 5. 复现所需最小信息

- 环境：`D:\venv-mosei`（CPython 3.12，torch CPU 版）
- 数据：`附件2/aligned_50.pkl`（3395/728/727）
- 随机种子：42（教师与学生同种子）
- 命令：见第 2 节，`--out_dir` 指向任一空目录
- 训练耗时：CPU 约 17–20 分钟 / 种子
