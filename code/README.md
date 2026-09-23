# 问题2 代码使用说明（MRF-Net）

> 状态：**已在真实附件2（aligned_50.pkl，3395/728/727）上端到端跑通**。
> 教师 1 轮 98.6 s（valid MAE 0.735）；学生 1 轮 103 s（含蒸馏）；附件3 推理与 CSV 输出正常。

## 0. 实测数据事实（写进论文"数据说明"）

用 `python code\inspect_data.py --version aligned` 得到：

| 项 | 实测结果 |
|---|---|
| 文件结构 | 顶层 `train/valid/test`；字段 `raw_text, audio, vision, id, text_bert, classification_labels, regression_labels, text` |
| **缺失字段** | **没有 `audio_lengths` / `vision_lengths` / `annotations`**（与题目表2 描述不一致） |
| 形状 | 文本 (N,50,768) float32；语音 (N,50,74) float64；视觉 (N,50,35) float64；`text_bert` (N,3,50) |
| 规模 | train 3395 / valid 728 / test 727，合计 4850 |
| 填充率 | ~0.49–0.54（**一半位置是尾部填充**，有效长度均值约 24/50） |
| 有效区间内真缺失率 | 文本 0.0000 / 语音 0.0004 / 视觉 ~0.010（视觉约 1% 样本有零星全零段） |
| 极性标签 | 取值 {0,1,2}，分布 967/758/1670（中性约 22%） |

**两个直接结论**：① 有效长度只能由"尾部全零"推断（代码已实现，`infer_length_from_zeros`）；
② 附件2 几乎没有真缺失 → **问题2 的缺失必须训练时模拟注入**；且"填充"≠"缺失"，建模中严格区分。

## 1. 目录与职责

```
code/
├─ requirements.txt
├─ make_dummy_data.py            # 生成合成数据（仅用于代码自检，绝不用于论文结果）
├─ run_smoke_test.cmd            # 一键自检：合成数据 → 训练 → 推理
├─ att_paths.py                  # 附件3/4 路径自动发现（目录名含中文，避免命令行写中文）
├─ inspect_data.py               # 附件2 结构核查（字段/形状/填充率/真缺失率）
├─ inspect_label.py              # label.xlsx 核查与 id 对照
├─ inspect_att34.py              # 附件3/4 多文件结构探查
├─ probe_att34.py                # 附件3/4 深度探查（逐文件 N、字段、缺失分布）
├─ probe_text.py                 # 文本缺失表示方式探查 + HF 连通性
├─ encode_att_text.py            # 用 bert-base-uncased 为附件3/4 生成 text(768) 特征
├─ verify_text_encoder.py        # 验证 BERT 编码与附件2 text 同源（实测余弦 1.0000）
├─ download_bert.py              # 通过 hf-mirror.com 下载 BERT
├─ diagnose_nan.py               # 训练 NaN 定位工具
├─ check_deliverables.py         # 交付物与模型体积审计
├─ _env_check.py                 # 环境自检
└─ 02_model/
   ├─ config.py                  # 全部超参与路径（改这里，不改代码）
   ├─ data_utils.py              # pkl 读取 / 有效长度推断 / 缺失检测 / 归一化 / 多文件拼接
   ├─ missing_sim.py             # 缺失模拟引擎（类型×位置×率×短碎片）+ 时间裁剪增强
   ├─ model.py                   # MRF-Net 骨干
   ├─ losses.py                  # 损失与指标（ACC/F1/MAE/Pearson/CCC、阈值搜索）
   ├─ runtime.py                 # 共享推理运行时（集成加载 + 场景推理）
   ├─ train.py                   # 教师→学生训练 + 评价 + 缺失因素扫描
   ├─ infer_att3.py              # 附件3/4 推理与结果 CSV
   ├─ calibrate_missing.py       # 缺失分布标定 + 反推训练模拟参数
   ├─ viz_results.py             # 论文图：混淆矩阵/散点/误差/缺失可视化/曲线
   ├─ error_analysis.py          # 5 维分层错误归因 + top-K 误差清单
   ├─ analyze_sweep.py           # ANOVA + 退化曲线拟合
   └─ tune_hyperparams.py        # valid 上的结构/超参搜索
```

## 附件3/4 使用流程（结构特殊，见文档 06）

附件3/4 与附件2 结构不同：**多文件、每文件 1 样本、无 `id`、无预计算的 `text`(768)**。

```powershell
# 1) 结构探查
python att_paths.py
python probe_att34.py

# 2) 验证文本特征同源（BERT 最后一层隐状态 vs 附件2 text）
python verify_text_encoder.py --n 8

# 3) 生成附件3/4 的文本特征（输出 data_att/att3_aligned.pkl、att4_aligned.pkl）
python encode_att_text.py --kind att3 --version aligned
python encode_att_text.py --kind att4 --version aligned

# 4) 缺失分布标定（输出 runs/q2/att3_missing_report.json）
python 02_model\calibrate_missing.py --pkl ..\data_att\att3_aligned.pkl --out_dir ..\runs\q2

# 5) 正式推理（用训练好的 runs/q2 模型）
python 02_model\infer_att3.py --pkl ..\data_att\att3_aligned.pkl --ckpt_dir ..\runs\q2 ^
  --out_csv ..\submission\pred_att3.csv --device cpu
python 02_model\infer_att3.py --pkl ..\data_att\att4_aligned.pkl --ckpt_dir ..\runs\q2 ^
  --out_csv ..\submission\pred_att4.csv --device cpu
```

> 完整的完成度审计与端到端实施流程见 `解决方案/05_问题2完成度审计与完整实施流程.md`；
> 附件3/4 的结构差异与文本特征方案见 `解决方案/06_附件3附件4数据核查与处理方案.md`。


## 2. 环境

```powershell
cd e:\数学建模\code
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python _env_check.py    # 结果写入 _env_check.txt
```

## 3. 数据放置

**方式一（推荐）**：把附件2 的 pkl 放在工作区任意位置（例如 `e:\数学建模\附件2\aligned_50.pkl`），
程序会自动搜索 `<version>_50.pkl`（跳过 `data_dummy`），无需手写中文路径。

**方式二**：显式指定

```
data\aligned_50.pkl        # 或 unaligned_50.pkl
python 02_model\train.py --data_dir "E:\数学建模\附件2"
```

**版本必须全程一致**：`--version aligned` 或 `--version unaligned`（当前附件只提供了 aligned）。

## 4. 运行

```powershell
# 0) 数据核查（先跑这个，产出论文"数据说明"小节所需数字）
python code\inspect_data.py --version aligned --save runs\data_report_real.json

# 1) 快速自检（合成数据，约 3 分钟，验证环境与全流程）
code\run_smoke_test.cmd

# 2) 正式训练（真实数据，5 个随机种子）
python code\02_model\train.py --version aligned --epochs_teacher 20 --epochs_student 30

# 3) 训练 + 缺失因素扫描（第 4 节规律分析）
python code\02_model\train.py --version aligned --sweep 1

# 4) 附件3 推理（多 ckpt 集成）
python code\02_model\infer_att3.py --pkl "路径\att3.pkl" ^
  --ckpt_dir runs\q2 --out_csv submission\pred_att3.csv --device cpu

# 5) 缺失因素统计分析（ANOVA + 退化曲线 + 图）
python code\02_model\analyze_sweep.py --csv runs\q2\sweep_valid_s42.csv
```

### 实测速度与时间预估（本机 12 线程 CPU，无 GPU）

| 阶段 | 单轮耗时 | 推荐轮数 | 单种子耗时 |
|---|---|---|---|
| 教师（全模态） | 98.6 s | 20–30 | 33–50 min |
| 学生（缺失+蒸馏） | 103 s | 30–50 | 52–86 min |

→ **单种子约 1.5–2 h，3 个种子约 5–6 h，5 个种子约 8–10 h**。
时间紧张时：`--hidden 96 --batch_size 128`，或先跑 1 个种子出结果，再用 3 个种子复现。

输出文件（默认落在 `runs\q2\`）：

| 文件 | 内容 |
|---|---|
| `student_s{seed}.pt` | 学生模型权重 + 归一化统计 + 阈值 + 配置（推理所需全部信息） |
| `history_s{seed}.csv` | 每个 epoch 的损失明细（reg/cls/rec/sim/diff/bal/kd）、valid MAE、当前课程参数 |
| `metrics_s{seed}.json` | valid/test 的 MAE/Pearson/CCC/ACC/Macro-F1 |
| `sweep_valid_s{seed}.csv` | 缺失因素扫描结果（含 aware / unaware 两列，用于"A0 性能急剧下降"对比） |
| `summary.json` | 多随机种子的聚合结果 |

## 5. 关键设计（对应论文的第 3 节）

| 模块 | 代码位置 | 作用 |
|---|---|---|
| 掩码感知编码 | `model.ModalityEncoder` | 把 `miss` 指示位与特征拼接，模型显式知道哪段缺失 |
| 跨模态重建 | `model.MRFNet.forward` 第 1 步 + `losses.recon_loss(missing=True)` | 只对缺失位置算重建损失（自监督，无额外数据） |
| 代理令牌 | `model.MRFNet.forward` 第 3 步（`beta` 门控） | 整模态缺失或重建不可靠时退回模态先验 |
| 共享/特有分解 | `share` / `spec` / `mix` + `sim_loss` / `diff_loss` | 模态不变表示 + 特有表示正交 |
| 动态专家 | `model.Expert` + `gate` + `balance_loss` | 按模态可用性路由，避免专家坍缩 |
| 不确定性 | `head_mu` / `head_lv` + `reg_nll` | 输出预测方差，可作置信度 |
| 知识蒸馏 | `losses.total_loss` 中 `teacher_out` 分支 | logit + 回归 + **关系蒸馏**（样本间相似度矩阵），权重随 epoch 指数衰减 |
| 课程学习 | `train.curriculum` | 缺失率 0→0.5 三阶段渐进 |
| 缺失模拟 | `missing_sim.inject_missing` | 训练时按 类型×位置×率 注入连续区间缺失 |
| 有效长度推断 | `data_utils.infer_length_from_zeros` | 附件2 无 `*_lengths` 字段，用尾部全零推断；**填充 ≠ 缺失** |
| 类别平衡 | `config.cls_balanced` + `losses.cls_ce` | 按训练集类频倒数加权 CE（评测为 Macro-F1，中性类易被忽略） |
| 三因素扫描 | `missing_sim.make_scenario_masks` + `train.run_experiment` 第 4 步 | 生成 ANOVA 所需的数据表 |

## 6. 与题目要求的对应

| 题目要求 | 落地位置 |
|---|---|
| 鲁棒模型原理/结构/目标函数/训练方案 | `model.py`、`losses.py`、`config.py`、`train.py` |
| 缺失类型、缺失率影响规律 + 消融 | `--sweep 1` 产出 `sweep_valid_*.csv` → 用 `statsmodels` 做 Type-III ANOVA |
| 附件3 全量预测结果 | `infer_att3.py` → `pred_att3.csv`（+ `_full.csv` 含缺失区间） |
| 验证集基础性能 / 可视化 / 错误归因 | `metrics_*.json` + 用 `pred` 数组自行出图（散点、混淆矩阵、分层统计） |
| 训练/验证/测试同一版本、同一接口 | `config.version` 单点控制；`infer_att3.py` 读取 ckpt 内配置，天然一致 |

## 7. 消融实验怎么做

`config.py` 里把对应损失权重置 0，或在 `model.py` 里关掉对应模块，然后重跑同一 seed：

| 消融 | 操作 |
|---|---|
| A0 无任何处理 | `infer_att3.py` 用 `--unaware`（特征置零但模型收不到掩码）——体现在 `sweep` 的 `unaware_*` 列 |
| A2 去掉掩码感知 | `ModalityEncoder.forward` 里把拼接的两维指示位去掉 |
| A3 去掉重建 | `config.lam_rec = 0, lam_rec0 = 0` |
| A4 去掉分解 | `config.lam_sim = 0, lam_diff = 0` |
| A5 去掉代理令牌 | `forward` 中跳过 beta 门控 |
| A6 去掉专家 | `n_experts = 1` |
| A7 去掉蒸馏 | `--no_distill` |
| A8 去掉不确定性 | `reg_nll` 换成 `mse` |
| A9 无课程学习 | `curric` 只保留一个区间 |

## 8. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `FileNotFoundError` | `--data_dir` 路径不对，或文件名不是 `aligned_50.pkl` / `unaligned_50.pkl` |
| 报 `KeyError: 'train'` | 该 pkl 是附件3/4 的单 split 文件，请用 `infer_att3.py` 而非 `train.py` |
| valid MAE 一直不降 | 先确认归一化是否生效；再把 `lr_backbone` 降到 5e-5、`hidden` 提到 256 |
| 显存不足 | 降 `--batch_size`，或把 `hidden` 降到 64 |
| 终端无输出 | 本机 PowerShell 偶发静默失败：改用任务或 `python xxx.py > log.txt 2>&1` 后读日志 |
