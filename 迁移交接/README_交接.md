# 迁移交接文档（多模态情感预测 · 2026 研赛 E 题 · 问题2）

> 生成时间：2026-09-23　｜　源机器：CPU 12 线程无 GPU　｜　目标机器：CPU 更强
> 本文件的作用：**替代无法迁移的对话记录**——新机器上的 AI 助手读完本文件即可无缝接手。

---

## 0. 先说清楚：什么能迁移、什么不能

| 项 | 能否迁移 | 说明 |
|---|---|---|
| **工作区文件** | ✅ 能 | 见第 2 节，必需部分 **3.79 GB**（含 BERT 缓存 4.21 GB；去掉 unaligned 附件2 可降到 ~1.2 GB） |
| **代码 / 方案文档 / 结果 / 模型权重** | ✅ 能 | 全部是普通文件 |
| **Copilot 对话记录** | ❌ **不能** | 会话存在本机 `AppData\Roaming\Code\User\workspaceStorage\...`，无官方导出/导入方式 |
| **挂起进程内存里的 v3 教师模型** | ❌ **不能** | 已按决定**主动终止**（教师 best MAE 0.5862 未落盘）→ 新机器上**重跑 v3** 即可（CPU 更强会更快） |
| **Python 环境** | ⚠️ 建议重装 | 用 `requirements.txt` 重装比复制 site-packages 可靠 |

**替代方案**：本文件 + 第 9 节的「交接提示词」。新机器上把提示词粘给 AI，它就能接着干。

---

## 1. 项目概况

- 赛题：**2026 中国研究生数学建模竞赛 E 题**——复杂场景下多模态情感预测的数学建模与算法设计
- 数据：CMU-MOSEI 系列（附件1 原始视频 100 条 / 附件2 标准化特征 4850 条 / 附件3 模态缺失测试集 / 附件4 可解释测试集）
- 本次专注：**问题2（模态信息缺失条件下的情感预测建模与验证）**，问题1/3 暂缓
- 评价指标（题目要求）：极性分类 → **Accuracy、F1**；强度回归 → **MAE、Pearson**；决策阈值在 valid 上选
- 全程使用 **aligned（对齐）版本**，K=50 个时间步

### 当前完成度

| 层面 | 完成度 |
|---|---|
| 代码层 | 100% |
| 数据层 | 100%（附件2/3/4 全部核查完毕，附件3/4 缺失文本特征已补齐） |
| 实验层 | ~45%（主训练进行中；超参搜索/缺失扫描/消融未跑） |
| 论文层 | ~15%（图与表脚本就绪并实测通过，正文未写） |

### 当前最好成绩（v2 模型，单种子）

| 指标 | valid | test |
|---|---|---|
| MAE | **0.5873** | **0.6521** |
| Pearson | 0.661 | 0.665 |
| CCC | 0.616 | 0.613 |
| Accuracy（θ=0.30） | 0.5962 | **0.6135** |
| Macro-F1（θ=0.30） | 0.5862 | **0.5935** |
| F1 neg/neu/pos | 0.655/0.447/0.656 | 0.706/0.397/0.677 |

参照基线：预测均值 MAE≈1.49；全判多数类 ACC≈0.30；随机 Macro-F1≈0.33。

---

## 2. 迁移文件清单

| 组 | 路径 | 大小 | 必要性 |
|---|---|---|---|
| 赛题与方案 | `复杂场景…docx`、`赛事文件.txt`、`解决方案/`（6 份 MD） | 1.1 MB | **必需** |
| 代码 | `code/`（含 `02_model/`，13 个模块 + 15 个工具脚本） | 0.3 MB | **必需** |
| 数据 | `附件2/`（aligned 993.8 MB + unaligned 2897 MB + label.xlsx） | 3711 MB | **必需，全量迁移**（按决定 unaligned 一起带走，备选版本） |
| 数据 | `附件3-模态缺失特征样本/`（对齐+未对齐各 30 个 pkl） | 15.6 MB | **必需** |
| 数据 | `附件4-可解释专项视频样本与特征文件/`（各 20 个 pkl + videos/） | 63.5 MB | **必需**（问题3 用） |
| 中间产物 | `data_att/att3_aligned.pkl`、`att4_aligned.pkl` | 8.4 MB | **必需**（已补齐的文本特征） |
| 中间产物 | `runs/`（checkpoint 6.5 MB/个 + metrics + sweep + 日志） | 83.9 MB | **必需** |
| 中间产物 | `submission/`、`figs/`、`paper_tables/` | 0.6 MB | **必需** |
| 模型缓存 | `E:\hf_cache`（bert-base-uncased） | 420.7 MB | 可选（新机器可重下） |
| 临时 | `data_dummy/`（107.8 MB） | — | **建议不迁**，用 `make_dummy_data.py` 重建 |

**合计：必需 3.79 GB（附件2 全量含 unaligned）；含 BERT 缓存 4.21 GB。**

> 📌 **迁移决定（2026-09-23）**：
> 1. 挂起的 v3 训练进程**已主动终止**（释放 2 GB 内存）→ 新机器上重跑 v3；
> 2. **附件2 全量迁移**（aligned + unaligned 一起带走，共 3.7 GB）。
> 因此整包约 **4.2 GB**（含 BERT 缓存）。
一键复制脚本已生成：`迁移交接/copy_to_usb.cmd`（改里面的 `TARGET` 后运行，用 robocopy 复制）。

> ⚠️ 路径建议：新机器上**保持同样的目录名**（例如 `D:\数学建模`）。代码里 `att_paths.py` / `Config.autodetect()` 会自动搜索附件，路径不同也能跑；但 `config.py` 里默认 `data_dir` 是 `E:\数学建模\data`，靠自动发现兜底。

---

## 3. 环境搭建（新机器）

```powershell
# 1) Python 3.12（或 3.10+）+ pip
python -V

# 2) 依赖（用清华镜像，本机实测可用）
cd code
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install transformers huggingface_hub safetensors -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) 自检
python _env_check.py && type _env_check.txt
```

**网络要点（源机器实测，新机器需重测）**
- ❌ `huggingface.co` 直连被拒（代理未运行）
- ✅ **`hf-mirror.com` 可用** → 所有涉及 BERT 的脚本已内置：
  `HF_ENDPOINT=https://hf-mirror.com`、`HF_HOME=<磁盘>\hf_cache`
- ✅ PyPI 清华镜像可用；`modelscope.cn` 也可用
- BERT 下载：`python download_bert.py`（若已复制 `hf_cache` 则把 `HF_HOME` 指向它即可）

**依赖清单**：`numpy / pandas / scikit-learn / scipy / torch(CPU) / openpyxl / matplotlib / tqdm / transformers / huggingface_hub / safetensors / statsmodels`
源机器实测版本：`numpy 2.5.3 / scipy 1.18.1 / sklearn 1.9.1 / pandas 3.0.6 / torch 2.14.0+cpu / matplotlib 3.11.2 / transformers 5.17.0`

---

## 4. 目录结构

```
数学建模/
├─ 复杂场景…docx / 赛事文件.txt          # 赛题
├─ 解决方案/                             # 00~06 六份方案与审计文档
├─ code/                                 # 全部代码
│  ├─ att_paths.py                       # 附件3/4 路径自动发现（目录名含中文）
│  ├─ inspect_data.py / inspect_att34.py / probe_att34.py / probe_text.py
│  ├─ verify_text_encoder.py             # ★ 验证 BERT 编码与附件2 text 同源（余弦 1.0000）
│  ├─ encode_att_text.py                 # ★ 为附件3/4 生成 text(768)
│  ├─ download_bert.py / diagnose_nan.py / check_deliverables.py
│  ├─ monitor_progress.py + monitor.cmd  # ★ 实时进度仪表盘
│  ├─ pause_training.ps1 / resume_training.ps1
│  ├─ make_migration_package.py
│  └─ 02_model/                          # config, data_utils, missing_sim, model, losses,
│                                        # runtime, train, infer_att3, calibrate_missing,
│                                        # viz_results, error_analysis, analyze_sweep,
│                                        # tune_hyperparams, make_tables
├─ 附件2/ 附件3-…/ 附件4-…/               # 数据
├─ data_att/                             # 附件3/4 文本特征（必带）
├─ runs/                                 # 权重与实验记录（必带）
├─ submission/ figs/ paper_tables/       # 交付物
└─ 迁移交接/                              # 本交接包
```

---

## 5. 关键数据事实（已实测，直接写进论文）

### 5.1 附件2（aligned_50.pkl，993.8 MB）
- 顶层 `train/valid/test`；字段：`raw_text, audio, vision, id, text_bert, classification_labels, regression_labels, text`
- ⚠️ **没有 `audio_lengths`/`vision_lengths`/`annotations`**（与题目表2 描述不符）→ 有效长度只能由"尾部连续全零"推断
- 规模 3395/728/727 = 4850；文本 (N,50,768)、语音 (N,50,74)、视觉 (N,50,35)
- **填充率 ~0.49–0.54**（一半位置是尾部填充，有效长度均值 ~24/50）
- **有效区间内真缺失率≈0**（text 0.0000 / audio 0.0004 / vision ~0.010）→ 附件2 本身几乎无缺失
- 标签 `classification_labels ∈ {0,1,2}`，分布 967/758/1670（中性仅 22%）
- unaligned 版：audio/vision 为 (N,500,·) 且**含 `*_lengths`**（max=500，语音 100Hz→仅 5s，存在尾部截断）

### 5.2 附件3 / 附件4（结构与附件2 **完全不同**）
- 附件3 = **30 个 pkl**，附件4 = **20 个 pkl**，**每文件只有 1 个样本**，各有「对齐版本/未对齐版本」
- 附件4 另带 `videos/` 20 个原始视频（问题3 关键帧回看用）
- ⚠️ **都没有 `id`** → 只能用文件名作编号（`附件3_01…30` / `01…20`）
- ⚠️ **都没有预计算的 `text`(768)**：对齐版只有 `text_bert(1,3,50)`；未对齐版只有 `raw_text`
- ⚠️ 附件4 的数组是**二维存储**（如 `audio(50,74)`），样本数判定要小心

### 5.3 ★ 文本特征同源验证（最关键的技术突破）
用 `bert-base-uncased` 的 **`last_hidden_state`** 编码附件2 的 `text_bert`，与附件2 提供的 `text` 对比：

| 方案 | 平均余弦 |
|---|---|
| **last_hidden_state** | **1.0000**（逐维相关 1.0，std/mean 完全一致） |
| hidden_states[-2] | 0.8151 |
| 最后 4 层均值 | 0.8509 |

→ 附件2 的 `text` 就是 BERT 最后一层隐状态。**因此用同一 BERT 为附件3/4 生成文本特征，训练好的模型无需重训即可推理**，同时满足题目"同一输入接口"要求。
→ 已生成 `data_att/att3_aligned.pkl`(30 条)、`data_att/att4_aligned.pkl`(20 条)

### 5.4 附件3 缺失分布（已标定，训练模拟参数据此设定）
| 模态 | 含缺失样本 | 缺失率 mean | p90 | max | 区间长 | 位置 头/中/尾 |
|---|---|---|---|---|---|---|
| text | **0/30（不可观测）** | 0 | 0 | 0 | — | — |
| audio | 17/30（56.7%） | 0.101 | 0.224 | 0.400 | ≈2 步 | 0.37/0.56/0.07 |
| vision | 18/30（60.0%） | 0.107 | 0.224 | 0.400 | ≈2 步 | 0.36/0.57/0.07 |

⚠️ **附件3 的文本缺失不可观测**：30 条样本的 `text_bert` 在有效区间内无任何全零列。
处理：文本按完整处理 + **训练时仍强制模拟文本缺失**兜底；论文必须写明此局限性。

---

## 6. 已完成的工作

### 6.1 代码（13 个模块 + 15 个工具脚本，全部实测跑通）

| 模块 | 职责 |
|---|---|
| `config.py` | 全部超参/路径/课程表，单点配置；含真实数据自动发现 |
| `data_utils.py` | pkl 读取（多文件/单文件、无标签兼容）、**有效长度推断**、**pad 与 miss 严格分离**、归一化、Dataset |
| `missing_sim.py` | 缺失注入（类型×位置×率×**短碎片 1~3 段**）+ 时间裁剪增强 + 场景掩码 |
| `model.py` | **MRF-Net**：掩码感知编码 → 跨模态重建 → 代理令牌 → 共享/特有分解 → 动态专家 MoE → 不确定性双头 |
| `losses.py` | SmoothL1(主) + NLL(辅 0.1) + 类平衡 CE + 重建/相似/正交/负载均衡 + 蒸馏；指标（含 **θ 口径与分类头口径双报告**） |
| `runtime.py` | 集成推理运行时（多 ckpt + 缺失场景 + unaware 基线） |
| `train.py` | 教师(全模态) → 学生(课程学习+缺失注入+蒸馏)；早停、fp16 保存、缺失因素扫描 |
| `infer_att3.py` | 附件3/4 推理 → CSV（含缺失率、缺失区间、置信度） |
| `calibrate_missing.py` | 缺失分布标定 + **反推训练模拟参数** |
| `viz_results.py` | 论文图：混淆矩阵/散点/误差分布/缺失可视化/训练曲线/退化曲线 |
| `error_analysis.py` | 5 维分层错误归因 + top-K 误差清单 |
| `analyze_sweep.py` | Type-III ANOVA + 三次多项式退化曲线拟合 |
| `tune_hyperparams.py` | valid 上的结构/超参搜索（8 组小网格 / 32 组全网格） |
| `make_tables.py` | 论文表格汇总导出（Markdown + CSV） |

### 6.2 已产出交付物

| 交付物 | 路径 | 状态 |
|---|---|---|
| 附件3 预测结果 | `submission/pred_att3.csv` + `_full.csv` | ✅ 30 行（θ=0.30） |
| 附件4 预测结果 | `submission/pred_att4.csv` + `_full.csv` | ✅ 20 行 |
| 模型权重 | `runs/q2/student_s42.pt`（float16, 6.47 MB） | ✅ v2 单种子 |
| 指标 | `runs/q2/metrics_s42.json`、`figs/test_metrics_q2.json` | ✅ |
| 论文图 | `figs/*_q2.png`（混淆矩阵/散点/误差/缺失可视化/训练曲线） | ✅ |
| 错误归因 | `runs/q2/error_groups_{valid,test}.csv`、`error_top20_*.csv` | ✅ |
| 论文表格 | `paper_tables/tables.md` + 4 个 CSV | ✅ |
| 附件3 缺失报告 | `runs/q2/att3_missing_report.json` + `att3_missing_per_sample.csv` | ✅ |

### 6.3 已修复的真实缺陷（教训，别重犯）

1. **附件2 无 `*_lengths` 字段** → 改用"尾部全零推断有效长度"
2. **pad（尾部填充）与 miss（真缺失）必须严格分离**，否则重建损失会算到填充区
3. **蒸馏权重过大导致学生完全不收敛** → `lam_kd0` 1.0→0.3、关系蒸馏×0.1、前 3 轮热启动（合成数据 MAE 0.41→0.19）
4. **时间裁剪导致 NaN**：附件2 的有效数据是**前缀结构**，裁剪后部分样本有效位为空 → 注意力全遮 → NaN。修复：裁剪窗口限制在各自原有效区间内 + 新增 `ensure_valid()` 安全位（可用 `diagnose_nan.py` 复现/验证）
5. **NLL 方差项"吸收"损失**：训练 loss 降但 valid MAE 升 → 改为 **SmoothL1(β=0.5) 主损失 + NLL 权重 0.1**
6. **过拟合**：最优只出现在第 3–6 轮 → 强正则（dropout 0.35 / wd 5e-4 / 输入噪声 0.15 / 标签平滑 0.1）：valid MAE 0.5926 → **0.5862**
7. **两套极性口径混用**（日志用分类头 argmax，提交用 θ 阈值）→ 改为同时报告，**θ 口径为主**（与提交一致）
8. checkpoint 13.5 MB × 5 种子超 50 MB 预算 → 改 **float16**（6.47 MB）
9. matplotlib 中文与样本 id 中的 `$`（`xxx$_$yyy`）导致崩溃 → 字体设置 + `$` 转义；`boxplot(labels=)` 在 mpl 3.11 已改名 `tick_labels=`
10. **`taskkill /IM python.exe` 不会终止 `.cmd` 父脚本** → 它继续执行下一行产生"孤儿训练进程"（曾两个 train.py 并行）；且会顺带杀掉监控脚本。查进程用
    `Get-CimInstance Win32_Process -Filter "name='python.exe'" | Select ProcessId,CreationDate,CommandLine`

---

## 7. 当前进度与后续命令

### 迁移时的进度快照（2026-09-23 19:5x）
- **v2（弱正则）seed 42 已完成** → `runs/q2/student_s42.pt`（valid MAE 0.5873 / test 0.6521）
  - 配套产物均已落盘：`metrics_s42.json`、`history_s42.csv`、`error_groups_*.csv`、`error_top20_*.csv`、`att3_missing_report.json`
- **v3（强正则）seed 42：已主动终止**
  - 终止时的进度：教师已完成（ep18 早停，**best valid MAE = 0.5862**，优于 v2 的 0.5926），学生跑到 ep02
  - `runs/q2v3/` 为空（每次种子全部跑完才落盘），**这份权重已丢失，需在新机器重跑**
- **结论**：新机器上的主训练目标就是重跑 v3（2 种子），并把产物输出到 `runs/q2v3`

### 新机器上的推荐执行顺序

```powershell
cd <新工作区>\code
python _env_check.py                                  # ① 环境自检
python check_deliverables.py                          # ② 交付物清单
python inspect_data.py --version aligned              # ③ 数据核查
python monitor.cmd 20                                 # ④ 开监控（另开一个终端）

# ⑤ 主训练（v3 强正则，2 种子 ~3 h；CPU 更强会更快）
python 02_model\train.py --version aligned --epochs_teacher 20 --epochs_student 30 --seed 42 ^
  --hidden 128 --batch_size 64 --dropout 0.35 --weight_decay 5e-4 ^
  --lr_backbone 5e-5 --lr_head 5e-4 --noise_sigma 0.15 --label_smoothing 0.1 --out_dir ..\runs\q2v3
python 02_model\train.py ... --seed 43 ... --out_dir ..\runs\q2v3     # 同上，换 seed

# ⑥ 用新模型重新推理附件3/4
python 02_model\infer_att3.py --pkl ..\data_att\att3_aligned.pkl --ckpt_dir ..\runs\q2v3 --out_csv ..\submission\pred_att3.csv
python 02_model\infer_att3.py --pkl ..\data_att\att4_aligned.pkl --ckpt_dir ..\runs\q2v3 --out_csv ..\submission\pred_att4.csv

# ⑦ 图表与归因
python 02_model\viz_results.py --ckpt_dir ..\runs\q2v3 --out_dir ..\figs --tag v3
python 02_model\error_analysis.py --ckpt_dir ..\runs\q2v3 --out_dir ..\runs\q2v3

# ⑧ 题目要求的实验（尚未做）
python 02_model\train.py --version aligned --sweep 1 --out_dir ..\runs\q2v3     # 缺失因素扫描
python 02_model\analyze_sweep.py --csv ..\runs\q2v3\sweep_valid_s42.csv --out_dir ..\figs
python 02_model\tune_hyperparams.py --stage screen                              # 超参搜索
python 02_model\train.py --no_distill ... --out_dir ..\runs\ablate_nokd         # 消融 A7
python 02_model\make_tables.py                                                  # 论文表格
```

---

## 8. 待办清单（题目要求但未完成）

| 优先级 | 事项 | 脚本 | 预计耗时 |
|---|---|---|---|
| P0 | 主训练（v3 强正则，2 种子） | `train.py` | 2–3 h |
| P0 | 附件3/4 用最终模型重新推理 | `infer_att3.py` | 5 min |
| P0 | **缺失类型/缺失率影响规律 + 消融**（题目第2项核心） | `train.py --sweep 1` + `analyze_sweep.py` | 2 h |
| P0 | **超参搜索**（题目要求 valid 上选） | `tune_hyperparams.py` | 2–3 h |
| P1 | 消融 A0–A9 | `train.py` 各开关 | 3 h |
| P1 | 论文正文（问题2 章节） | — | 2 h |
| P2 | `requirements_lock.txt`（pip freeze） | — | 5 min |
| P2 | 附件打包 + 匿名性检查（≤50 MB，禁单位/姓名/队号） | — | 30 min |

### 模型已知问题（优化方向，见 `解决方案/05` 第 3 节）
- 过拟合未根除（v3 也只是 +1%）；MAE 0.6521 仍高于文献 ~0.55–0.58
- **向均值收缩**：负向 bias +0.61、正向 −0.54 → 可用"分数重标定"直接修
- 中性类最弱（F1 0.397）；θ 目前按 Macro-F1 选，若主报 ACC 需另选 θ
- valid(728)/test(727) 样本量小 → ACC 标准误 ≈1.8%，**<2% 的提升需 McNemar 检验**

---

## 9. 交接提示词（复制粘贴给新机器的 AI 助手）

```
我在做 2026 研究生数学建模竞赛 E 题「复杂场景下多模态情感预测」，当前专注**问题2**
（模态缺失鲁棒情感预测）。工作区就在当前目录，请先完整阅读：

1. `迁移交接/README_交接.md`（交接总纲：数据事实、已完成工作、缺陷与坑、待办）
2. `解决方案/05_问题2完成度审计与完整实施流程.md`（完成度审计 + 端到端流程）
3. `解决方案/06_附件3附件4数据核查与处理方案.md`（附件3/4 结构差异与文本特征方案）
4. `code/README.md`（代码结构与运行方式）

请遵守以下既有约定（已踩过坑）：
- 全程使用**对齐（aligned）版本**；附件3/4 已用 bert-base-uncased 补齐 text(768)，
  特征文件在 `data_att/`（不要重新生成，除非有明确理由）
- 报告中**极性指标主用 θ 阈值口径**（与提交 CSV 一致），同时给出分类头口径（acc_head/f1_head）
- 缺失模拟参数据附件3 实测分布设定（见 README_交接.md 5.4）
- 附件3 的文本缺失**不可观测**，论文必须写明该局限性

当前状态：v2 模型已完成（`runs/q2/student_s42.pt`，test MAE 0.6521 / ACC 0.6135 / MacroF1 0.5935），
附件3/4 的预测 CSV 已产出。**下一步请按 README_交接.md 第 7 节执行**：
先重跑 v3 强正则训练（2 种子）→ 重新推理 → 做题目要求的缺失因素扫描 + 超参搜索 + 消融 → 出图出表。

注意本机环境：Python 3.12 + torch CPU；huggingface.co 直连不通，必须用
`HF_ENDPOINT=https://hf-mirror.com`（脚本已内置）；长任务请写日志文件再读，不要依赖终端回显。
```

---

## 10. 关键坑速查（新机器沿用）

| 现象 | 处理 |
|---|---|
| 终端命令无回显 / 静默失败 | 改为"写 .cmd + 重定向日志 + 读文件"；用任务方式执行而非交互终端 |
| `.cmd` 里写中文路径报"系统找不到指定的路径" | cmd.exe 按 OEM 代码页读文件 → 用 `%~dp0` 相对路径；Python 源文件里用中文没问题 |
| 长任务输出看不到 | `python -u` + 重定向；或等任务结束后读日志 |
| 杀进程后出现"孤儿训练进程" | `taskkill /IM python.exe` 不杀父 `.cmd` → 用 `Get-CimInstance Win32_Process` 定位 PID 后 `Stop-Process`/`taskkill /PID` |
| 想临时让出 CPU | `code\pause_training.ps1`（NtSuspendProcess 挂起，可恢复）/ `resume_training.ps1` |
| 离线无法下载 BERT | 把 `E:\hf_cache`（420 MB）一起拷过去，并设 `HF_HOME` 指向它 |
| 训练出 NaN | `python diagnose_nan.py` 逐阶段定位（历史原因见 6.3 第 4 条） |
