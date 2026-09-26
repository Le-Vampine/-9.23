# 问题3 代码使用说明（可解释性情感预测）

> 状态：在真实数据（附件2 aligned、附件3/4 修正口径）上端到端跑通；预测与问题2 **逐位一致**
> （生成 `submission/pred_explain_att4.csv` 时内建断言校验：分类标签一致率 1.000、强度分数最大差 < 1e-3）。

## 0. 一句话设计

**不改预测、只加解释**：骨干（`runs/ens_top2` 两种子集成）全程冻结，解释由三层构成——
模态级用可精确枚举的 Shapley 值（3 模态 8 子集）、片段级用积分梯度与遮挡、证据级回映到
原始文本字符区间 / 语音秒区间 / 视频关键帧。解释路径可训练（并联证据头 + 解释性正则），
但不接入预测通路，故问题三的预测与问题二完全一致。

## 1. 目录与职责

```
code/03_xai/
├─ fetch_assets.py          # 下载 BERT 词表（226 KB，行号即 token id，无需装 transformers）
├─ att_meta.py              # S0：抽取 text_bert/raw_text/视频时长，修正文本有效长度口径（D1）
├─ d1_recalib.py            # D1=A：用修正口径重出 pred_att3.csv / pred_att4.csv 并出对比报告
├─ xai_common.py            # 骨干加载 / batch 构造（含掩码置零，与推理链路一致）/ 屏蔽与遮挡算子
├─ style.py                 # 统一绘图风格（宋体+Times，四色学术配色）
├─ shapley.py               # S1：精确 Shapley（4 口径组合、样本级+数据集级、精确性自检）
├─ ig_gradcam.py            # S2：积分梯度 / 注意力 / 注意力×梯度
├─ occlusion.py             # S3：窗口遮挡重要性（模型无关基准）
├─ temporal_fuse.py         # S4a：三源归一化与 γ 标定（单纯形枚举，约束 γ≥0.1）
├─ metrics_xai.py           # S4b：忠实性/充分性/优劣 AUC/稀疏性/稳定性
├─ evidence.py              # S5：词元解码→词/字符区间、段号→秒、视频关键帧与音频能量
├─ explain_card.py          # S6a：解释卡 JSON 与四联图
├─ infer_explain_att4.py    # S6b：主入口 → submission/pred_explain_att4.csv + explain_cards/
├─ q3_valid_analysis.py     # S7：分层统计、八子集反事实、模态忠实性、一致性检验、错误归因
├─ paper_figs_q3.py         # S8：论文图（Q3_{内容}_{图表类型}.png）与说明文字
├─ verify_q3.py             # S9：交付自查（行数/口径/证据合法性/体积/匿名）
├─ train_evidence_head.py   # D4/D9：并联证据头训练（冻结骨干 + 解释性正则）
├─ make_handover.py         # 打包 handover_problem3/
└─ run_pipeline.py          # 一键串起全部 17 步，逐步落日志到 runs/q3/logs/
```

## 2. 一键复现

```powershell
cd <仓库根>\code\03_xai

# ① 环境与静态资源（一次性）
uv pip install --python D:\venv-mosei\Scripts\python.exe imageio-ffmpeg
python fetch_assets.py

# ② 口径修正（决策 D1=A；会备份原缓存与提交文件到 _backup_preD1/）
python att_meta.py --kind both
python d1_recalib.py

# ③ 主流程（17 步，CPU 约 20 分钟；日志在 runs\q3\logs\）
python run_pipeline.py

# ④ 可解释性证据头（可选，冻结骨干只训小头，约 10 分钟）
python train_evidence_head.py --epochs 12

# ⑤ 交付自查与打包
python verify_q3.py
python make_handover.py
```

## 3. 关键参数与口径（写入论文的可复现信息）

| 项 | 取值 | 说明 |
|---|---|---|
| 骨干 | `runs/ens_top2`（`student_s42.pt` + `student_s43.pt`） | 与问题2 主模型同一份权重 |
| 决策阈值 θ | 0.325 | 仅由验证集选定（问题2 口径） |
| Shapley 目标 | 强度 μ（主）、极性 $p_{+}-p_{-}$（对照） | 4 个口径组合全部报告 |
| 屏蔽语义 | `miss=1`（主，模型训练语义）、`pad=1`（对照） | 两者差异作为稳健性证据 |
| π 归一化 | softmax$\lvert\varphi\rvert$（主，无退化）、ReLU 归一（对照，如实报告退化样本数） | 择优依据为与真实删模态退化的秩相关 |
| 积分梯度 | 步数 20，基线=训练集均值（归一化空间零向量） | 另报告零基线对照 |
| 遮挡 | 窗口 3 步，用相邻段替换；att4 步长 1、valid 步长 2 | 保持序列长度，避免长度伪影 |
| γ 融合 | 单纯形枚举（每源 ≥ 0.1），以与遮挡源秩相关最大为准则 | 等权方案作对照 |
| 解释质量 | top-$r$ 取 $r=5$；稳定性扰动 $\sigma=0.05$ | 删除/插入曲线 5 段积分 |
| 随机性 | 除遮挡窗口顺序外无随机；证据头训练设固定随机种子 | 结果可复现 |

## 4. 与问题2 的接口

* 只读问题2 的权重、`data_utils.py`、`runtime.load_ensemble`、`train.prep/stack_masks/unstack_masks`，
  **不修改**任何问题2 代码；
* 输入数据 `data_att/att3_aligned.pkl`、`att4_aligned.pkl` 由本目录 `att_meta.py` 补齐
  `text_bert`/`text_lengths`/`raw_text`（数值数组逐位未变），修正文本有效长度口径（见下）；
* 提交文件 `submission/pred_explain_att4.csv` 的预测列与 `submission/pred_att4.csv` 逐位一致。

## 5. 口径修正说明（决策 D1=A）

`data_att` 缓存原先未写入 `text_bert`，导致 `build_split` 退化为"尾部全零推断"，
文本 50 步被全部当作有效内容；而附件2 的文本尾部填充行是 BERT 对 `[PAD]` 的隐状态（非零），
唯一可靠的长度来源是注意力掩码。实测代价（附件2 验证集同模型对照）：MAE 0.5866 → 0.8514、
ACC 0.6223 → 0.3310。本目录的 `att_meta.py` 从原始附件 pkl 取回掩码并重写缓存
（原文件备份于 `data_att/_backup_preD1/`），随后 `d1_recalib.py` 重出附件3/4 预测，
对比报告见 `runs/q3/d1_recalib_report.md`。

## 6. 已知局限（论文如实披露）

1. 语音时段由段号按等距比例映射得到，附件2 不提供逐词时间戳，故为近似；
2. 本模型的时间证据较分散（top-5 片段替换对预测的改变量小），与问题2 实测的"帧注意力熵接近上限"一致，
   故以模态级解释为主要依据、片段级为辅；
3. 附件4 仅 20 条、含 1 条视觉整模态缺失，统计结论按分层报告；
4. 注意力源与遮挡源的秩相关很低，本文**不使用**注意力作为解释结论，仅作基线对照。
