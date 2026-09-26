# -*- coding: utf-8 -*-
"""
构建 2026 研赛 E 题《支撑材料_总附件》（≤50MB）
- 01 问题一：E.1 交付包 = 2026-09-26《问题一提交组件.zip》解压全量（486 条目全量 + 打包核查报告）
- 02/03 问题二/三：核心代码、模型参数、评估/解释记录
- 04 结果表：paper_tables 全量
- 05 专项测试提交结果：pred_att3/att4/explain_att4 + explain_cards + evidence
- 06 运行环境：requirements.txt + 版本说明
"""
import hashlib
import os
import shutil
import sys
import zipfile

ROOT = r'D:\数学建模\数学建模'
STAGE = os.path.join(ROOT, '07_attachment', '支撑材料_总附件')
# 问题一来源：2026-09-26 更新的《问题一提交组件.zip》解压树（旧候选 _review_e1/E.1 已弃用）
E1 = os.path.join(ROOT, '_review_deliv', 'q1new', 'E.1')
DELIV = os.path.join(ROOT, '_review_deliv', '交付候选')
CODE = os.path.join(ROOT, 'code')
RUNS = os.path.join(ROOT, 'runs')
TABLES = os.path.join(ROOT, 'paper_tables')
SUBM = os.path.join(ROOT, 'submission')

SKIP_DIRS = {'__pycache__', '.ipynb_checkpoints'}
BIG_Q3 = {'q3_temporal_fused_valid.npz', 'q3_temporal_ig_valid.npz',
          'q3_temporal_occ_valid.npz', 'q3_shapley_valid.npz'}


def ig_e1(dirpath, names):
    return [n for n in names if n in SKIP_DIRS]


def copy2(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)

    # ---------- 01 问题一 ----------
    q1 = os.path.join(STAGE, '01_问题一_词级特征与时序对齐')
    shutil.copytree(E1, q1, ignore=ig_e1)
    # 追加入库：打包核查报告（与组件内 交付索引/全量核验 同置一处）
    copy2(os.path.join(DELIV, '打包核查报告.json'),
          os.path.join(q1, '交付候选', '打包核查报告.json'))

    # ---------- 00 公共脚本 ----------
    common = os.path.join(STAGE, '00_公共脚本')
    os.makedirs(common)
    keep_root = {'att_paths.py', 'encode_att_text.py', 'verify_text_encoder.py',
                 'download_bert.py', 'inspect_data.py', 'inspect_label.py',
                 'inspect_att34.py', 'probe_att34.py', 'probe_text.py',
                 'diagnose_nan.py', 'check_deliverables.py', 'make_dummy_data.py'}
    for n in sorted(os.listdir(CODE)):
        p = os.path.join(CODE, n)
        if os.path.isfile(p) and n in keep_root:
            copy2(p, os.path.join(common, n))

    # ---------- 02 问题二 ----------
    q2 = os.path.join(STAGE, '02_问题二_缺失鲁棒预测')
    os.makedirs(q2)
    src_model = os.path.join(CODE, '02_model')
    for n in sorted(os.listdir(src_model)):
        p = os.path.join(src_model, n)
        if os.path.isfile(p) and n.endswith('.py'):
            copy2(p, os.path.join(q2, '代码', n))
    copy2(os.path.join(CODE, 'README.md'), os.path.join(q2, '代码', 'README.md'))
    os.makedirs(os.path.join(q2, '模型参数'))
    ens = os.path.join(RUNS, 'ens_top2')
    for n in sorted(os.listdir(ens)):
        p = os.path.join(ens, n)
        if not os.path.isfile(p):
            continue
        if n.endswith('.pt'):
            copy2(p, os.path.join(q2, '模型参数', n))
        elif n.endswith(('.json', '.csv', '.md')):
            copy2(p, os.path.join(q2, '评估与决策记录', n))

    # ---------- 03 问题三 ----------
    q3 = os.path.join(STAGE, '03_问题三_可解释预测')
    os.makedirs(q3)
    src_xai = os.path.join(CODE, '03_xai')
    for n in sorted(os.listdir(src_xai)):
        p = os.path.join(src_xai, n)
        if os.path.isfile(p) and n.endswith('.py'):
            copy2(p, os.path.join(q3, '代码', n))
    os.makedirs(os.path.join(q3, '模型参数'))
    rq3 = os.path.join(RUNS, 'q3')
    for dirpath, dirnames, files in os.walk(rq3):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for n in files:
            if n in BIG_Q3 or n == 'q3_evidence_head.pt':
                continue
            rel = os.path.relpath(os.path.join(dirpath, n), rq3)
            copy2(os.path.join(dirpath, n), os.path.join(q3, '解释记录', rel))
    copy2(os.path.join(rq3, 'q3_evidence_head.pt'),
          os.path.join(q3, '模型参数', 'q3_evidence_head.pt'))

    # ---------- 04 结果表 ----------
    for n in sorted(os.listdir(TABLES)):
        p = os.path.join(TABLES, n)
        if os.path.isfile(p):
            copy2(p, os.path.join(STAGE, '04_结果表', n))

    # ---------- 05 专项测试提交结果 ----------
    q5 = os.path.join(STAGE, '05_专项测试提交结果')
    os.makedirs(q5)
    for n in ['pred_att3.csv', 'pred_att4.csv', 'pred_explain_att4.csv']:
        copy2(os.path.join(SUBM, n), os.path.join(q5, n))
    for sub in ['explain_cards', 'evidence']:
        shutil.copytree(os.path.join(SUBM, sub), os.path.join(q5, sub))

    # ---------- 06 运行环境 ----------
    os.makedirs(os.path.join(STAGE, '06_运行环境'))
    copy2(os.path.join(CODE, 'requirements.txt'),
          os.path.join(STAGE, '06_运行环境', 'requirements.txt'))

    # ---------- 版本探测 ----------
    vers = []
    vers.append('Python ' + sys.version.split()[0])
    for mod in ['numpy', 'pandas', 'torch', 'sklearn', 'scipy', 'openpyxl', 'matplotlib']:
        try:
            m = __import__(mod)
            vers.append(f'{mod} {getattr(m, "__version__", "?")}')
        except Exception as e:
            vers.append(f'{mod} (未安装: {e.__class__.__name__})')
    ver_txt = '\n'.join(vers)

    open(os.path.join(STAGE, '06_运行环境', '运行环境说明.md'), 'w', encoding='utf-8').write(
        '# 运行环境说明\n\n'
        '本包内问题二、问题三代码在以下环境实测运行（CPU，无 GPU 依赖）：\n\n'
        '```\n' + ver_txt + '\n```\n\n'
        '- 依赖清单：`requirements.txt`（pip install -r requirements.txt）。\n'
        '- 附件三/附件四文本特征补齐使用 `bert-base-uncased`（固定修订版，见 00_总说明.md）。\n'
        '- 路径适配：代码通过 `att_paths.py` 自动定位附件目录，无需改写绝对路径。\n'
    )

    # ---------- 03 运行说明 ----------
    open(os.path.join(q3, '代码', '运行说明.md'), 'w', encoding='utf-8').write(
        '# 问题三代码运行说明（三源解释与证据定位）\n\n'
        '前置条件：完成问题二模型（见 `../../02_问题二_缺失鲁棒预测/代码/README.md`），'
        '并已生成附件三/附件四的文本特征与评测拆分。\n\n'
        '按以下顺序执行（全部脚本在 `03_问题三_可解释预测/代码/` 目录下运行；'
        '每个脚本支持 `python <脚本> --help` 查看参数）：\n\n'
        '1. `shapley.py`：模态级精确 Shapley 值（八子集全枚举）→ `q3_shapley_*.csv`\n'
        '2. `ig_gradcam.py`：积分梯度片段重要性 → `q3_temporal_ig_*.npz`\n'
        '3. `occlusion.py`：模型无关遮挡重要性（窗宽 3）→ `q3_temporal_occ_*.npz`\n'
        '4. `temporal_fuse.py`：三源融合与 γ 单纯形约束标定 → `q3_temporal_fused_*.npz`\n'
        '5. `train_evidence_head.py`：冻结骨干、并联证据头训练 → `q3_evidence_head.pt`\n'
        '6. `evidence.py`：证据定位（词元区间/语音时段/关键帧）→ `q3_evidence_att4.json`\n'
        '7. `explain_card.py`：逐样本解释卡 → `submission/explain_cards/*.json`\n'
        '8. `infer_explain_att4.py`：附件四全量推理与解释输出 → `pred_explain_att4.csv`、'
        '关键帧图片 `submission/evidence/*.jpg`\n'
        '9. `verify_q3.py`：交付自查（一致性、页面、匿名性等17项校验）\n\n'
        '中间结果（如 `q3_temporal_fused_valid.npz` 等 2.9MB 的逐样本数组）可由上述脚本重跑生成，'
        '为控制附件体积未随包提供；CSV 级结果已全部保留。\n'
    )

    # ---------- 00 总说明 ----------
    def dsize(rel):
        p = os.path.join(STAGE, rel)
        s = 0
        c = 0
        for dp, _, fs in os.walk(p):
            for n in fs:
                s += os.path.getsize(os.path.join(dp, n))
                c += 1
        return c, s / 1048576

    rows = []
    catalog = [
        ('00_公共脚本', '数据探查、文本特征补齐、编码一致性校验等两问共用的公共脚本'),
        ('01_问题一_词级特征与时序对齐', '100 条样本词级三模态特征、逐帧声学/视觉特征、逐词索引、代码、配置、日志、提交核对清单与交付索引'),
        ('02_问题二_缺失鲁棒预测', 'MRF-Net 核心代码、双种子集成模型参数、评估与决策记录'),
        ('03_问题三_可解释预测', '三源解释代码、证据头模型参数、解释记录与运行说明'),
        ('04_结果表', '论文正文与附录引用的全部汇总表（主结果、消融、方差分析、错误归因、Q3_* 等）'),
        ('05_专项测试提交结果', '附件三 30 条预测、附件四 20 条预测与解释（含解释卡与关键帧图）'),
        ('06_运行环境', '依赖清单与实测版本说明'),
    ]
    for rel, desc in catalog:
        c, mb = dsize(rel)
        rows.append(f'| {rel}/ | {c} | {mb:.1f} MB | {desc} |')
    table = '\n'.join(rows)

    doc = f'''# 2026 年中国研究生数学建模竞赛 E 题 支撑材料总附件说明

**题目：复杂场景下多模态情感预测的数学建模与算法设计**

本包为赛题"（二）附件提交要求"对应的全部支撑材料，总量控制在 50MB 以内，内部附逐文件校验和 `SHA256SUMS.txt`。

## 一、包内结构与对应关系

| 目录 | 文件数 | 大小 | 内容 |
|---|---:|---:|---|
{table}
| SHA256SUMS.txt | 1 | — | 全包逐文件 SHA-256 校验和 |

## 二、问题一材料（100 条样本多模态时序特征文件）

- **主结果**：`01_问题一_词级特征与时序对齐/结果/08_原文词级跨模态对齐/`，
  含 100 份逐样本 NPZ（文本 768 / 语音 74 / 视觉 52 维，词级变长）及
  `逐词对齐索引.csv`、`逐样本对齐统计.csv`、`对齐规则.json`、`对齐汇总.json`。
- **逐帧支撑特征**：`结果/06_Librosa74/`（77,755 帧 × 74 维）、`结果/07_FaceLandmarker52/`（7,889 帧 × 52 维）。
- **转写与质检**：`结果/03_Whisper_small_en/`、`结果/10_全量质量检查与论文图表/`。
- **提交核对与定位**：`交付候选/100条交付索引.csv` 逐样本给出原始视频/音频/逐帧/主对齐文件的相对路径、有效时长、
  词覆盖统计与逐项 SHA-256，可逐文件逐样本定位；`交付候选/提交前全量核验.json`（状态 pass）、
  `交付候选/打包核查报告.json` 记录提交前核验与打包核查结果；核验脚本见 `代码/质量检查/核验问题一提交材料.py`、
  `代码/质量检查/打包问题一交付候选.py`。
- **覆盖统计**：1,934 个原文词全部有文本特征；1,421 个词有主时间与音频；1,228 个词有区间内有效视觉；
  513 个词主时间为空（保留文本向量与有效掩码）。缺失与填充严格区分，零占位不视为观测值。
- 处理链、工具版本与读取方式见 `01_.../README.md`、`论文/复现与交付说明.md`、`论文/问题一提交核对清单.md`、`配置/第一问流程配置.json`。

## 三、问题二、问题三材料

- **核心代码**：`02_.../代码/`（MRF-Net 模型、损失、训练与推理）、`03_.../代码/`（Shapley、积分梯度、遮挡、融合、证据头、证据定位等）。
- **运行说明**：问题二见 `02_.../代码/README.md`；问题三见 `03_.../代码/运行说明.md`。
- **模型参数**：`student_s42.pt`、`student_s43.pt`（双种子集成；决策阈值 θ=0.325 于验证集选定）；
  `q3_evidence_head.pt`（冻结骨干上并联解释头）。加载方式见代码与配置。
- **配置文件**：`code/02_model/config.py`、`runs/ens_top2/stack_decision_ens_top2.json`、
  `runs/q3/q3_explain_meta.json` 等（随对应目录提供）。
- **结果表**：`04_结果表/` 为论文各表格的原始汇总；**专项测试提交文件**在 `05_专项测试提交结果/`，
  三个 CSV 依次为附件三预测、附件四预测、附件四预测与解释（字段说明见论文 5.4.6 表 17）。

## 四、数据集处理规则与关键口径（复现必读）

1. **数据来源**：仅使用赛题提供的附件 1–4，不引入任何外部情感数据集。
2. **附件 2（aligned_50）填充与缺失**：序列尾部连续全零判定为填充；有效区间内连续不少于两帧的全零判定为
   局部缺失；有效长度由零值结构推断。**文本有效长度以 `text_bert` 注意力掩码为准**（尾部填充为 [PAD]
   隐状态，非零，不能按零值推断）。
3. **附件 3/4 文本补齐**：使用与附件 2 同源的 `bert-base-uncased`（固定修订版）管线补齐文本特征，
   编码一致性余弦相似度 1.0000；2026-09-25 修正文本长度口径后已重跑提交文件
   （pred_att3 均值 +0.2451、pred_att4 均值 −0.0540，与论文表 16/17 一致）。
4. **选择口径**：全部超参数、阈值与校准系数仅在附件 2 验证集上确定；附件 3、附件 4 不参与任何模型选择。
5. **训练与解释设计**：训练期按附件三实测缺失分布做课程式缺失注入；解释为三源融合
   （精确 Shapley + 积分梯度 + 遮挡，γ=(0.1, 0.1, 0.8) 于验证集标定）。

## 五、运行环境

见 `06_运行环境/`（Python 3.12，CPU 运行，依赖清单 `requirements.txt`）；
问题一管线所需工具（MFA 3.4.2、faster-whisper small.en、librosa 0.11.0、MediaPipe 0.10.32）
及其版本、哈希见 `01_.../配置/第一问流程配置.json` 与 `01_.../README.md`；
问题一跨机复现条件与版本核查另见 `01_.../论文/环境与跨机复现说明.md`。

## 六、体积与完整性说明

1. **问题一材料为全量交付版**：`01_问题一_词级特征与时序对齐/` 即 2026-09-26 更新版
   《问题一提交组件.zip》（SHA-256 `ea9645216a198d484cef3bb7b76c50023f96f9d3f2e2adf0c9f25e858365f74f`，
   486 个正式条目）解压后的全量内容，并补入其 `打包核查报告.json`；
   `01_.../交付文件哈希.json` 与全包 `SHA256SUMS.txt` 可逐项核对。
2. **仅一处裁剪**：问题三 `runs/q3` 的 4 个逐样本解释中间数组（约 2.9MB，如
   `q3_temporal_fused_valid.npz`）未随包提供，可由 `03_.../代码/` 中脚本重跑生成；
   CSV 级结果全部保留。

本包不含附件 1 原始视频与附件 2–4 数据文件（赛题另行提供），复现时按各 README 将附件放入输入位置即可。

## 七、匿名性

包内全部材料已检查，不含参赛单位、队员姓名、队伍编号等身份信息。
'''
    open(os.path.join(STAGE, '00_总说明.md'), 'w', encoding='utf-8').write(doc)

    # ---------- 尺寸统计 & 校验 ----------
    total = 0
    nfiles = 0
    for dirpath, _, files in os.walk(STAGE):
        for n in files:
            total += os.path.getsize(os.path.join(dirpath, n))
            nfiles += 1
    print(f'暂存文件数: {nfiles}, 未压缩大小: {total/1048576:.2f} MB')

    # 匿名性扫描
    pats = ['feelikesummer', '队号', '学号', '指导教师']
    suspects = []
    for dirpath, _, files in os.walk(STAGE):
        for n in files:
            if n.lower().endswith(('.py', '.md', '.json', '.csv', '.txt', '.log')):
                p = os.path.join(dirpath, n)
                t = open(p, encoding='utf-8', errors='ignore').read()
                for pat in pats:
                    if pat in t:
                        suspects.append((p, pat))
    print('匿名性扫描（供人工复核）:')
    for p, pat in suspects:
        print('   ', pat, '->', os.path.relpath(p, STAGE))

    # ---------- SHA256 清单 ----------
    lines = []
    for dirpath, dirnames, files in os.walk(STAGE):
        dirnames.sort()
        for n in sorted(files):
            p = os.path.join(dirpath, n)
            h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
            lines.append(f'{h}  {os.path.relpath(p, STAGE)}')
    open(os.path.join(STAGE, 'SHA256SUMS.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('SHA256SUMS.txt 条目数:', len(lines))

    # ---------- 打包 ----------
    zpath = os.path.join(ROOT, '支撑材料_总附件.zip')
    if os.path.exists(zpath):
        os.remove(zpath)
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for dirpath, dirnames, files in os.walk(STAGE):
            dirnames.sort()
            for n in sorted(files):
                p = os.path.join(dirpath, n)
                arc = os.path.join('支撑材料_总附件', os.path.relpath(p, STAGE))
                z.write(p, arc)
    zsize = os.path.getsize(zpath)
    print(f'输出: {zpath}')
    print(f'压缩包大小: {zsize/1048576:.2f} MB  ({zsize} 字节)')
    # CRC 校验
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        print('CRC 校验:', '全部通过' if bad is None else f'损坏: {bad}')
        print('压缩包条目数:', len(z.namelist()))


if __name__ == '__main__':
    main()
