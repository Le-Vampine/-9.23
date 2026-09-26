# -*- coding: utf-8 -*-
"""生成「问题2 输出交付包」：把实验数据、图表、代码、流程架构、方案文档整理成
一个自包含目录，附带清单与阅读顺序说明，便于交给写论文的同学。

用法
----
    python code/make_q2_handover.py            # 构建/刷新交付包
    python code/make_q2_handover.py --zip      # 构建后再打一个 zip

不要修改任何原始产物，本脚本只做复制与清单生成。
"""
import argparse
import csv
import io
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "handover_problem2")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- 数据表
# (源路径, 交付包内文件名, 说明, 对应论文位置)
TABLES = [
    ("paper_tables/main_recipe.md", "主模型配方与复现命令.md",
     "主模型的权威超参与完整训练命令，逐项标注了来源（训练脚本 / 配置文件默认值），"
     "并指出重建项在最终配方中被关闭", "§问题2 关键参数表（溯源）"),
    ("paper_tables/ckpt_configs.csv", "checkpoint配置反查.csv",
     "从各 checkpoint 反查出的配置字段，用于核对「论文写的方法」与「实际跑的模型」是否一致",
     "§问题2 关键参数表（核对记录）"),
    ("paper_tables/main_results_summary.csv", "主模型基础性能_汇总.csv",
     "主模型在验证集与测试划分上的 MAE / Pearson / CCC / ACC / Macro-F1，"
     "同时给出分类头主口径与阈值口径", "§问题2 求解结果 表"),
    ("paper_tables/main_results_per_seed.csv", "主模型基础性能_逐种子.csv",
     "同上，按种子拆开，用于说明种子波动量级", "§问题2 求解结果 脚注"),
    ("paper_tables/ablation_paired.csv", "消融实验_同种子配对.csv",
     "【以此为准】同种子、同基准配方下的单开关消融结果，含差值、标准分与判定。"
     "此前的 ablation_clean/results 存在基准错配，已废弃", "§问题2 消融实验 表"),
    ("paper_tables/ablation_paired.md", "消融实验_同种子配对.md",
     "同上，markdown 版，含基准配方的种子数与噪声估计", "§问题2 消融实验 表"),
    ("paper_tables/ablation_clean.csv", "消融实验_旧版_勿引用.csv",
     "【已废弃】旧消融表的干净性能。其「完整模型」用的是 q2v3 配方，"
     "与被消融的配置不是同一个模型，差值不可用，仅留作对照", "不引用"),
    ("paper_tables/ablation_results.csv", "消融实验_旧版逐种子_勿引用.csv",
     "【已废弃】旧消融的逐种子原始指标", "不引用"),
    ("paper_tables/ablation.md", "消融实验_旧版摘要_勿引用.md",
     "【已废弃】旧消融摘要", "不引用"),
    ("paper_tables/missing_law_anova_summary.csv", "缺失规律_三因素方差分析.csv",
     "缺失类型 × 缺失位置 × 缺失率 的三因素方差分析：自由度、F、p、效应量 η² 与偏 η²，"
     "覆盖朴素融合基线与本文模型", "§问题2 缺失规律 表（效应量）"),
    ("paper_tables/missing_law_by_type_rho.csv", "缺失规律_类型×缺失率.csv",
     "按缺失类型与缺失率交叉汇总的退化量", "§问题2 缺失规律 退化曲线图数据"),
    ("paper_tables/missing_law_by_type_position.csv", "缺失规律_类型×缺失位置.csv",
     "按缺失类型与缺失位置交叉汇总的退化量", "§问题2 缺失规律 热力图数据"),
    ("paper_tables/missing_law_by_position_excl_full.csv", "缺失规律_位置边际.csv",
     "排除整模态缺失情形后的位置边际退化，用于给出头部/中部/尾部/随机的排序",
     "§问题2 缺失规律 正文排序"),
    ("paper_tables/missing_law_clean_baseline.csv", "缺失规律_无缺失基线.csv",
     "各模型的无缺失基线指标，退化量的计算基准", "§问题2 缺失规律 基准"),
    ("paper_tables/missing_law_summary.md", "缺失规律_摘要.md",
     "缺失规律一节的全部结论，文字版，可直接作为起草素材", "§问题2 缺失规律 起草"),
    ("paper_tables/missing_duration_map.csv", "缺失时长_率与帧数换算.csv",
     "缺失率与绝对缺失帧数的换算：文本有效长度恒为 50 帧，语音 24.59、视觉 23.82",
     "§问题2 缺失规律 时长口径"),
    ("paper_tables/missing_duration_map.md", "缺失时长_率与帧数换算.md",
     "同上，含有效长度分布与换算说明", "§问题2 缺失规律 时长口径"),
    ("paper_tables/missing_profile.csv", "附件3缺失分布标定.csv",
     "附件3 实测缺失结构（含缺失样本比例、缺失率分位、区间长度、位置分布）",
     "§问题2 训练期缺失注入参数的依据"),
    ("paper_tables/missing_profile.md", "附件3缺失分布标定.md",
     "同上，markdown 版", "§问题2 训练期缺失注入参数的依据"),
    ("paper_tables/error_attribution_groups.csv", "错误归因_五维分层.csv",
     "按强度幅值、文本长度、视觉长度、极性类别、是否含缺失五个维度分层的误差与准确率",
     "§问题2 可视化与错误归因 图数据"),
    ("paper_tables/att3_pred_summary.csv", "附件3预测汇总.csv",
     "附件3 全量预测汇总：极性构成、平均强度、平均置信度，含缺失与无缺失分组对比",
     "§问题2 附件3 结果展示 表"),
    ("paper_tables/att3_pred_summary.md", "附件3预测汇总.md",
     "同上，markdown 版", "§问题2 附件3 结果展示 表"),
    ("paper_tables/att3_missing_distribution.csv", "附件3缺失分布.csv",
     "附件3 各模态含缺失比例与平均缺失率", "§问题2 附件3 缺失分布图数据"),
    ("paper_tables/tune_screen_vs_full.csv", "超参搜索_两阶段对比.csv",
     "快速筛选阶段与完整训练阶段的指标与排名，含两阶段的秩相关",
     "§问题2 超参搜索 图数据"),
    ("paper_tables/tune_seed_noise.csv", "超参搜索_种子噪声估计.csv",
     "各多种子 run 的标准差，作为显著性判定的噪声基准", "§问题2 超参搜索 显著性依据"),
    ("paper_tables/tune_full_report.md", "超参搜索_完整分析.md",
     "超参搜索的完整分析：全量阶段结果、排序可迁移性、显著性判定", "§问题2 超参搜索 起草"),
    ("paper_tables/tables.md", "早期汇总表_部分口径已过期.md",
     "【谨慎使用】项目早期的汇总表，部分口径已修订", "不引用"),
]

# ---------------------------------------------------------------- 图表（论文用图）
# (05_paper/figures 内文件名, 说明, 对应论文位置)
FIGURES = [
    ("Q2_求解流程图.png",
     "问题二求解流程图（蛇形横向，九个环节）", "§问题2 开头"),
    ("Q2_网络结构图.png",
     "MRF-Net 网络结构：三模态 × 四阶段 + 门控融合 + 三个输出头", "§问题2 掩码感知编码 开头"),
    ("Q2_强度预测与真值散点_散点图.png",
     "强度预测值与真值散点，显示两端系统性收缩", "§问题2 求解结果"),
    ("Q2_最终模型混淆矩阵_混淆矩阵.png",
     "三分类混淆矩阵，中性类双向混淆", "§问题2 求解结果"),
    ("Q2_决策阈值敏感性曲线_折线图.png",
     "阈值敏感性曲线，准确率与宏观 F1 同时在 0.28 附近达峰", "§问题2 决策阈值"),
    ("Q2_缺失掩码可视化_时序图.png",
     "填充掩码与缺失掩码的时序可视化，验证二者可分离", "§问题2 求解结果"),
    ("Q2_缺失退化MAE曲线_折线图.png",
     "各缺失类型下 MAE 随缺失率的退化曲线", "§问题2 缺失规律"),
    ("Q2_缺失退化F1曲线_折线图.png",
     "同上，宏观 F1 视角", "§问题2 缺失规律（备选）"),
    ("Q2_模态类型-缺失位置热力图_热力图.png",
     "缺失类型 × 缺失位置的退化热力图（MAE 增量）", "§问题2 缺失规律"),
    ("Q2_模态类型-缺失位置F1热力图_热力图.png",
     "同上，宏观 F1 视角", "§问题2 缺失规律（备选）"),
    ("Q2_四模型缺失退化对比_折线图.png",
     "四个模型（两个朴素融合基线 + 本文单种子/集成）在同一扫描协议下的退化对比",
     "§问题2 缺失规律"),
    ("Q2_超参搜索screen与full对比_哑铃图.png",
     "超参搜索两阶段对比：宏观 F1 排名一致，MAE 排名反转", "§问题2 消融与超参搜索"),
    ("Q2_强度误差分布_直方图.png",
     "强度绝对误差分布直方图，显示长尾", "§问题2 可视化与错误归因"),
    ("Q2_错误归因五维分层_条形图.png",
     "五维分层错误归因（强度、文本长度、视觉长度、极性、缺失状态）",
     "§问题2 可视化与错误归因"),
    ("Q2_风险覆盖率曲线_折线图.png",
     "风险覆盖率曲线（按不确定度剔除样本）", "§问题2 不确定性与拒识"),
    ("Q2_拒识对准确率影响_折线图.png",
     "两类不确定度对分类准确率影响的对比", "§问题2 不确定性与拒识"),
    ("Q2_不确定度与误差关系_散点图.png",
     "总不确定度与绝对误差的散点关系", "§问题2 不确定性与拒识"),
    ("Q2_强度校准前后对比_散点图.png",
     "仿射校准前后的强度对比", "§问题2 决策阈值与校准（对照）"),
    ("Q2_附件3预测结果构成_分组条形图.png",
     "附件3 预测极性构成与强度/置信度对比（含缺失 vs 无缺失）", "§问题2 附件3 结果展示"),
    ("Q2_附件3预测强度与置信度分布_直方图.png",
     "附件3 预测强度与置信度分布", "§问题2 附件3 结果展示"),
    ("Q2_附件3附件4缺失分布_双轴条形图.png",
     "附件3/附件4 实测缺失分布", "§问题2 附件3/4 输入条件"),
    ("Q2_缺失场景综合退化_折线图.png",
     "缺失场景综合退化（备选视角）", "§问题2 缺失规律（备选）"),
    ("Q2_消融实验对比_发散条形图.png",
     "【基于旧口径】消融发散条形图，数据来自已废弃的消融表", "不引用"),
    ("Q2_超参数搜索结果_柱状图.png",
     "【已替换】仅含快速筛选阶段的超参柱状图，已被哑铃图取代", "不引用"),
    ("Q2_训练损失曲线_折线图.png",
     "训练损失与验证指标随轮数的变化", "§问题2 训练方案（可选）"),
]

# ---------------------------------------------------------------- 代码
CODE_FILES = [
    ("code/02_model/config.py", "配置总入口：全部超参与路径"),
    ("code/02_model/data_utils.py", "数据读取、有效长度推断、缺失检测、归一化"),
    ("code/02_model/missing_sim.py", "缺失模拟引擎（类型×位置×缺失率×碎片）与时间裁剪增强"),
    ("code/02_model/model.py", "MRF-Net 骨架：掩码编码、共享-特有分解、门控融合、多任务头"),
    ("code/02_model/losses.py", "全部损失项与评价指标（ACC/F1/MAE/Pearson/CCC、阈值搜索）"),
    ("code/02_model/runtime.py", "共享推理运行时：集成加载与场景推理"),
    ("code/02_model/train.py", "教师-学生训练、评价、缺失扫描的主流程"),
    ("code/02_model/infer_att3.py", "附件3/附件4 推理并写出提交 CSV"),
    ("code/02_model/calibrate_missing.py", "缺失分布标定"),
    ("code/02_model/analyze_sweep.py", "退化扫描的方差分析与曲线拟合"),
    ("code/02_model/sweep_eval.py", "退化扫描执行器"),
    ("code/02_model/sweep_degradation.py", "退化扫描（单模型版）"),
    ("code/02_model/missing_law_tables.py", "缺失规律汇总表生成"),
    ("code/02_model/missing_duration_map.py", "缺失率与绝对缺失帧数换算"),
    ("code/02_model/matched_duration_check.py", "绝对缺失帧数匹配下的三模态退化重比"),
    ("code/02_model/error_analysis.py", "五维分层错误归因"),
    ("code/02_model/perclass_report.py", "逐类指标报告"),
    ("code/02_model/uncertainty_report.py", "四类不确定度、风险覆盖率/AURC、拒识与分组分析"),
    ("code/02_model/ordinal_probe.py", "有序回归头探针（负结果，已记录）"),
    ("code/02_model/neutral_probe.py", "中性边界后处理探针（负结果，已记录）"),
    ("code/02_model/disfluency_impact.py", "非流利标记影响分析（探针）"),
    ("code/02_model/inspect_disfluency.py", "非流利标记盘点"),
    ("code/02_model/label_ceiling.py", "标签噪声上限估计"),
    ("code/02_model/label_stats.py", "标签分布统计"),
    ("code/02_model/postproc_tune.py", "后处理调参（阈值/校准）"),
    ("code/02_model/acc2_tune.py", "二分类（Acc-2）口径调参"),
    ("code/02_model/stack_decision.py", "决策层堆叠对照"),
    ("code/02_model/ens_combo.py", "集成组合枚举与选择"),
    ("code/02_model/make_tables.py", "论文表格生成"),
    ("code/02_model/viz_results.py", "论文基础图生成"),
    ("code/02_model/paper_figs_q2.py", "论文图生成（自然配色）"),
    ("code/02_model/paper_fig_tune_screenfull.py", "超参搜索两阶段对比哑铃图"),
    ("code/02_model/ablation_report.py", "消融报告生成"),
    ("code/02_model/ablation_paired.py", "【修订版】同种子配对消融与显著性"),
    ("code/02_model/att3_predict_report.py", "附件3 预测汇总与展示图"),
    ("code/02_model/tune_hyperparams.py", "两阶段超参搜索"),
    ("code/02_model/tune_full_report.py", "全量阶段结果与显著性核验"),
    ("code/02_model/dump_ckpt_config.py", "从 checkpoint 反查配置"),
    ("code/02_model/missing_probe.py", "缺失敏感性探针"),
    ("code/02_model/missing_profile.py", "缺失分布剖析"),
    ("code/02_model/report_p0.py", "主报告生成（含口径重算）"),
    ("code/02_model/resave_fp32.py", "checkpoint 精度转换"),
    ("code/att_paths.py", "附件3/4 路径自动发现"),
    ("code/inspect_data.py", "附件2 结构核查"),
    ("code/inspect_att34.py", "附件3/4 结构探查"),
    ("code/probe_att34.py", "附件3/4 深度探查"),
    ("code/probe_text.py", "文本缺失表示方式探查"),
    ("code/encode_att_text.py", "用 BERT 为附件3/4 生成文本特征"),
    ("code/verify_text_encoder.py", "验证文本编码与附件2 同源"),
    ("code/download_bert.py", "下载 BERT 权重"),
    ("code/make_dummy_data.py", "合成数据生成（仅供自检，不用于论文结果）"),
    ("code/check_deliverables.py", "交付物与模型体积审计"),
    ("code/make_q2_handover.py", "本脚本：生成交付包"),
    ("code/requirements.txt", "依赖清单"),
    ("code/README.md", "代码使用说明"),
]

DOCS = [
    ("解决方案/00_总体方案与统一接口.md", "总体方案与三问统一接口"),
    ("解决方案/02_问题2_缺失鲁棒情感预测.md", "问题2 的原始方案设计"),
    ("解决方案/05_问题2完成度审计与完整实施流程.md", "问题2 的完成度审计"),
    ("解决方案/06_附件3附件4数据核查与处理方案.md", "附件3/4 的数据核查与文本特征补齐方案"),
    ("解决方案/07_问题2指标口径与合规性说明.md", "【重要】指标口径与合规性说明"),
    ("解决方案/08_问题2缺失规律与错误归因结论.md", "缺失规律、错误归因与不确定性结论"),
    ("解决方案/09_问题2修订记录与口径校正.md", "【重要】本轮修订记录与口径校正"),
    ("解决方案/10_问题2输出清单与交接说明.md", "【入口】输出清单、核心数字与写作骨架"),
]

# 交付包入口文档（复制后改名，放在包根目录）
INDEX_SRC = "解决方案/10_问题2输出清单与交接说明.md"
INDEX_NAME = "00_问题2输出清单.md"

ARCH = [
    ("05_paper/tikz/总体架构图_五层建模框架.tex", "总体架构图 TikZ 源（五层建模框架，全文仅此一张）",
     "架构图源文件.tex"),
    ("05_paper/tikz/总体架构图_五层建模框架.pdf", "总体架构图 矢量版", "架构图.pdf"),
    ("05_paper/tikz/架构图预览.png", "总体架构图 位图预览", "架构图_预览.png"),
    ("05_paper/tikz/Q2_网络结构图_body.tex", "网络结构图 TikZ 主体", "网络结构图_主体.tex"),
    ("05_paper/tikz/_build_Q2_net.tex", "网络结构图 独立编译入口", "网络结构图_编译入口.tex"),
    ("05_paper/tikz/_build_Q2_net.pdf", "网络结构图 矢量版", "网络结构图.pdf"),
    ("05_paper/figures/Q2_网络结构图.png", "网络结构图 位图（论文实际引入的版本）", "网络结构图.png"),
    ("05_paper/tikz/Q2_求解流程图_body.tex", "求解流程图 TikZ 主体（蛇形）", "求解流程图_主体.tex"),
    ("05_paper/tikz/_build_Q2_flow.tex", "求解流程图 独立编译入口", "求解流程图_编译入口.tex"),
    ("05_paper/tikz/_build_Q2_flow.pdf", "求解流程图 矢量版", "求解流程图.pdf"),
    ("05_paper/figures/Q2_求解流程图.png", "求解流程图 位图（论文实际引入的版本）", "求解流程图.png"),
]

SUB = [("submission/pred_att3.csv", "附件3 预测结果（30 条，正式提交版）", "pred_att3.csv"),
       ("submission/pred_att4.csv", "附件4 预测结果（20 条，正式提交版）", "pred_att4.csv")]
# 附件4 的可解释字段尚缺，特意在清单里标出来

# ------------------------------------------------- 文献对比（供写论文时补充）
LIT = [
    ("解决方案/11_问题2文献对比与引用挂接.md", "文献对比与引用挂接.md",
     "【必读】三个口径陷阱 + 本文可对比指标 + 对比表模板 + 8 条孤儿文献的引用挂接位置 "
     "+ 赛题提供的 10 条参考文献", "§相关问题二 求解结果 / 相关工作 / 模型评价"),
    ("paper_tables/lit_our_comparable.csv", "本文可对比指标.csv",
     "本文在文献口径（Acc-2）与题目口径（三分类）下的全部指标，含区分与来源文件",
     "§问题二 对比表数据来源"),
    ("paper_tables/lit_comparison_template.csv", "方法对比表模板.csv",
     "文献方法对比表的表头与待填行，数字列刻意留空以免编造（填表规则见文献对比与引用挂接.md）",
     "§问题二 方法对比表"),
]


def cp(src, dst_dir, name):
    s = os.path.join(ROOT, src.replace("/", os.sep))
    if not os.path.isfile(s):
        return None, 0
    os.makedirs(dst_dir, exist_ok=True)
    d = os.path.join(dst_dir, name)
    shutil.copy2(s, d)
    return d, os.path.getsize(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    dirs = {k: os.path.join(OUT, v) for k, v in [
        ("table", "01_实验数据表"),
        ("fig", "02_图表"),
        ("arch", "03_流程与架构"),
        ("code", "04_核心代码"),
        ("doc", "05_方案文档"),
        ("sub", "06_提交文件"),
        ("lit", "07_文献对比"),
    ]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    man = []

    for src, name, desc, where in TABLES:
        p, sz = cp(src, dirs["table"], name)
        if p is None:
            print("  [MISS] %s" % src)
            continue
        man.append(("实验数据表", "01_实验数据表/" + name, src, desc, where, sz))

    for name, desc, where in FIGURES:
        src = "05_paper/figures/" + name
        p, sz = cp(src, dirs["fig"], name)
        if p is None:
            print("  [MISS] %s" % src)
            continue
        man.append(("图表", "02_图表/" + name, src, desc, where, sz))

    for src, desc, name in ARCH:
        p, sz = cp(src, dirs["arch"], name)
        if p is None:
            print("  [MISS] %s" % src)
            continue
        man.append(("流程与架构", "03_流程与架构/" + name, src, desc, "§问题2 结构图/流程图", sz))

    for src, desc in CODE_FILES:
        name = src.split("/")[-1]
        p, sz = cp(src, dirs["code"], name)
        if p is None:
            print("  [MISS] %s" % src)
            continue
        man.append(("核心代码", "04_核心代码/" + name, src, desc, "支撑材料/附录代码", sz))

    for src, desc in DOCS:
        name = src.split("/")[-1]
        p, sz = cp(src, dirs["doc"], name)
        if p is None:
            print("  [MISS] %s" % src)
            continue
        man.append(("方案文档", "05_方案文档/" + name, src, desc, "方案与修订依据", sz))

    for src, desc, name in SUB:
        p, sz = cp(src, dirs["sub"], name)
        if p is None:
            continue
        man.append(("提交文件", "06_提交文件/" + name, src, desc, "§问题2 附件三/四 推理结果", sz))
    for src, name, desc, where in LIT:
        p, sz = cp(src, dirs["lit"], name)
        if p is None:
            print("  [MISS] %s" % src)
            continue
        man.append(("文献对比", "07_文献对比/" + name, src, desc, where, sz))
    # ------------------------------------------------------------ 入口文档
    p, sz = cp(INDEX_SRC, OUT, INDEX_NAME)
    if p is None:
        print("  [MISS] %s" % INDEX_SRC)
    else:
        man.insert(0, ("入口文档", INDEX_NAME, INDEX_SRC,
                       "输出清单、核心数字速查、阅读顺序、论文写作骨架与未完成事项",
                       "建议先读这份", sz))

    # ------------------------------------------------------------ manifest
    mp = os.path.join(OUT, "manifest.csv")
    with open(mp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["分组", "交付包内路径", "原始路径", "说明", "对应论文位置", "字节"])
        for r in man:
            w.writerow(r)

    total = sum(r[5] for r in man)
    print("\n[INFO] 共 %d 个文件，合计 %.2f MB" % (len(man), total / 1048576.0))

    if a.zip:
        zp = os.path.join(ROOT, "handover_problem2.zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for r, _, fs in os.walk(OUT):
                for fn in fs:
                    fp = os.path.join(r, fn)
                    z.write(fp, os.path.relpath(fp, ROOT))
        print("[OK] %s  (%.2f MB)" % (os.path.basename(zp),
                                      os.path.getsize(zp) / 1048576.0))
    print("[OK] %s" % OUT)


if __name__ == "__main__":
    main()
