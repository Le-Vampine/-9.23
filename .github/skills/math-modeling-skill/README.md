# Math Modeling Skill（数学建模竞赛全流程 Skill）

[![Skill](https://img.shields.io/badge/Agent-Skill-blueviolet)](https://github.com/) [![LaTeX](https://img.shields.io/badge/LaTeX-cumcmthesis-green)](templates/cumcmthesis/) [![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-blue)](scripts/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个面向**数学建模竞赛**（国赛 CUMCM / 美赛 MCM-ICM / 校赛 / 企业赛 / 研究生赛等，不限定特定赛事）的**Agent Skill**：把从「读题拆解」到「论文交付」的完整流程标准化为 7 个阶段，每条规则都来自真实竞赛与真实智能体工程中被评审纠正过的教训，可直接加载到 Claude Code / Codex / WorkBuddy 等 agentic 编码工具中使用。

## ✨ 核心特性

- **读题拆解**：赛题文件多模态解析、附件数据结构探查、按 Q1..Qn 拆题并判定问题类型（回归 / 优化 / 评估 / 仿真 / 预测 / 分类）
- **方案选择**：大道至简偏好序（机理模型优先）、每问 2–3 个差异化候选方案、可解释性验证、诚实披露原则
- **模型搭建**：参数少且每个参数有业务含义；机器学习只作非参数稳健性检验，不作主角
- **代码实现**：防伪数据红线（严禁 `np.random` 模拟数据替代真实数据）、绝对路径读真实文件、Search-Replace 增量纠错（不重写全文）、可复现清单
- **可视化**：nature / science / IEEE 三套学术配色、八大绘图类型选择指南、中文宋体 + 西文 Times、TikZ 蛇形（横向）流程图、TikZ 论文总体架构图
- **论文写作**：cumcmthesis 模板（离线 cls 随仓库自带）、摘要约 80% 版心填充、每问 ≥10 处公式推导 / 定义 / 约束、图表排版硬规范（就近 / 说明 100–150 字 / 单页占比 ≤2/3 / 禁图挨图）、正文 21–30 页
- **审稿修改**：九维评审、三级问题清单、版面逐页视觉审查、图片乱码三道关、修复循环
- **降 AI 味**：论文读起来像资深参赛者手写，反模式清单见 `references/writing.md` §5

## 📁 目录结构

```
math-modeling-skill/
├── SKILL.md                          # Skill 主入口：核心理念 + 7 阶段工作流
├── references/                       # 按任务阶段划分的详细规范
│   ├── parsing.md                    # 读题协议、数据探查、拆题 JSON
│   ├── modeling.md                   # 大道至简偏好序、候选方案、优化模型完整形式
│   ├── coding.md                     # 防伪数据红线、Search-Replace 纠错、可复现清单
│   ├── visualization.md              # 三套配色、八大图型、TikZ 蛇形流程图、图片命名
│   ├── writing.md                    # cumcmthesis 骨架、摘要规则、图表排版硬规范
│   └── review.md                     # 九维评审、三级问题清单、版面视觉审查
├── skills/
│   └── tikz-architecture-diagram/    # 子技能：论文总体架构图（分层架构 TikZ 画法）
│       ├── SKILL.md
│       ├── scripts/check_overlap.py  # 文字防遮盖核验
│       └── templates/                # fig.tex 模板 + build.sh 构建管线
├── templates/
│   ├── cumcmthesis/cumcmthesis.cls   # cumcmthesis v2.9 离线副本（xelatex 编译兜底）
│   └── flow_snake.tex                # 蛇形流程图 TikZ 模板
└── scripts/
    └── check_layout.py               # 交付前版面自动体检（五项违规报行号）
```

## 🚀 快速开始

### 1. 环境依赖

- Python ≥ 3.10（`pandas`、`openpyxl`、`matplotlib`、`pdfplumber`）
- TeX Live / CTeX（`xelatex` + `cumcmthesis`，仓库已带离线 cls 兜底）
- 建议 `kpsewhich cumcmthesis.cls` 检查系统是否已装，未装则把 `templates/cumcmthesis/cumcmthesis.cls` 复制到论文目录

### 2. 安装为 Agent Skill

**Claude Code / WorkBuddy**（个人级，全项目可用）：

```bash
# 克隆到用户级 skills 目录
git clone https://github.com/<your-username>/math-modeling-skill.git \
  ~/.claude/skills/math-modeling-skill   # Claude Code
# 或 ~/.workbuddy/skills/math-modeling-skill  # WorkBuddy
```

**项目级**（仅当前项目团队可用）：克隆到 `<workspace>/.workbuddy/skills/` 下即可。

安装后，只要对话中出现「数学建模 / 数模 / 国赛 / 美赛 / 建模论文 / CUMCM / MCM / ICM」等关键词，agent 即会自动加载本 skill；也可以显式引用 `@skill:math-modeling-skill`。

### 3. 一次竞赛任务的标准推进

按 SKILL.md 中的 7 个阶段推进，每阶段有明确输入 / 产物 / 检查点：

| 阶段 | 内容 | 产物 |
|---|---|---|
| 0 环境自检 | 依赖检查、cls 就位、目录骨架 | `logs/env_check.json` |
| 1 读题拆解 | 题面解析、数据探查、拆题 | `task_package.json` |
| 2 方案建模 | 每问 2–3 候选、用户点选 | `modeling_doc.json` |
| 3 代码求解 | 真实数据、防伪红线、增量纠错 | `Q*/solve.py` + `results/` |
| 4 可视化 | 架构图 1 张 + 各问流程图 + 数据图 | `figures/` |
| 5 论文写作 | 分章节生成 + 编译修复循环 | `main.tex` / `main.pdf` |
| 6 审稿修改 | 九维评审 + 版面体检 + 修复循环 | `review_report.json` |
| 7 交付 | 论文 PDF + 支撑材料 zip | `06_delivery/` |

时间紧只读一个文件的话，读 `references/review.md`。

### 4. 交付前一键版面体检

```bash
python scripts/check_layout.py <论文目录>
```

自动扫描五项违规并报行号：图表就近、说明文字 100–150 字、单页图表占比 ≤2/3、禁图挨图、正文页数 21–30、乱码。

## 🧭 两类结构图，别画错

| 类型 | 数量 | 位置 | 画法 |
|---|---|---|---|
| 论文总体架构图 | 全篇 1 张 | 问题分析节末尾 | `skills/tikz-architecture-diagram`（分层架构，Okabe-Ito 配色） |
| 各问解题流程图 | 每问 1 张 | 该问模型建立开头 | `templates/flow_snake.tex`（蛇形横向，现代柔和 6 色） |

两类图画风不混用。

## 📜 许可证

[MIT](LICENSE)。其中 `templates/cumcmthesis/cumcmthesis.cls` 为 [cumcmthesis](https://github.com/latexstudio/CUMCMThesis)（LaTeX Studio）官方类文件的离线副本，遵循其原有许可证，此处仅作离线兜底，以系统安装版本优先。

## ⚠️ 免责声明

本仓库是竞赛方法论的标准化沉淀，仅供学习与研究使用。使用 AI 辅助参赛时请遵守目标赛事的官方规则（部分赛事要求披露 AI 工具使用情况）。
