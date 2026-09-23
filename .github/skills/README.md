# 工作区 Agent Skills（项目级）

安装位置：`e:\数学建模\.github\skills\`
安装日期：2026-09-23
来源：GitHub 上游仓库（经 `ghproxy.net` 镜像下载），**已做本地改造**（见文末）。

---

## 已安装的 5 个 Skill

| 目录 | frontmatter `name` | 用途 | 上游仓库 |
|---|---|---|---|
| `math-modeling/` | `math-modeling` | 数学建模全流程：建模手 / 编程手 / 论文手 三角色 + 阶段内 Subagent 质检；自带 docx / xlsx / pdf / latex / figure / 文献检索六套工具说明 | `XiaoMaColtAI/math-modeling-skill` |
| `math-modeling-skill/` | `math-modeling-skill` | 读题拆解 → 方案选择 → 建模 → 代码实现 → 可视化风格 → LaTeX 论文写作 → 九维审稿 + 版面自动体检（`scripts/check_layout.py`） | `LEEHAHAHAHA/math-modeling-skill` |
| `tikz-architecture-diagram/` | `tikz-architecture-diagram` | 顶刊风格「系统 N 层架构图」TikZ 绘制（层带 + 层标块 + 组件盒 + 层间箭头，xelatex 编译） | `LEEHAHAHAHA/math-modeling-skill`（子 skill） |
| `using-opentikz/` | `using-opentikz` | OpenTikZ 图库：TikZ 图标、可编辑架构/管线/流程模板、期刊级配色与标注速查 | `opentikz/opentikz` |
| `skill-inspector/` | `skill-inspector` | 安装第三方 skill 前的安全审查清单（静态证据 + 语义审查，产出 APPROVE / CAUTION / REJECT） | `NVIDIA/SkillSpector` |

**分工建议**：两个建模 skill 定位有重叠 —— 日常主用 `math-modeling-skill`（论文规范最完整），需要三角色分工与交付时间管控时切 `math-modeling`；画图统一走 `using-opentikz` / `tikz-architecture-diagram`。

---

## 如何使用

1. 本目录属于 VS Code 项目级 skill 路径，**首次安装后需重载窗口**：
   命令面板（`Ctrl+Shift+P`）→ `Developer: Reload Window`。
2. 在 Copilot Chat 输入 `/` 查看可用 skill，或直接用自然语言触发
   （例如「帮我把这道题的模型架构画成 TikZ 架构图」）。
3. `skill-inspector` 需要 `skillspector` CLI 才能跑静态扫描；本机未安装该 CLI，
   该 skill 会自动退化为**人工源码审查清单**，仍可使用。

---

## 上游内容保留 / 本地改造记录

**保留**
- 各 SKILL.md 的 `name` / `description` 原样未改（这是 skill 的发现面，改动会破坏自动触发）。
- 上游 LICENSE / LICENSE-CODE / LICENSE-CONTENT 随目录保留，使用前请自行确认许可条款（部分资源为 CC 系列，商用/再分发前请阅读）。
- 各 skill 正文引用的子资源全部带上：`references/`、`assets/`、`tools/`、`templates/`、`icons/`、`examples/`、`catalog.json` 等。

**移除（对 Copilot 无效或冗余）**
| 路径 | 原因 |
|---|---|
| `math-modeling/dsh-plugin/` | DSH（DeepSeek Harness）插件打包目录，59 个文件 / 749 KB，其中含一份 `name: math-modeling` 的**重复 SKILL.md**，会与顶层 skill 名称冲突；且其安装说明写死上游作者路径，对本机无用 |
| `math-modeling/scripts/sync_dsh_plugin.py` | 只服务于上面的 `dsh-plugin/` 镜像，删除后已无用途（如将来需要，可从上游仓库重新获取） |
| XMC 的 `imgs/`（约 4.6 MB 截图）、`tests/`、`.github/`、`CHANGELOG.md` | 与 skill 运行无关 |
| opentikz 的 `skills-demos/`（含 185 KB 截图）、`assets/`、`.claude-plugin/`、`.github/`、`CLAUDE.md` | 仓库站点/演示素材，非 skill 资源 |
| SkillSpector 的整个 Python 工程（`src/` `tests/` `docs/` `uv.lock` 等） | 体积大且需要单独 `pip install`；只保留 `skills/skill-inspector/SKILL.md` |

**改写**
- `using-opentikz/SKILL.md` §0「Locate the library root (`OTROOT`)」新增第 0 条：
  本包装形式下 `OTROOT` **就是包含该 SKILL.md 的目录**（OpenTikZ 库整体放在同目录）。
  原因：原逻辑依赖 `${CLAUDE_PLUGIN_ROOT}` 或「`SKILL.md` 的上两级目录」，
  在 Copilot 的单 skill 目录布局下两者都不成立。原有 1/2/3 条保留未删。

---

## 目录结构

```
.github/skills/
├── math-modeling/                  # 421 文件中约一半在这里
│   ├── SKILL.md  ·  使用指南.md  ·  VERSION
│   ├── references/  （含 roles/{建模手,编程手,论文手}/ 三个角色子 Skill）
│   ├── assets/      （01~07 算法说明，约 430 KB）
│   └── tools/       （docx / figure / latex / paper_search / pdf / xlsx）
├── math-modeling-skill/
│   ├── SKILL.md  ·  LICENSE  ·  README.md
│   ├── references/  （coding / modeling / parsing / review / visualization / writing）
│   ├── scripts/check_layout.py
│   └── templates/   （flow_snake.tex + cumcmthesis/cumcmthesis.cls）
├── tikz-architecture-diagram/SKILL.md
├── using-opentikz/
│   ├── SKILL.md  ·  catalog.json  ·  meta.schema.json
│   ├── templates/  icons/  reference/  examples/  tools/  docs/
│   └── LICENSE-CODE  ·  LICENSE-CONTENT
└── skill-inspector/SKILL.md
```

> 各 skill 内部（`tools/*/SKILL.md`、`references/roles/*/SKILL.md`）还嵌套着若干
> 子 SKILL.md，它们是**被正文按路径引用的资源**，不是独立 skill 入口，请勿重命名
> 或搬移，否则父级 SKILL.md 的引用会失效。
