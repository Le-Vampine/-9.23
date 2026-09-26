# 工作区 Agent Skills（项目级）

安装位置：`d:\数学建模\数学建模\.github\skills\`
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

0. **先接线**：VS Code 只在 `<工作区根>/.github/skills`、`.claude/skills`、`.agents/skills`
   下发现项目级 skill（仅直接子目录，不递归）。当前工作区根是 `d:\数学建模`，
   而本目录在 `d:\数学建模\数学建模\.github\skills`，**深了一层**，默认扫不到。
   已在 `d:\数学建模\.vscode\settings.json` 登记：

   ```jsonc
   "chat.agentSkillsLocations": { "数学建模/.github/skills": true }
   ```

   该设置只接受相对路径（以每个工作区文件夹为基准）或 `~/` 开头；绝对路径、
   反斜杠、通配符会被 VS Code 拒绝。若今后改为直接打开 `d:\数学建模\数学建模`
   作为工作区，这条设置就多余了，可删除。
1. **重载窗口生效**：命令面板（`Ctrl+Shift+P`）→ `Developer: Reload Window`。
2. 在 Copilot Chat 输入 `/` 查看可用 skill，或直接用自然语言触发
   （例如「帮我把这道题的模型架构画成 TikZ 架构图」）。
3. `skill-inspector` 需要 `skillspector` CLI 才能跑静态扫描；本机未安装该 CLI，
   该 skill 会自动退化为**人工源码审查清单**，仍可使用。

---

## 本机运行环境（2026-09-25 补齐）

skill 的正文是纯 instruction，但其中不少环节会调用脚本，需要本机工具链。已配好：

**Python**
- 解释器：`D:\venv-mosei\Scripts\python.exe`（项目原有 venv，CPython 3.12.14；
  没有另建第二个环境，以免与项目代码的智能提示/依赖打架）
- 本次补装（原有 numpy/pandas/matplotlib/scipy/scikit-learn/openpyxl/tqdm 已具备）：
  python-docx / lxml / defusedxml / pypdf / PyPDF2 / pdfplumber / pymupdf /
  pdf2image / jsonschema / validators / scienceplots / requests
- ⚠️ 系统 `python` 命令仍是 Microsoft Store 占位符（会报 “Python was not found”）。
  新开的终端里由 Python 扩展自动激活上面的环境；旧终端需重启，或直接写绝对路径。

**LaTeX**
- MiKTeX 25.12，位于 `%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`（已写入用户 PATH）
- 已设 `[MPM]AutoInstall=1`，编译时缺宏包会自动从 CTAN 下载，不会卡在交互提示
- 已实测通过：`xelatex` + `ctexart`（中文）+ `tikz` + `pgfplots` 编译出 PDF；
  `cumcmthesis.cls`（v2.9 离线副本）可正常加载编译

**环境自检工具包**
- `d:\\数学建模\\.env_check\\`：`cjk_tikz_test.tex`（中文+TikZ+pgfplots）、
  `cls_test.tex` + `cumcmthesis.cls`（论文模板）。改过环境后重跑一次即可确认链路没坏：
  `xelatex -interaction=nonstopmode -halt-on-error <file>.tex`

**其他**
- Node.js 未安装：本目录下没有任何 `.js/.mjs` 脚本，不影响使用
- `skillspector` CLI 未安装：`skill-inspector` 会自动降级为人工源码审查
- `convert_pdf_to_images.py` 依赖 poppler 的 `pdftoppm`/`pdfinfo`；MiKTeX 已自带这两个
  可执行文件，只要 PATH 里有 MiKTeX bin 目录即可用

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
- 修正断链（本安装改为「平铺」布局后，原本按上游目录层级写的相对路径失效）：
  - `math-modeling-skill/SKILL.md`：`skills/tikz-architecture-diagram/SKILL.md`
    → `../tikz-architecture-diagram/SKILL.md`。
  - `tikz-architecture-diagram/SKILL.md`：`math-modeling-skill/references/{visualization,writing}.md`
    → `../math-modeling-skill/references/{visualization,writing}.md`，并在开头注明本地同级布局。
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
