"""Build the polished Problem 1 section and its integration-ready TeX fragment.

The DOCX is for review; the TeX fragment is for integration into the main paper.
All statistics and the 100-row appendix are read from audited E.1 results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


BASE = Path(__file__).resolve().parents[2]
PAPER = BASE / "论文"
ASSETS = PAPER / "图表精修" / "论文插图"
EVIDENCE = BASE / "结果" / "10_全量质量检查与论文图表"
STATS = json.loads((EVIDENCE / "全量核验与论文数据.json").read_text(encoding="utf-8"))
ROWS = json.loads((EVIDENCE / "全量样本统计.json").read_text(encoding="utf-8"))
# This is the maintained Problem 1 manuscript; earlier drafts are not touched.
DOCX = PAPER / "问题一_五节结构与图表精修稿.docx"
TEX = PAPER / "问题一_五节结构与图表精修稿.tex"


def install_assets() -> None:
    required = (
        "fig01_flow.png", "fig02_evidence.png", "fig03_coverage.png", "fig04_distribution.png",
        "fig05_case_A.png", "fig06_case_B.png", "fig07_case_visual_missing.png", "fig08_case_asr_empty.png",
        "eq01.png", "eq02.png", "eq03.png", "eq04.png",
    )
    for name in required:
        if not (ASSETS / name).is_file():
            raise FileNotFoundError(ASSETS / name)


install_assets()


def text_run(run, name="宋体", size=None, bold=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    return run


def add_field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    for tag, value in (("begin", None), (None, instruction), ("separate", None), ("end", None)):
        if tag:
            item = OxmlElement("w:fldChar")
            item.set(qn("w:fldCharType"), tag)
        else:
            item = OxmlElement("w:instrText")
            item.set(qn("xml:space"), "preserve")
            item.text = value
        run._r.append(item)


doc = Document()
sec = doc.sections[0]
sec.page_width = Cm(21)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.3)
sec.bottom_margin = Cm(2.25)
sec.left_margin = Cm(2.55)
sec.right_margin = Cm(2.55)
sec.footer_distance = Cm(1.05)
normal = doc.styles["Normal"]
normal.font.name = "宋体"
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.font.size = Pt(11.5)
normal.font.color.rgb = RGBColor(0, 0, 0)
normal.paragraph_format.line_spacing = 1.25
normal.paragraph_format.space_after = Pt(4)
normal.paragraph_format.first_line_indent = Pt(23)
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

for style_name, size, before, after in (
    ("Title", 14, 0, 9),
    ("Heading 1", 13, 14, 7),
    ("Heading 2", 12, 13, 6),
):
    style = doc.styles[style_name]
    style.font.name = "黑体"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    style.font.size = Pt(size)
    style.font.bold = True
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.paragraph_format.first_line_indent = Pt(0)
    style.paragraph_format.space_before = Pt(before)
    style.paragraph_format.space_after = Pt(after)
    style.paragraph_format.keep_with_next = True
doc.styles["Title"].paragraph_format.first_line_indent = Pt(0)
doc.styles["Caption"].paragraph_format.first_line_indent = Pt(0)
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_field(footer, " PAGE ")

tex: list[str] = [
    "% 问题一五节结构正文片段。并入 main.tex 后由 LaTeX 自动顺延全篇图表和公式编号。",
    "% 前置宏包：graphicx, booktabs, longtable, pdflscape, array。",
    "% 将本文件与“图表精修/论文插图”文件夹置于同一目录。",
    "",
]


def tex_escape(value: str) -> str:
    out = []
    for c in value:
        out.append({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}.get(c, c))
    return "".join(out)


def paragraph(value: str, keep_together=False) -> None:
    p = doc.add_paragraph(value)
    p.paragraph_format.keep_together = keep_together
    tex.extend([tex_escape(value), ""])


def paragraph_lead(lead: str, body: str) -> None:
    p = doc.add_paragraph()
    text_run(p.add_run(lead), "宋体", 11.5, True)
    p.add_run(body)
    tex.extend([r"\textbf{" + tex_escape(lead) + "}" + tex_escape(body), ""])


def heading(value: str, level=2) -> None:
    style = "Title" if level == 0 else "Heading 1" if level == 1 else "Heading 2"
    p = doc.add_paragraph(style=style)
    if level == 0:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    text_run(p.add_run(value), "黑体", 14 if level == 0 else 13 if level == 1 else 12, True)
    if level == 0:
        # Word's built-in Title style may contain a blue bottom rule.
        bdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "nil")
        bdr.append(bottom)
        p._p.get_or_add_pPr().append(bdr)
    command = r"\section*{" if level == 0 else r"\subsection*{" if level == 1 else r"\subsubsection*{"
    tex.extend([command + tex_escape(value) + "}", ""])


def caption(value: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = False
    text_run(p.add_run(value), "黑体", 10.5)


def fig(filename: str, description: str, width_cm=15.7) -> None:
    image = ASSETS / filename
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(image), width=Cm(width_cm))
    caption(description)
    tex.extend([
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\includegraphics[width=0.98\linewidth]{图表精修/论文插图/" + filename + "}",
        r"\caption{" + tex_escape(description.split(" ", 2)[-1]) + "}",
        r"\end{figure}",
        "",
    ])


def cell_rule(cell, edge: str, size: int) -> None:
    """Draw only horizontal rules, following the paper's three-line tables."""
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    rule = OxmlElement("w:" + edge)
    rule.set(qn("w:val"), "single")
    rule.set(qn("w:sz"), str(size))
    rule.set(qn("w:color"), "222222")
    borders.append(rule)


def cell_margin(cell, twips=55) -> None:
    mar = OxmlElement("w:tcMar")
    for side in ("top", "start", "bottom", "end"):
        el = OxmlElement("w:" + side)
        el.set(qn("w:w"), str(twips))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    cell._tc.get_or_add_tcPr().append(mar)


def add_table(title: str, headers: list[str], rows: list[list[str]], widths: list[float], font_size=9.1, appendix=False, page_break_before=False) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.page_break_before = page_break_before
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    text_run(p.add_run(title), "黑体", 10.5)
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for j, header in enumerate(headers):
        table.columns[j].width = Cm(widths[j])
        c = table.rows[0].cells[j]
        c.width = Cm(widths[j])
        c.text = header
        cell_rule(c, "top", 12)
        cell_rule(c, "bottom", 7)
        cell_margin(c, 28 if appendix else 55)
        for cp in c.paragraphs:
            cp.paragraph_format.first_line_indent = Pt(0)
            cp.paragraph_format.space_after = Pt(0)
            cp.paragraph_format.keep_with_next = True
            for run in cp.runs:
                text_run(run, "黑体", font_size, True)
    trPr = table.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    trPr.append(repeat)
    for i, row in enumerate(rows):
        cells = table.add_row().cells
        trPr = table.rows[-1]._tr.get_or_add_trPr()
        no_split = OxmlElement("w:cantSplit")
        trPr.append(no_split)
        for j, value in enumerate(row):
            cells[j].width = Cm(widths[j])
            cells[j].text = str(value)
            cells[j].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            cell_margin(cells[j], 25 if appendix else 50)
            if i == len(rows) - 1:
                cell_rule(cells[j], "bottom", 12)
            for cp in cells[j].paragraphs:
                cp.paragraph_format.first_line_indent = Pt(0)
                cp.paragraph_format.space_after = Pt(0)
                cp.paragraph_format.line_spacing = 1.02 if appendix else 1.1
                cp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 1 else WD_ALIGN_PARAGRAPH.CENTER
                for run in cp.runs:
                    text_run(run, "宋体", font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    if appendix:
        return
    cols = "".join("p{" + str(round(0.96 * w / sum(widths), 3)) + r"\linewidth}" for w in widths)
    tex.extend([
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{" + tex_escape(title.split(" ", 2)[-1]) + "}",
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
        " & ".join(tex_escape(x) for x in headers) + r" \\",
        r"\midrule",
    ])
    for row in rows:
        tex.append(" & ".join(tex_escape(str(x)) for x in row) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def formula(latex: str, number: int, width_cm=12.8) -> None:
    path = ASSETS / f"eq{number:02d}.png"
    if not path.exists():
        raise FileNotFoundError(f"Run 生成问题一公式.py first: {path}")
    table = doc.add_table(rows=1, cols=2)
    table.autofit = False
    table.columns[0].width = Cm(13.8)
    table.columns[1].width = Cm(1.7)
    table.cell(0, 0).width = Cm(13.8)
    table.cell(0, 1).width = Cm(1.7)
    no_split = OxmlElement("w:cantSplit")
    table.rows[0]._tr.get_or_add_trPr().append(no_split)
    for c in table.rows[0].cells:
        cell_margin(c, 0)
    p = table.cell(0, 0).paragraphs[0]
    p.paragraph_format.first_line_indent = Pt(0)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(min(width_cm, 13.3)))
    q = table.cell(0, 1).paragraphs[0]
    q.paragraph_format.first_line_indent = Pt(0)
    q.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    text_run(q.add_run(f"（{number}）"), "宋体", 10.5)
    tex.extend([r"\begin{equation}", latex, r"\end{equation}", ""])


heading("问题一模型的建立与求解", level=0)
paragraph("附件1包含100条英文原始视频及其文本、情感标签。问题一要求从原始声画信息中提取文本、语音和视觉特征，并建立统一的时间对应关系。本文以题目text列的原文词为索引，先确定可用于聚合的词时间区间，再将声学帧和视频帧映射到相应词上。最终100条样本均形成变长词级结果：1,934个原文词全部取得文本表示，1,421词取得通过自动筛查的主时间及声学表示，1,228词还取得区间内有效视觉表示。")

heading("问题分析", level=1)
heading("求解流程与建模目标")
paragraph("本问的关键在于统一三种不同粒度的数据：原文以词为单位，语音按短时帧采样，视觉按视频帧采样。视频与音轨的时间起点、识别文本与题目原文的词序差异，以及部分画面无法检出有效人脸，都会影响逐词对应。因此，本文依次解决素材对应、原文词时间定位、三模态特征提取和词区间聚合四项任务，并对每个词记录时间及模态的有效状态。")
paragraph("具体流程如图2所示。首先依据工作簿编号、文本和标签建立样本清单，完整解码原视频并分离WAV；随后由Whisper small.en识别音频、MFA定位识别词，再将识别词与原文作保序映射。文本分支直接编码题目原文，语音与视觉分支分别提取74维声学帧和52维表情帧；经时间筛查后，以原文词区间汇聚声画特征并保存矩阵、掩码和质量记录。情感标签只用于清单核对，不参与特征或时间计算。")
fig("fig01_flow.png", "图 2 问题一词级三模态特征构建流程")

heading("数据处理与分析", level=1)
heading("原始素材核验与时间轴")
paragraph("建立以视频编号为主键的100行样本清单后，逐一核查MP4、text与标签的对应关系。100条视频及其音轨均可完整解码；从视频重新提取的PCM与已保存WAV的哈希逐样本一致。视频有效时长为2.257—29.288 s（中位数6.722 s），WAV为2.228—29.109 s（中位数6.586 s）。两种媒体各自保留实测结束时间，不强行截取共同长度。")
paragraph("时间统一采用源媒体时间作为参照。WAV样本时间从音轨首个呈现时间戳（PTS）起算，按式（1）映射至源媒体；视觉帧直接保存解码PTS。这一处理保留了可能存在的音轨起点偏移，也避免用帧号和标称帧率近似真实画面时间。")
formula(r"t_{i}^{\mathrm{source}}=t_{i}^{\mathrm{WAV}}+\Delta_i^{\mathrm{audio}}", 1, 11.0)
paragraph("式中，tᵢᵂᴬⱽ为第i条WAV内的相对时间，tᵢˢᵒᵘʳᶜᵉ为对应的源媒体时间，Δᵢᵃᵘᵈⁱᵒ为音轨起点偏移，单位均为秒。本批100条样本的音轨与视频流起点均为0 s；偏移量仍写入逐样本记录，使时间换算规则可直接复用于非零起点素材。")

heading("原文词序列与时间证据")
paragraph("题目提供的text列决定最终词面与词序。Whisper small.en仅从音频生成识别文本[20]；MFA 3.4.2再以WAV和非空识别文本估计识别词区间[21]。两条工具链的职责不同：Whisper提供“听到了哪些词”的证据，MFA为该识别词序列定位；BERT始终编码题目原文。为比较词序，将原文和识别文本转为小写，统一弯引号并忽略词间标点；缩写不展开，数字与其拼写形式也不视为同词。归一化后，29条样本整句同词，69条存在词序差异，2条为空转写。98条非空识别文本中，MFA为95条导出词区间，另3条没有对齐输出。")
paragraph("获得MFA区间后，仍需将“识别词时间”转为“原文词时间”。本文建立原文—Whisper和Whisper—MFA词层两段保序映射，分别求最长公共子序列（LCS）。仅当一个同词位置配对出现在所有最优LCS中，且对应的目标位置唯一时，才沿两段映射继承MFA区间。因此，“不一致句中的唯一同词”指整句虽有差异，但某个原文词能确定地对应同一个识别词及MFA词；它不要求该词在句中只出现一次。MFA的静音和未知词标签不参与映射。")
add_table("表 2 原文—Whisper—MFA词序映射示例", ["样本与原文片段", "Whisper片段", "映射与结果"], [
    ["A层：their life, their marriage", "their life, their marriage", "词序一致；对应位置逐级映射到MFA区间。"],
    ["B层：the US government", "the U.S. government", "US被分为us，而U.S.分为u、s，故US不赋时；government为原文#10→Whisper#11→MFA#11，区间2.21—2.77 s。"],
    ["重复词：as as the product", "As the product", "两个原文as都可能对应识别出的唯一as，最优映射不唯一；两词主时间均留空。"],
], [5.0, 4.4, 6.0], 8.7, page_break_before=True)
paragraph("表2列出三种实际情况，依次来自样本-vxjVxOeScU$_$4、-UuX1xuaiiE$_$6与-lzEya4AM_4$_$6。第一例可按相同词序逐级对应。第二例虽然US与U.S.的切词方式不同，但后面的government仍有唯一对应：原文第10词→Whisper第11词→MFA第11词，获得2.21—2.77 s。第三例中原文连续出现两个as，识别文本仅有一个as；两种位置选择同样最优，故这两个原文词均不赋予主时间。")
paragraph("对继承的候选区间进一步检查时间范围、起止顺序、词序和相邻重叠：起点须非负、终点须晚于起点且不超过WAV时长，相邻区间重叠不得超过0.05 s；短于0.02 s或长于1.5 s的词标记异常。对于两工具均有边界估计的词，若任一起点或终点相差超过0.5 s，则保留候选记录但不纳入主时间。0.5 s是排查明显分歧的规则阈值，不代表真实边界误差。整句同词且通过筛查的词定义为A层；有句级差异但能经两段唯一映射并通过筛查的词定义为B层。两层的主时间均采用MFA区间。")
add_table("表 3 原文词时间的自动筛查结果", ["结果类别", "词数", "处理"], [
    ["A层：全文一致", "520", "采用MFA主时间"],
    ["B层：不一致句中的唯一同词", "901", "采用MFA主时间，单独记录证据层级"],
    ["Whisper/MFA边界分歧超过0.5 s", "99", "主时间留空，保留MFA候选"],
    ["未匹配或最优映射不唯一", "272", "主时间留空"],
    ["MFA无对齐输出", "77", "主时间留空"],
    ["Whisper无转写", "65", "主时间留空"],
], [5.6, 1.6, 8.2])
fig("fig02_evidence.png", "图 3 原文词时间证据层级与边界筛查")
paragraph("表3的各类结果互斥，合计1,934词；图3进一步展示证据层级、留空原因及两种时间估计的边界差。最终1,421词取得可用于声画聚合的主时间，覆盖率为73.47%；其中901词来自句级转写不一致的样本，体现了逐词映射对有效数据的恢复能力。其余513词保留空起止时间及无效掩码，同时保留原文词和文本特征；逐词映射位置、候选区间与筛查原因一并保存。")

heading("模型建立", level=1)
heading("三模态情感特征的定义与提取")
paragraph("根据语言内容、发声方式和面部表情对情感的互补描述，分别构建文本、语音、视觉三条特征分支。表4给出特征定义及原始采样粒度；三分支先独立提取，再以原文词时间汇合。本阶段不使用全语料统计量对特征作统一标准化。")
add_table("表 4 三模态特征的定义、维度与原始粒度", ["模态", "定义", "维度", "原始粒度"], [
    ["文本", "BERT-base-uncased最后一层；原文词内WordPiece均值", "768", "原文词"],
    ["语音", "20 MFCC＋20一阶差分＋20二阶差分＋6谱/能量＋6频带对比＋F0＋浊音概率", "74", "10 ms帧移"],
    ["视觉", "Face Landmarker的52个blendshape系数；另存单人脸有效掩码", "52", "约10 fps的真实视频帧"],
], [1.5, 10.3, 1.2, 2.4])
paragraph_lead("（1）文本特征。", "题目text列提供1,934个原文词，Whisper转写不进入BERT。固定修订版google-bert/bert-base-uncased[19]对整句编码，保留标点和句内上下文；利用字符偏移，将同一原文词对应的WordPiece归并。设Pᵢⱼ为第i句第j词的子词位置集，hᵢᵣ⁽ᴸ⁾为BERT末层第r个子词向量，按式（2）求均值得到768维词表示。模型未微调，最长输入84个token，低于512 token上限。词时间缺失不影响文本编码。")
formula(r"x_{ij}^{t}=\frac{1}{|P_{ij}|}\sum_{r\in P_{ij}}h_{ir}^{(L)},\qquad x_{ij}^{t}\in\mathbb{R}^{768}", 2, 10.8)
paragraph_lead("（2）语音特征。", "对完整的16 kHz单声道WAV使用2048点居中分析窗和160点帧移；因此帧中心每10 ms出现一次，分析窗覆盖0.128 s。利用librosa[22]逐帧提取20维MFCC，并在9帧邻域上分别求20维一阶、二阶差分，以同时描述谱包络及其动态变化。另提取RMS、过零率、谱质心、谱带宽、85%谱滚降、谱平坦度六项能量与频谱指标，以及六个频带的谱对比度；使用pYIN在50—600 Hz范围内估计基频F0与浊音概率。最终维度为20＋20＋20＋6＋6＋2＝74。无声帧的F0以零占位并另存有声掩码。100条样本共生成77,755个声学帧，其中51,538帧取得有效F0。")
paragraph_lead("（3）视觉特征。", "从原MP4按约10 fps采样真实解码帧，并保留每帧PTS。MediaPipe Face Landmarker在VIDEO模式下输出52个面部blendshape系数[23]，用于表征表情动作强度。以人脸数及归一化关键点边框面积判断帧有效性：仅当恰有一张脸且面积不小于0.01时，保存该帧的52维向量并将有效掩码置1；其他帧以零向量占位、掩码置0。100条视频共采样7,889帧，其中6,471帧有效、1,366帧无人脸、52帧多人脸；13条视频没有合格的单人脸帧。特征值与有效掩码分开保存，可明确区分“没有视觉观测”与“表情系数较低”。")

heading("以原文词为基准的跨模态对齐")
paragraph("设第i条样本有nᵢ个原文词，第j词通过筛查的主时间区间为Iᵢⱼ=[sᵢⱼ,eᵢⱼ)，其零点为WAV起点。音频第k帧取分析窗的真实中心时间tᵢₖᵃ；视频第ℓ帧先由源媒体PTS减去式（1）的音轨起点偏移，得到同一WAV时间轴上的tᵢℓᵛ。式（3）分别筛选落入词区间的声学帧及有效视觉帧，其中qᵢℓ为该视频帧的单人脸有效标记。半开区间避免相邻词边界上的帧重复归属；无主时间的原文词不建立帧集合。")
formula(r"\mathcal{K}_{ij}^{a}=\{k:t_{ik}^{a}\in[s_{ij},e_{ij})\},\qquad\mathcal{K}_{ij}^{v}=\{\ell:t_{i\ell}^{v}\in[s_{ij},e_{ij}),\ q_{i\ell}=1\}", 3, 15.0)
paragraph("按式（4）对词区间内的74维声学帧aᵢₖ和52维视觉帧vᵢℓ分别求均值，使所得xᵢⱼᵃ、xᵢⱼᵛ与原文第j词的文本向量占据矩阵同一行。声学帧中心相隔10 ms，视觉帧约每0.1 s采样一次，故短词可能取得声学均值而没有区间内视觉帧。集合为空时，该模态向量用零占位、有效掩码置0；F0分量仅汇总有声帧，并另存F0有效掩码。区间外最近的有效画面只保存为候选记录，不计入主视觉矩阵。")
formula(r"x_{ij}^{a}=\frac{\sum_{k\in\mathcal{K}_{ij}^{a}}a_{ik}}{|\mathcal{K}_{ij}^{a}|},\qquad x_{ij}^{v}=\frac{\sum_{\ell\in\mathcal{K}_{ij}^{v}}v_{i\ell}}{|\mathcal{K}_{ij}^{v}|}", 4, 13.8)
add_table("表 5 词级对齐模型的主要符号与含义", ["符号", "含义与单位"], [
    ["i，j；nᵢ", "样本、原文词序号；第i条样本的原文词数。"],
    ["tᵢˢᵒᵘʳᶜᵉ，tᵢᵂᴬⱽ，Δᵢᵃᵘᵈⁱᵒ", "源媒体时间、WAV相对时间、音轨起点偏移；单位均为秒。"],
    ["Pᵢⱼ，hᵢᵣ⁽ᴸ⁾，xᵢⱼᵗ", "第j个原文词所属WordPiece位置集、BERT末层第r子词向量、768维词向量。"],
    ["sᵢⱼ，eᵢⱼ；tᵢₖᵃ，tᵢℓᵛ", "词区间起止时间；声学帧中心和视频帧PTS经换算后的WAV时间，单位为秒。"],
    ["Kᵢⱼᵃ，Kᵢⱼᵛ；qᵢℓ", "落入词区间的有效声学/视觉帧索引集；单人脸有效标记（0或1）。"],
    ["aᵢₖ，vᵢℓ；xᵢⱼᵃ，xᵢⱼᵛ", "原始帧74维声学和52维视觉向量；均值聚合后的词级模态向量。"],
], [4.3, 11.1], 8.7, page_break_before=True)
paragraph("式（1）—（4）的符号见表5。每条样本的文本、声学和视觉矩阵尺寸分别为nᵢ×768、nᵢ×74和nᵢ×52；三矩阵行数均为原文词数nᵢ。文件同时保存词时间、A/B证据层级、音视频帧索引与计数、有效长度和文本、时间、声学、视觉四类掩码。模态帧集合为空时，式（4）不计算均值，而以零占位和无效掩码表示；没有主时间的词仍有BERT向量。样本词数为5—65，中位数18，本阶段保留变长序列；训练时的补齐位置另以padding掩码标识。")

heading("结果分析与可视化", level=1)
heading("全量结果与质量核查")
paragraph("对100份逐样本结果逐条复查原文词索引、矩阵维数、时间范围与递增性、帧计数、掩码和逐词状态，100份文件均通过结构核对，1,934行文本向量均能回溯至原文词。图4以全部1,934词为统一分母比较三模态覆盖情况。")
fig("fig03_coverage.png", "图 4 全量原文词的三模态覆盖情况")
paragraph("由图4可见，文本特征覆盖全部原文词；1,421词具有主时间及声学均值，占73.47%；其中1,228词取得区间内有效视觉，占全部原文词的63.50%、有主时间词的86.42%。在193个有主时间而无有效视觉的词中，44词的区间内没有采样帧，149词虽有采样帧却未形成合格单人脸特征。该拆分明确了采样密度与画面有效性两类原因。")
paragraph("为检验结果是否集中于少数样本，图5将100条视频分别按主时间和视觉词覆盖率绘制，并给出对应的经验累计分布。散点颜色区分整句同词、词序不一致及空转写；A—D标记图6—9的案例。")
fig("fig04_distribution.png", "图 5 100条样本的词时间与视觉覆盖率分布")
paragraph("图5显示样本之间存在明显差异：16条样本没有主词时间，21条没有词级主视觉，其中13条的原视频采样帧内没有合格单人脸。所有样本仍保存完整声学逐帧矩阵和视频采样记录。附表A1列出100条样本的编号、媒体时长、特征维数、对齐粒度及有效词数，便于逐条核对。")

heading("典型样本的多层证据展示")
paragraph("表6选择四类具有代表性的样本：完整对齐、局部词恢复、有词时间但无有效视觉，以及空转写。图6—9均展示整条样本时间轴，并沿视频选取五帧实际画面；波形、词区间与逐词掩码共同构成可追溯的对齐证据。")
selected = {r["sample_id"]: r for r in ROWS}
cases = [
    ("A层完整", "-vxjVxOeScU$_$4"),
    ("B层局部恢复", "-UuX1xuaiiE$_$6"),
    ("主时间有、视觉缺失", "-HwX2H8Z4hY$_$9"),
    ("空转写、主时间缺失", "-yRb-Jum7EQ$_$1"),
]
add_table("表 6 四类典型样本的词级对齐结果", ["状态", "样本编号", "原文词", "主时间", "有效视觉"], [
    [kind, sample_id, str(selected[sample_id]["word_count"]), str(selected[sample_id]["timed_words"]), str(selected[sample_id]["visual_words"])] for kind, sample_id in cases
], [3.5, 5.5, 2.0, 2.0, 2.4], 8.6)
paragraph("样本-vxjVxOeScU$_$4的21个原文词均通过主时间和视觉筛查。图6从句首You're至句末long连续展示波形与词区间，并在五个不同词区间给出真实画面；四类掩码均有效，说明该样本完成了整句声画聚合。")
fig("fig05_case_A.png", "图 6 A层完整对齐样本的整句多层证据", 15.3)
paragraph("样本-UuX1xuaiiE$_$6的22个原文词与整句识别文本不一致，但20词经两段唯一映射及边界筛查获得主时间，18词取得有效视觉。图7中，原文第9词US因U.S.的切词差异而不赋时，第1词I因两工具边界分歧被筛除；其余可定位词仍可参与声画聚合。")
fig("fig06_case_B.png", "图 7 B层局部恢复样本的整句多层证据", 15.3)
paragraph("样本-HwX2H8Z4hY$_$9有22个原文词，其中17词可聚合声学帧，却没有词取得有效视觉。图8的五帧覆盖整段视频，画面主体位于电视屏幕内，均未通过单人脸有效性规则；相应视觉掩码为空，文本、时间和声学结果仍完整保留。")
fig("fig07_case_visual_missing.png", "图 8 有词时间而无有效视觉的整句证据", 15.3)
paragraph("样本-yRb-Jum7EQ$_$1含49个原文词，Whisper未给出转写。图9仍展示整段约29 s的音频变化及五帧视频画面；由于缺少可映射的识别词序列，原文词不分配声画区间。该样本49词的BERT表示全部保留，词级时间、声学及视觉掩码为0。")
fig("fig08_case_asr_empty.png", "图 9 空转写样本的整句声画与词级状态", 15.3)

heading("复现与交付")
paragraph("最终交付100份逐样本NPZ、逐词CSV索引和全量统计表，逐词矩阵与时间区间、音视频帧映射、有效长度及掩码配套保存。表7汇总主要模型、参数和结果位置；完整模型哈希、逐样本参数、代码及质量日志保存在E.1目录，支持从原始素材追溯至词级矩阵。")
add_table("表 7 主要模型参数与结果文件位置", ["环节", "关键设置", "结果入口"], [
    ["转写与词时间", "Whisper small.en；MFA 3.4.2；边界差阈值0.5 s", "结果/03；日志/第4步"],
    ["文本", "bert-base-uncased；768维；原文WordPiece均值", "结果/05_BERT_base_uncased"],
    ["语音", "librosa 0.11.0；16 kHz；2048点窗/160点帧移；74维", "结果/06_Librosa74"],
    ["视觉", "MediaPipe 0.10.32；约10 fps；52维；单人脸面积≥0.01", "结果/07_FaceLandmarker52"],
    ["词级交付", "半开词区间、真实帧时间、三模态矩阵和掩码", "结果/08_原文词级跨模态对齐"],
], [2.3, 8.6, 4.5], 8.4)
paragraph("本节从附件1原始视频得到的是变长词级表示，视觉维度为52。问题二、三使用附件2提供的固定50步对齐特征，视觉维度为35；因此，本节结果作为独立的原始素材特征交付。若与后两问联合建模，需另行定义时间重采样、视觉维度映射和归一化规则。")
heading("本章小结", level=1)
paragraph("本文以题目原文词为统一索引，完成100条视频的素材核验、词时间定位、BERT文本编码、74维声学与52维视觉提取，并形成可追溯的词级三模态矩阵。1,934词全部取得768维文本特征，1,421词取得主时间及声学表示，1,228词取得区间内有效视觉。两段保序映射使901个句级转写不一致的词仍可获得主时间；四类典型样本进一步展示了完整对齐、局部恢复及缺失处理。后续使用结果时，时间和模态掩码与特征矩阵共同构成模型输入，保留了每个词的数据有效性信息。", keep_together=True)

# References are numbered after the existing [1]–[18] bibliography of main.pdf.
ref_title = doc.add_paragraph(style="Heading 2")
ref_title.add_run("本节参考文献（整合时并入全文参考文献表）")
tex.extend(["% 下列文献并入全文参考文献表，按最终引用顺序重新编号。", ""])
references = [
    "[19] Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of deep bidirectional transformers for language understanding[C]. NAACL-HLT, 2019: 4171–4186.",
    "[20] Radford A, Kim J W, Xu T, et al. Robust speech recognition via large-scale weak supervision[C]. ICML, 2023: 28492–28518.",
    "[21] McAuliffe M, Socolof M, Mihuc S, et al. Montreal Forced Aligner: Trainable text-speech alignment using Kaldi[C]. Interspeech, 2017.",
    "[22] McFee B, Raffel C, Liang D, et al. librosa: Audio and music signal analysis in Python[C]. SciPy, 2015: 18–25.",
    "[23] Google AI for Developers. MediaPipe Face Landmarker: 52 blendshape coefficients[EB/OL]. https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/drawing_styles/face_landmarker/Blendshapes.",
]
for item in references:
    p = doc.add_paragraph(item)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.left_indent = Pt(22)
    p.paragraph_format.first_line_indent = Pt(-22)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.1
    for run in p.runs:
        text_run(run, "Times New Roman", 10.5)
    tex.extend([tex_escape(item), ""])

# The standalone Word file keeps the 100-sample table with Problem 1.
# When merging TeX into main, place the longtable after the paper's references or in its appendix.
tex.append("% 并入main时建议将下面的附表A1移至全文附录，正文末尾保留交叉引用。")
land = doc.add_section(WD_SECTION.NEW_PAGE)
land.orientation = WD_ORIENT.LANDSCAPE
land.page_width = Cm(29.7)
land.page_height = Cm(21)
land.top_margin = Cm(1.45)
land.bottom_margin = Cm(1.45)
land.left_margin = Cm(1.3)
land.right_margin = Cm(1.3)
land.footer_distance = Cm(0.7)
heading("附表A1 附件1全部100条样本的特征交付汇总", level=0)
paragraph("原始有效时长分别按视频流与完整WAV实测，不将两者强行截成同一结束点。每条均交付原文词级文本768维、声学74维、视觉52维矩阵，聚合粒度为原文词；“主时间/视觉”列分别表示有自动筛查通过时间和区间内合格单人脸视觉的词数。")
headers = ["序号", "样本编号", "视频/WAV s", "词数", "主时间", "视觉", "模态类型与维数", "对齐粒度", "转写比对"]
group_name = {"exact_text": "一致", "text_mismatch": "不一致", "empty_asr": "空转写"}
table_rows = [[
    str(r["order"]), r["sample_id"], f'{r["video_duration_s"]:.2f}/{r["wav_duration_s"]:.2f}',
    str(r["word_count"]), str(r["timed_words"]), str(r["visual_words"]),
    "T768/A74/V52", "原文词", group_name[r["transcript_group"]],
] for r in ROWS]
appendix_widths = [1.2, 7.2, 3.0, 1.5, 1.8, 1.6, 4.2, 2.5, 2.2]
add_table("附表 A1 100条原始样本的编号、媒体有效时长、维数、对齐粒度和覆盖数。", headers, table_rows[:25], appendix_widths, 8.5, appendix=True)
for start, stop in ((25, 50), (50, 75), (75, 100)):
    continuation = doc.add_section(WD_SECTION.NEW_PAGE)
    continuation.orientation = WD_ORIENT.LANDSCAPE
    continuation.page_width = Cm(29.7)
    continuation.page_height = Cm(21)
    continuation.top_margin = Cm(1.45)
    continuation.bottom_margin = Cm(1.45)
    continuation.left_margin = Cm(1.3)
    continuation.right_margin = Cm(1.3)
    continuation.footer_distance = Cm(0.7)
    add_table("附表 A1（续）100条原始样本的特征交付汇总。", headers, table_rows[start:stop], appendix_widths, 8.5, appendix=True)

tex.extend([
    r"\begin{landscape}",
    r"\scriptsize",
    r"\begin{longtable}{r l r r r r l l l}",
    r"\caption{附件1全部100条样本的特征交付汇总。视频/WAV时长为完整媒体流实测秒数；主时间和视觉列为有效原文词数。}\\",
    r"\toprule",
    " & ".join(tex_escape(x) for x in headers) + r" \\",
    r"\midrule\endfirsthead",
    r"\caption[]{附表A1（续）附件1全部100条样本的特征交付汇总}\\",
    r"\toprule",
    " & ".join(tex_escape(x) for x in headers) + r" \\",
    r"\midrule\endhead",
])
for row in table_rows:
    tex.append(" & ".join(tex_escape(x) for x in row) + r" \\")
tex.extend([r"\bottomrule", r"\end{longtable}", r"\end{landscape}", ""])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx-output", type=Path, default=DOCX)
    parser.add_argument("--tex-output", type=Path, default=TEX)
    args = parser.parse_args()
    PAPER.mkdir(exist_ok=True)
    assert len(ROWS) == 100 and sum(r["word_count"] for r in ROWS) == STATS["word_count"]
    args.docx_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.docx_output)
    args.tex_output.write_text("\n".join(tex), encoding="utf-8")
    print(json.dumps({"docx": str(args.docx_output), "tex": str(args.tex_output), "figures": 8, "sample_rows": len(ROWS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
