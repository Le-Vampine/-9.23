# E.1：第一问新流程工作区

本目录是按用户确认的新流程建立的独立工作区。旧项目和旧版特征结果不作为本目录的下游输入；原始标签、文本、样本数量和样本编号均保留。

## 当前进度

| 步骤 | 状态 | 输出 |
| --- | --- | --- |
| 1. 样本清单与编号、路径、文本、标签核对 | 已完成 | `输入/样本清单/` |
| 2. 视频检查、音频分离与原始时间轴映射 | 已完成 | `输入/音频WAV/`、`日志/第2步视频检查与音频分离/` |
| 3. Whisper small.en 英文识别与词级时间戳 | 已完成 | `结果/03_Whisper_small_en/逐样本/`、`日志/第3步Whisper_small.en/` |
| 4. 原文比对与词时间戳自动筛查 | MFA批量对齐及映射已完成；3条对齐失败按缺失保留 | `日志/第4步文本一致性检查/` |
| 5. 原文 BERT-base-uncased 逐词编码 | 已完成：100 条、1,934 词，全部生成 768 维向量 | `结果/05_BERT_base_uncased/` |
| 6. Librosa 74 维声学时序特征 | 已完成：100 条、77,755 帧 | `结果/06_Librosa74/` |
| 7. MediaPipe 52 维表情时序特征及有效掩码 | 已完成：100 条、7,889 个采样帧 | `结果/07_FaceLandmarker52/` |
| 8. 按原文词时间聚合音视频 | 已完成：1,421 个词有音频，1,228 个词有区间内有效视觉 | `结果/08_原文词级跨模态对齐/` |
| 9. 保存对齐矩阵、时间映射、长度和掩码 | 已完成：100 份逐样本 NPZ 与逐词索引 | `结果/08_原文词级跨模态对齐/` |
| 10. 全量质量检查与汇总图 | 已完成：100 条与 1,934 词的文件、维度、长度、掩码核验；5 张论文图 | `结果/10_全量质量检查与论文图表/` |
| 11. 交付候选与论文说明 | 已完成候选：核心附件、复现说明和问题一五节精修稿 | `论文/`、`交付候选/` |

第 1、2 步的 MP4、Excel 和 WAV 输入均位于本目录下；100 个 MP4 从原始附件逐条复制并做 SHA-256 完整性对照。WAV 是此前已存在的完整提取结果，已在本目录中再次与源 MP4 临时提取的 PCM 样本逐条核验一致。

第 1 步核查结果：100 行标签、100 个唯一编号、100 个视频，标签文件副本和视频副本均与原始附件一致。第 2 步核查结果：100/100 视频音频流完整解码通过；100/100 个 WAV 与从新目录内源 MP4 临时提取的 PCM 逐样本哈希一致；全部视频和音轨起点均为 0 秒；没有样本需要复核。

第 3 步已对 100 条 WAV 运行本目录内的 `small.en` 模型，生成 100 份识别 JSON，共 1,956 个词级时间戳。识别器没有读取题目原始转写文本。词时间是 Whisper 的估计值，不是强制对齐结果。98 条得到非空转写；`-NFrJFQijFE$_$1` 和 `-yRb-Jum7EQ$_$1` 两条音频有明显信号但没有转写，已标记为第 4 步优先核对项，保持空值，不补造词。

第 4 步用程序逐条比较工作簿原文和 Whisper 转写，并生成逐词编辑对照。比较时忽略大小写和标点、统一弯引号；不自动扩展缩写或把数字词互换。100 条中，29 条归一化后完全一致，69 条有词差异，2 条没有 Whisper 转写。

原先得到的 1,568 个时间是 Whisper 的估计。本轮改用 MFA 3.4.2 对 98 条非空 Whisper 转写强制对齐，再把 MFA 词时间映射回题目原文。MFA 曾因 Kalpy 延迟导入 librosa 时 Numba 默认缓存目录初始化停滞；将 `NUMBA_CACHE_DIR` 移到 E.1 本地后，批次在约 59 秒内完成。98 条中 95 条导出词级 JSON，3 条（`sample_0022`、`sample_0054`、`sample_0098`）对齐失败，相关原文词留空。失败样本及逐词原因见 MFA 对齐映射汇总表。

当前主时间戳采用 MFA 时间，不采用 Whisper 时间；Whisper 时间只作为边界差异筛查。对文本完全一致的样本，只保留唯一映射且通过区间检查的 MFA 时间。对文本不一致的样本，若原文词在官方原文→Whisper、Whisper→MFA 两段映射中都属于所有最优 LCS 下的唯一必选同词，MFA 区间通过检查，且可比对时两种方法的任一边界差不超过 0.5 秒，则保留为 B 层主时间并明确标记文本不一致；完全一致样本为 A 层。超过 0.5 秒的 99 个词保留 MFA 候选以供追踪，但主时间留空。0.5 秒是本项目的自动筛查阈值，不是误差界限或精度保证。

1,934 个原文词中，最终有 1,421 个 MFA 主时间（73.5%）：520 个 A 层完全一致词、901 个 B 层唯一同词映射词。513 个词留空：99 个边界差异超阈值、272 个未能唯一映射到 ASR 词、77 个来自三条 MFA 失败样本、65 个来自两条空转写样本。MFA 强制对齐仍是给定转写条件下的模型估计，不证明词必然被说出，也没有人工边界真值；无需人工听音或标边界。

## 时间轴约定

WAV 不做时间裁剪或尾部补齐。WAV 的本地 0 秒对应源音轨首个呈现时间戳；每条样本均记录音轨起点与视频起点的偏移，并使用：

`源媒体时间 = WAV 时间 + 源音轨起点`

视频与音轨时长不同时保留各自真实时长，并在逐样本报告中记录差值。

## 已确认的目标特征维数

- 文本：`google-bert/bert-base-uncased`，768 维；模型修订版 `86b5e0934494bd15c9632b12f734a8a67f723594`。
- 音频：Librosa，74 维：20 MFCC、20 一阶差分、20 二阶差分、6 个能量/谱统计量、6 个频带对比度、F0 与浊音概率。
- 视觉：MediaPipe Face Landmarker，52 维，另存有效性标记。

## 复现第 1–3 步

在项目根目录运行（需使用含 `openpyxl` 的 Python，以及 FFmpeg/FFprobe）：

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' '.\E.1\代码\样本清单\建立样本清单.py'
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' '.\E.1\代码\媒体检查与音频分离\检查并分离音频.py' --ffmpeg 'D:\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe' --ffprobe 'D:\ffmpeg-master-latest-win64-gpl-shared\bin\ffprobe.exe'
```

第 2 步脚本会完整解码检查 MP4，并临时重提音频与现有 WAV 做 PCM 逐样本比对。对已有 WAV 只核验、不覆盖；缺少 WAV 时才补提。输出报告写入 `日志/第2步视频检查与音频分离/`。

第 3 步复现命令使用本机已有的 faster-whisper 环境，模型已复制并校验在 E.1 内：

```powershell
& '.\E题第一问项目\.whisper_env\Scripts\python.exe' '.\E.1\代码\Whisper识别\运行WhisperSmallEn.py' --cpu-threads 4
& '.\E题第一问项目\.whisper_env\Scripts\python.exe' '.\E.1\代码\Whisper识别\核验Whisper输出.py'
```

逐条原始 ASR 输出在 `结果/03_Whisper_small_en/逐样本/`；全量转写表、模型哈希和运行参数在 `日志/第3步Whisper_small.en/`。

第 4 步自动比对、MFA 对齐与原文词时间映射可复现为：

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' '.\E.1\代码\文本比对与强制对齐\自动比对原始文本与Whisper.py'
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' '.\E.1\代码\文本比对与强制对齐\恢复原文逐词时间.py'
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' '.\E.1\代码\文本比对与强制对齐\准备MFA全量Whisper转写语料.py'
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' '.\E.1\代码\文本比对与强制对齐\运行MFA命令.py' align_whisper
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' '.\E.1\代码\文本比对与强制对齐\映射MFA时间回原文.py'
```

最终逐词时间见 `日志/第4步文本一致性检查/原文词级时间_MFA候选及最终采用.csv`；其中 `final_primary_*` 列是后续特征对齐采用的时间，`final_primary_valid_mask` 为有效掩码。`mfa_candidate_*` 列保留未纳入主时间的 MFA 候选。逐样本统计和规则说明在 `MFA原文时间映射逐样本统计.csv`、`MFA原文时间映射汇总.json`。Whisper 原始估计及旧版逐词恢复表只作对照，不能表述为人工真值或精度验证结果。

## 第 5 步：原文 BERT 文本特征

文本输入直接读取 `输入/原始标签/label-100.xlsx` 的 `label` 工作表 `text` 列，并逐条对照样本清单中的 `text`。**不使用 Whisper 转写做文本编码。**原文按第 4 步的词切分规则产生 1,934 个词，并逐词核对第 4 步的 `sample_id`、`official_word_index` 和词面；因此第 5 步特征行可按编号与词级时间表连接。

模型固定为官方 `google-bert/bert-base-uncased` 修订版 `86b5e0934494bd15c9632b12f734a8a67f723594`，不进行微调。官方 ONNX 文件公开的输出是掩码语言模型的 30,522 维词表 logits；`准备BERT编码器.py` 从同一文件保留最后一层编码器及其原权重，去掉词表预测头，输出每个 BERT token 的 768 维隐藏状态。原文整句送入 BERT，保留标点作为上下文；每个原文词由其覆盖的 WordPiece 最后一层向量算术平均得到。正文字段只做 NFKC 与弯引号规范化，以匹配第 4 步词切分；没有用音频时间或 Whisper 文本改变词向量。

输出文件：

| 文件 | 内容 |
| --- | --- |
| `结果/05_BERT_base_uncased/原文逐词BERT特征.npy` | `float32`，形状 `(1934, 768)`；第 `i` 行对应索引表 `feature_row_0based=i` |
| `结果/05_BERT_base_uncased/原文逐词索引.csv` | 样本编号、原文词序、BERT 子词位置、特征和主时间有效标记 |
| `结果/05_BERT_base_uncased/逐样本统计.csv` | 每条样本的矩阵起始行和词数 |
| `结果/05_BERT_base_uncased/运行报告.json` | 输入/模型/输出哈希、版本、维度与覆盖数 |

100 条样本的 1,934 个原文词均有文本向量；其中 1,421 个词有第 4 步通过筛查的主时间，513 个词主时间为空。**文本特征有效不等于语音时间有效。**第 8 步仅对有有效时间的词进行词区间音视频聚合；其余词保留文本向量和缺失时间掩码。

本机复现命令（运行环境使用项目已有的 `.whisper_env`；所需 `onnx` 和 `openpyxl` 分别放在 E.1 的 `运行环境/onnx_packages`、`运行环境/text_packages`）：

```powershell
& '.\E题第一问项目\.whisper_env\Scripts\python.exe' '.\E.1\代码\文本编码\准备BERT编码器.py'
& '.\E题第一问项目\.whisper_env\Scripts\python.exe' '.\E.1\代码\文本编码\提取原文BERT词特征.py'
```

官方下载模型与词表保存在 `模型/BERT_base_uncased/`。官方 `model.onnx` 的 SHA-256 为 `44d7a2896d341c51fb1eba89aea3a590e6af0ce33e25481136f7eeecb62e5f7f`；导出编码器的 SHA-256 与转换说明在同目录 `model_encoder_768.json`，输入与特征文件哈希见运行报告。全部 100 条输入长度均低于 BERT 的 512 token 上限，最长为 84 token。

## 第 6 步：Librosa 声学时序特征

直接从第 2 步核验过的 100 条完整 WAV 提取帧特征，不裁剪音频，也不依据文本或 Whisper 转写选择帧。每条 WAV 均为 16 kHz、单声道、PCM16。采用 2,048 采样点的居中分析窗和 160 采样点的帧移；相邻帧中心相隔 0.01 秒。`frame_time_wav_s` 为 WAV 起点起算的分析窗中心，`frame_time_source_s` 在此基础上加第 2 步记录的源音轨起点偏移。聚合到词时，应使用这些帧中心与第 4 步主词时间相交。

每帧的 74 列依次为：20 个 MFCC、20 个一阶差分、20 个二阶差分、RMS、过零率、谱质心、谱带宽、85% 谱滚降频率、谱平坦度、6 个频带对比度、F0 和浊音概率。`librosa.feature.spectral_contrast` 使用 `n_bands=5`，它实际输出 6 个频带。基频由 `librosa.pyin` 在 50–600 Hz 范围估计；未判为有声的帧将 F0 存为 0，同时在 `f0_voiced_mask` 中标为 0，不能把该 0 当成真实 0 Hz 测量值。浊音概率和有声标记是模型估计，不是人工语音活动真值。当前逐帧原始特征没有做跨语料标准化。

输出在 `结果/06_Librosa74/逐样本/`，每个 `.npz` 包含 `features` (`帧数 × 74`, float32)、`frame_time_wav_s`、`frame_time_source_s`、`f0_voiced_mask` 和 `frame_valid_mask`。列名与参数见 `特征定义.json`；逐样本帧数、哈希和基频覆盖见 `逐样本统计.csv`；总数和版本见 `运行报告.json`。共得到 77,755 帧，其中 51,538 帧有有效 F0。两条样本的有效 F0 帧数为 0，声学矩阵仍正常保存并保留基频掩码。

本机复现命令使用已安装 Librosa 0.11.0 的 MFA Python 环境；脚本会把 Numba 缓存放在 E.1 中，并设置该环境的 DLL 路径：

```powershell
& 'C:\Users\Administrator\miniforge3\envs\mfa\python.exe' '.\E.1\代码\声学特征\提取Librosa74维.py'
```

## 第 7 步：MediaPipe 表情特征

从 E.1 内第 2 步核验的 100 条原始 MP4 解码视频帧，按约 10 fps 从实际解码帧中选帧。每个保存的时间是原帧呈现时间戳（PTS），不是人为构造的等间隔时间；同时保存 `frame_time_wav_s = frame_time_source_s - 源音轨起点`，供第 8 步与词时间连接。全部 23,241 个原始视频帧已解码，其中选取 7,889 帧进行 Face Landmarker 推理。

使用 MediaPipe Face Landmarker `float16/1` 模型、VIDEO 模式、最多检测 3 张人脸，开启 `output_face_blendshapes`。52 维依次为模型的标准表情系数，包含 `_neutral`；完整列顺序见 `结果/07_FaceLandmarker52/特征定义.json`。官方模型文件单独保存在 `模型/FaceLandmarker/face_landmarker.task`，SHA-256 为 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`。

仅在检测到**恰好一张脸**且归一化关键点边框面积至少为 0.01 时，将其 52 维系数标为有效。无人脸、多人脸或脸过小时保留该帧的真实时间，将系数写为零占位，同时设置 `face_valid_mask=0`；零占位不表示中性表情。每个 `.npz` 还保存 `face_detected_mask`、`multi_face_mask`、`face_count`、主脸边框面积及源视频帧序号。总计 7,889 个采样帧中，6,471 帧有有效单人脸表情特征，1,366 帧无人脸，52 帧多人脸；13 条视频没有任何有效人脸帧，其视觉结果仍保留，后续词级聚合应使用掩码。

3 条视频出现“真实末帧 PTS 晚于容器报告的结束时间”的元数据差异。程序保留真实帧 PTS，在 `逐样本统计.csv` 中记录超出的帧数和秒数，没有改写时间或删帧。逐样本 `.npz` 位于 `结果/07_FaceLandmarker52/逐样本/`，汇总见 `运行报告.json`。模型的 52 类定义和视频模式接口可参阅 [Google 官方 Blendshapes 文档](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/drawing_styles/face_landmarker/Blendshapes)与 [FaceLandmarker API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarker)。

本机复现命令使用已有的 `.vision_env`（MediaPipe 0.10.32、PyAV 18.1.0）：

```powershell
& '.\E题第一问项目\.vision_env\Scripts\python.exe' '.\E.1\代码\视觉特征\提取FaceLandmarker52维.py'
```

## 第 10–11 步：全量核验、论文图表与交付候选

`代码/质量检查/生成第10步论文证据.py` 逐条核对主 NPZ 与输入清单、原文词时间和逐词索引，生成 100 条全量汇总表、统计 JSON 和基础论文图。图2优化后的问题一正文为 `论文/问题一_五节结构与图表精修稿_图2优化版.docx`，含 8 张图、7 张表、4 个公式及附表 A1 的 100 条逐样本记录；同名 PDF 用于预览，`问题一_五节结构与图表精修稿.tex` 可用于并入 main。`论文/图表精修/论文插图/` 是论文图和公式的源目录，`论文/图表精修/流程图_可编辑/editable.pptx` 为原生可编辑流程图，`论文/图表精修/问题一图表总览_图2优化版.pptx` 汇集全部图。`论文/复现与交付说明.md`、`论文/环境与跨机复现说明.md` 解释主矩阵读取、掩码、版本和重跑条件。

作图源文件保留在 `代码/论文/`；流程图的定稿预览为 `论文/图表精修/流程图_可编辑/preview.png`。MFA 声学模型与词典的保留位置为 `MFA/models/`，批次运行时的 `work*` 目录与 Numba 缓存属于可重新生成的临时产物。

`代码/质量检查/核验问题一提交材料.py` 重新检查原附件哈希、100份逐模态与主特征、1,934个词、掩码、论文附表100行和图2嵌入，并生成 `交付候选/100条交付索引.csv` 与 `提交前全量核验.json`。`代码/质量检查/打包问题一交付候选.py` 生成问题一压缩组件与CRC、条目哈希和容量报告，保留100份主对齐NPZ、逐帧特征、代码、配置、必要日志、交付索引及复现说明，排除原始MP4、WAV、模型权重及Python环境。正式赛事附件仍须与问题二、三的材料合计核算≤50 MB；问题一组件不代表完整参赛附件。

在本工作区查看 `交付候选/请提交前阅读_问题一.md` 可找到本次提交组件和论文稿。旧 `问题一核心附件.zip` 仅作历史候选，勿作为本次提交组件；独立解压 `问题一提交组件.zip` 后，可直接阅读其中的 `E.1/论文/问题一提交核对清单.md` 与 `E.1/论文/环境与跨机复现说明.md`。

```powershell
& '.\E题第一问项目\.vision_env\Scripts\python.exe' '.\E.1\代码\质量检查\生成第10步论文证据.py'
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' '.\E.1\代码\质量检查\打包问题一交付候选.py'
```

## 第 8–9 步：按原文词时间对齐并保存三模态矩阵

只采用第 4 步 `final_primary_*` 的 1,421 个有效词区间作为时间基准，保留其 A/B 证据层级；不以 Whisper 的原始词时间替代缺失时间。每个词取 `[start,end)` 内的声学帧中心和真实视频帧 PTS。74 维音频对区间内有效帧求均值，其中 F0 列只对 `f0_voiced_mask=1` 的帧求均值。52 维视觉只对区间内 `face_valid_mask=1` 的单人脸帧求均值，不跨无效人脸帧插值。

主结果为 100 份逐样本 `.npz`，位于 `结果/08_原文词级跨模态对齐/逐样本/`。每份按原文词序保存 `text`（词数×768）、`audio`（词数×74）、`visual`（词数×52）、词起止时间、`valid_length`、文本/词时间/音频/F0/视觉有效掩码和参与聚合的帧数。没有可信词时间的 513 个词仍有原文 BERT 向量；其时间为 NaN，音视频向量为零占位且掩码为 0。零占位不能解释为实际声学或表情值。`逐词对齐索引.csv` 记录每词使用的音视频帧索引范围、时间来源、状态和证据层级；`逐样本对齐统计.csv` 与 `对齐汇总.json` 汇总覆盖率。

全量统计：1,934 个原文词均有文本特征；1,421 个词有第 4 步主时间且全部有对应声学帧；其中 1,228 个词在区间内还有有效人脸帧。余下 193 个有时间词中，44 个区间内无采样视频帧、149 个有帧但无有效单人脸。另为 45 个缺少区间内有效人脸的词保存了距离词区间不超过 0.05 秒的**近邻视觉候选**，放在 `visual_nearest_candidate` 并有独立掩码；主 `visual` 矩阵仍保持缺失。1,302 个词在其音频区间内有有效 F0 帧。

复现命令：

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' '.\E.1\代码\词级对齐\按原文词时间聚合音视频.py'
```

详细规则见 `结果/08_原文词级跨模态对齐/对齐规则.json`。第 10 步尚需做跨样本质量检查与典型样本对齐图，不能将模型时间戳的自动筛查表述为人工真值精度验证。
