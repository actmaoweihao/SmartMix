# SmartMix

SmartMix 是一个本地运行的智能混音工作台。它面向音乐爱好者、活动策划者和轻量 DJ 场景：上传多首歌曲后，系统会自动分析 BPM、调性、Camelot 编码、能量、响度、节拍网格和可过渡片段，帮助你生成更顺耳的播放顺序，预览两两重叠的混音效果，并导出 MP3 或 WAV。

项目由 Vite 前端和 FastAPI 后端组成，核心音频分析与导出在本地完成，不需要云端账号。

当前版本还包含两个偏 DJ 教学和调试的工作流：

- “教学入口”会基于当前 Deck A 推荐下一首歌、接歌方法、操作步骤，并可调用后端生成真实无缝试听。
- “分轨调试”会在歌曲上传分析完成后自动排队调用 Demucs，生成真实 vocals / drums / bass / other 分轨；等待期间用灰色模拟分轨和扫描态占位，完成后自动切到蓝色真实分轨。

## 主要功能

- 多音频上传：支持常见音频格式，包括 MP3、WAV、FLAC、M4A、OGG、AAC、AIFF、OPUS、WEBM 等。
- 自动音频分析：识别时长、BPM、Key、Camelot、多指标能量、LUFS、真峰值、beat grid、bar、phrase、波形峰值和推荐过渡点。
- 智能排序：支持综合推荐、谐和优先、BPM 升序、BPM 降序、能量弧线和原始顺序。
- 两首歌衔接评分：上传任意两首歌，按 Camelot、BPM、能量和结构可过渡性计算 A 到 B / B 到 A 的匹配分。
- 自动匹配修复：根据两歌差异自动生成调性、速度和能量处理方案，并输出修复后的音频版本。
- Harmonic tuning 建议：当两首歌调性不够兼容时，给出可调到的 Camelot 目标、半音数和质量风险。
- 实时混音预览：使用 Web Audio API 在浏览器里预览淡入淡出、EQ、滤波扫频和动态 EQ。
- DJ 教学与无缝试听：推荐下一首歌、接法、步骤和风险提示，并可生成后端渲染的真实过渡试听。
- Demucs 分轨调试：歌曲上传并完成基础分析后会自动排队生成 vocals、drums、bass、other 四个真实 stems；调试界面会在分轨未完成时显示灰色模拟波形和扫描态，完成后切换到蓝色真实分轨。
- 可视化编辑：显示选中歌曲波形，可拖动 IN/OUT 手柄调整入点和出点；时间线展示整段混音结构。
- 双 Deck 控制：对当前过渡的上一首和下一首分别调节增益、Low、Mid、High。
- 精准过渡：按 4/8/16 小节计算重叠区间，结合 intro/outro 候选点和人声密度避让。
- 后端导出：按当前排序、过渡点、EQ、响度归一化、节拍同步和混音策略导出 MP3/WAV。
- 项目保存与加载：把曲目顺序、过渡点、设置和混音参数保存到本地后端数据目录。
- 高质量调音：可选安装 Demucs 和 Rubber Band，对歌曲做 Camelot 调性转换。

## 快速开始

### 1. 准备环境

建议使用：

- Node.js 20+
- pnpm
- Python 3.11+

Windows、macOS、Linux 都可以运行。MP3 导出使用 `imageio-ffmpeg` 自带的 ffmpeg，通常不需要单独安装系统 ffmpeg。

### 2. 安装依赖

```bash
pnpm install
pnpm setup:backend
```

如果还没有 pnpm，可以先安装：

```bash
npm install -g pnpm
```

### 3. 启动项目

```bash
pnpm dev
```

启动后访问：

- 前端工作台：http://127.0.0.1:3000
- 后端健康检查：http://127.0.0.1:8002/api/health

`pnpm dev` 会同时启动：

- `pnpm frontend`：Vite 前端，默认端口 `3000`
- `pnpm backend`：FastAPI 后端，默认端口 `8002`

## 基础使用流程

1. 打开 http://127.0.0.1:3000。
2. 点击“选择音频”，或把音频文件拖进上传区域。
3. 等待每首歌分析完成，列表中会显示时长、BPM、调性、风格、能量和过渡信息；如果已安装 Demucs 依赖，系统会在后台自动排队生成真实分轨。
4. 在左侧选择排序策略，然后点击“应用排序”。
5. 调整过渡时长、AI 精准小节混音、响度归一化、滤波和 EQ 设置。
6. 点击“预览”，在浏览器里试听当前混音。
7. 点击波形或时间线跳转播放位置；拖动 IN/OUT 手柄调整每首歌的入点和出点。
8. 打开“分轨调试”，选择任意已上传歌曲；未分轨完成时先听模拟分轨，完成后自动切到真实 Demucs stems。
9. 打开“教学入口”可查看推荐接法、操作步骤，并生成真实无缝试听；点击“使用这个接法”会把试听 cue 和过渡音频同步到时间线。
10. 在 Deck Mixer 里微调当前过渡两首歌的增益和三段 EQ。
11. 选择 MP3 或 WAV，点击“导出”。
12. 导出完成后点击下载链接保存混音文件。

## 功能说明

### 上传与分析

每个上传文件都会先在浏览器中解码，生成本地预览波形；同时上传到后端，由 `librosa` 完成更稳定的分析。后端返回：

- `duration`：歌曲时长
- `bpm`：估算 BPM
- `beats` / `bars` / `phrases`：节拍、小节和 phrase 时间点
- `key` / `camelot`：调性和 Camelot 编码
- `style` / `style_label` / `style_profile`：风格分类、显示名称和风格特征画像
- `energy` / `energy_profile`：多指标能量画像，包括 LUFS、RMS 分位数、crest factor、低频比例、动态范围和 intro/outro 相对能量
- `intro_low` / `outro_low`：首尾低能量时长
- `loudness_lufs` / `true_peak_db`：响度指标
- `transition_candidates`：推荐入点、出点、人声密度和置信度
- `peaks`：用于波形绘制的峰值数组

如果后端暂时不可用，前端会尝试用浏览器本地算法做 fallback 分析，但精度会低于后端。

### 分轨调试

分轨调试依赖可选的 Demucs 依赖。安装后：

```bash
pnpm setup:tuning
```

上传歌曲并完成基础分析后，前端会自动把歌曲加入 Demucs 分轨队列，依次调用后端 `/api/tracks/{track_id}/stems` 生成四个 stem：

- `vocals`：人声
- `drums`：鼓
- `bass`：贝斯/低频
- `other`：其他乐器与伴奏

生成结果会缓存在：

```text
backend/data/stems/{track_id}/demucs_api/
```

打开“分轨调试”时，选择哪首歌就会加载哪首歌的分轨状态：

- `Demucs 等待分轨` / `Demucs 分轨中`：先显示灰色模拟分轨波形，并用扫描光提示后台仍在处理。
- `Demucs 真分轨`：自动加载真实 stem 音频，波形切换为蓝色，M/S/音量控制直接作用在真实分轨上。
- 如果用户在模拟分轨播放中等待，真实分轨加载完成后会从当前进度自动切换到真实 stems。

如果未安装 Demucs，分轨调试仍可使用浏览器里的滤波模拟模式，但不会得到真正独立的人声、鼓、贝斯和其他声部。

### 排序策略

- 综合推荐：优先让相邻歌曲的 BPM 接近，同时考虑 Camelot 谐和与能量差。
- 谐和优先：提高 Camelot 匹配权重，适合更重视调性顺滑的歌单。
- BPM 升序：按速度从慢到快。
- BPM 降序：按速度从快到慢。
- 能量弧线：先逐步升能量，再在后段回落，适合活动暖场到高潮再收束。
- 原始顺序：保留上传顺序。

综合推荐和谐和优先都使用贪心最近邻算法：先从能量较低的歌开始，每一步选择与当前歌衔接成本最低的下一首。

### 两首歌匹配评分

Pair Match 面板可以单独上传两首歌，后端会分别分析它们，并计算：

- A 到 B 的衔接分
- B 到 A 的衔接分
- 推荐方向
- 总分等级
- Camelot、BPM、Energy、Structure 四个分项
- 可选 harmonic tuning 建议
- 可选自动修复方案：点击“自动修复匹配”后，后端会选择处理 A 或 B，生成调性、速度、能量处理计划，并输出修复后的 WAV/MP3。

评分满分为 100。总分权重为：

```text
total = 0.45 * Camelot
      + 0.30 * BPM
      + 0.15 * Energy
      + 0.10 * Structure
```

### DJ 教学与无缝试听

教学入口使用前端 TypeScript 业务模块把当前选中曲目作为 Deck A，并对候选曲目生成推荐：

- `recommendNextTracks()`：综合 Camelot、BPM、风格、能量、phrase、vocal conflict 和新手难度给候选排序。
- `explainTransition()`：把推荐接法解释成适合用户阅读的原因。
- `stepByStep`：输出 cue、loop、EQ、filter、crossfader 等操作步骤。
- `generateTeachingPreview()`：调用后端 `/api/transition-preview`，让后端按推荐 cue、节拍对齐、tempo/pitch 计划和 Demucs/Spleeter/full-mix fallback 生成一段真实过渡试听。

点击“使用这个接法”时，前端会把试听返回的实际 `outgoingCue`、`incomingCue`、`renderOverlapDuration` 和 `appliedTransitionPreview` 写回时间线。后续浏览器预览和后端导出都会优先复用这段已渲染的过渡音频，避免“教学试听”和“最终导出”听起来不一致。

### 过渡与预览

SmartMix 会为相邻歌曲计算一个过渡计划：

- 上一首从 `outro` 候选点开始淡出。
- 下一首从 `intro - overlap` 处开始进入，使其 intro 锚点对齐到过渡结束附近。
- 开启 AI 精准小节混音时，过渡时长会按 4/8/16 小节换算。
- 开启自动过渡时，会结合首尾低能量区缩短或限制过渡时长。
- 过短歌曲会把过渡时长限制在较短歌曲的 35% 以内。

浏览器预览使用 Web Audio API，后端导出使用 Python 音频管线；两边都复用相近的时间线和过渡策略。

### 混音策略

可选策略包括：

- AI 自动判断：根据人声密度、BPM 差和能量变化选择策略。
- 保留人声清晰：在人声较密的过渡里减少中频冲突。
- 低频交换切入：让旧歌低频更快退场，新歌低频更早建立。
- 平滑氛围过渡：更温和的等功率淡化和中高频进入。
- 快速切歌点：更短、更有 DJ 切换感的过渡。

滤波模式包括：

- AI 动态 EQ 避让
- 低通扫频
- 高通抬入
- 关闭滤波

### 导出

导出会把当前所有就绪曲目、排序、IN/OUT 点、Deck Mixer、全局 EQ、过渡参数和格式发送给后端。

支持：

- MP3：默认 192 kbps，用于分享和快速试听。
- WAV：PCM 16-bit，用于后期处理或保留更高质量。

可选导出处理：

- 节拍同步：把歌曲速度拉近到歌单 BPM 中位数，限制在 0.88x 到 1.12x。
- 响度归一化：默认目标 `-16 LUFS`。
- AI 精准过渡：使用小节长度、候选点和动态 EQ 生成重叠区。

导出文件保存在：

```text
backend/data/exports/
```

如果某个过渡已经在教学入口里生成并应用了无缝试听，导出时会把 `appliedTransitionPreview` 嵌入对应的交叠区；否则使用后端 `render_mix()` 的常规 crossfade / dynamic EQ 渲染路径。

### 项目保存与加载

项目会保存到后端本地 JSON 文件：

```text
backend/data/projects/
```

保存内容包括：

- 项目名称
- 曲目列表和曲目分析结果
- 当前排序
- IN/OUT 点
- 每首歌 mixer 设置
- 全局混音设置

加载项目时，前端会通过 `/api/tracks/{track_id}/audio` 重新取回后端保存的音频并恢复波形。

加载完成后，前端也会为每首 ready 曲目重新检查并排队加载 Demucs 分轨。如果 `backend/data/stems/{track_id}/demucs_api/` 已存在完整缓存，会直接读取缓存。

## 高质量 Camelot 调音

SmartMix 提供可选的调性转换管线，适合把某首歌调到更容易与另一首歌谐和衔接的 Camelot 编码。

安装可选依赖：

```bash
pnpm setup:tuning
```

如果想获得更好的调音质量，建议额外安装 Rubber Band CLI，并让 `rubberband-r3` 或 `rubberband` 在 `PATH` 中可用。

Windows 可选安装方式：

```bash
winget install BreakfastQuay.RubberBand
```

或：

```bash
choco install rubberband
```

CLI 示例：

```bash
python tune_quality.py "song.mp3" --source 9A --target 3A -o "song_3A.wav"
```

强制使用 CUDA Demucs 分轨：

```bash
python tune_quality.py "song.mp3" --source 9A --target 3A --device cuda -o "song_3A.wav"
```

API 示例：

```http
POST /api/tracks/{track_id}/tune
Content-Type: application/json

{
  "targetCamelot": "3A",
  "direction": "nearest",
  "format": "wav",
  "device": "auto"
}
```

调音管线会优先尝试：

```text
歌曲
  -> Demucs 分离 vocals / drums / bass / other
  -> vocals 用 Rubber Band R3 保留 formant 后移调
  -> bass 和 other 移调
  -> drums 尽量不移调
  -> 合成 stems
  -> 轻微 EQ 修饰
  -> 响度归一化
  -> 保存成新的 SmartMix 曲目
```

如果 Demucs 或 Rubber Band 不可用，会 fallback 到整首歌移调或 `librosa` 移调，质量会下降但功能仍可运行。

## 常用命令

```bash
pnpm dev              # 同时启动前端和后端
pnpm frontend         # 只启动 Vite 前端
pnpm backend          # 只启动 FastAPI 后端
pnpm setup:backend    # 安装后端基础依赖
pnpm setup:tuning     # 安装可选调音依赖
pnpm check            # 检查前端 JS 语法
pnpm typecheck        # TypeScript 类型检查
pnpm test             # 运行前端 Vitest 单元测试
pnpm test:backend     # 运行后端单元测试
```

## API 概览

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/health` | 后端健康检查 |
| `POST` | `/api/tracks` | 上传并分析单首歌 |
| `GET` | `/api/tracks/{track_id}/audio` | 获取已上传音频 |
| `POST` | `/api/tracks/{track_id}/stems` | 使用 Demucs 生成或读取缓存的真实分轨 |
| `GET` | `/api/tracks/{track_id}/stems/{stem_name}/audio` | 获取单个 stem 音频，`stem_name` 为 `vocals`、`drums`、`bass` 或 `other` |
| `POST` | `/api/tracks/{track_id}/tune` | 把曲目调到指定 Camelot |
| `POST` | `/api/match` | 计算两首歌衔接匹配分 |
| `POST` | `/api/match/repair` | 自动生成并渲染匹配修复版本 |
| `POST` | `/api/transition-preview` | 生成教学入口使用的真实无缝过渡试听 |
| `POST` | `/api/export` | 导出当前混音 |
| `GET` | `/api/exports/{filename}` | 下载导出文件 |
| `POST` | `/api/projects` | 保存项目 |
| `GET` | `/api/projects` | 列出已保存项目 |
| `GET` | `/api/projects/{project_id}` | 加载项目 |

## 项目结构

```text
SmartMix/
  index.html
  package.json
  src/
    main.js                 # 前端状态、UI、上传、预览、排序、教学、分轨调试、导出
    styles.css              # 工作台样式
    analysis/               # BPM、调性、能量、风格、人声/乐句评分模型
    transitions/            # fade、beatmix、bass swap、echo out 等推荐构建器
    seamless/               # cue 对齐、tempo/pitch 计划、stem automation
    audio/                  # crossfade、limiter、loudness、toolchain 类型
    explain/                # 推荐解释文案
    practice/               # DJ 练习计划
  backend/
    main.py                 # FastAPI 入口和 API 路由
    analysis.py             # librosa 音频分析、节拍/调性/能量/过渡候选
    matching.py             # 两歌匹配评分、Camelot 距离、调音建议
    transition.py           # 过渡计划计算
    mixing.py               # 后端混音、动态 EQ、节拍同步、MP3/WAV 导出
    seamless.py             # 真实无缝过渡试听、cue 修正、stem 级过渡渲染
    repair.py               # 两歌匹配自动修复计划与渲染
    loudness.py             # LUFS 测量和响度归一化
    tuning.py               # Camelot 调音、Demucs/Rubber Band/librosa fallback
    storage.py              # 本地上传、导出、项目 JSON 和 stems 缓存路径
    test_audio_engine.py    # 音频管线单元测试
    requirements.txt
    requirements-tuning.txt
  tune_quality.py           # 高质量调音 CLI
  tune_test.py              # 简化的 9A -> 3A 调音测试 CLI
  TECHNICAL_DESIGN.md       # 技术方案文档
```

运行后会生成本地数据目录：

```text
backend/data/
  uploads/                  # 上传音频和分析 JSON
  exports/                  # 导出的 MP3/WAV
  projects/                 # 保存的项目 JSON
  stems/                    # Demucs 生成的 vocals/drums/bass/other 分轨缓存
```

## 新手常见问题

### 页面提示后端未启动

确认已经运行：

```bash
pnpm dev
```

再打开：

```text
http://127.0.0.1:8002/api/health
```

如果返回 `{"ok": true}`，说明后端正常。

### Python 依赖安装失败

先确认 Python 版本：

```bash
python --version
```

建议使用 Python 3.11+。然后重新执行：

```bash
python -m pip install --upgrade pip
pnpm setup:backend
```

### MP3/WAV 导出失败

先运行后端测试确认基础音频管线可用：

```bash
pnpm test:backend
```

如果只 MP3 失败，重点检查 `imageio-ffmpeg` 是否安装成功：

```bash
python -m pip install imageio-ffmpeg
```

### 调音质量不好

高质量调音依赖 Demucs 和 Rubber Band。只安装基础依赖时，系统会用整首歌或 `librosa` fallback，效果会更像原型验证。建议安装：

```bash
pnpm setup:tuning
winget install BreakfastQuay.RubberBand
```

### 分轨调试一直是灰色模拟波形

真实分轨依赖 Demucs。先安装可选依赖：

```bash
pnpm setup:tuning
```

然后重启后端或重新运行：

```bash
pnpm dev
```

上传歌曲后，SmartMix 会在基础分析完成时自动排队跑 Demucs。长歌或 CPU 模式可能需要较久；等待期间分轨调试界面会显示灰色模拟波形和扫描态。完成后会自动加载真实 stems，波形变为蓝色。已生成的结果会缓存在 `backend/data/stems/`，下次选择同一首歌会直接复用缓存。

### 分析结果不准

BPM、Key 和 Camelot 都是算法估计，复杂现场录音、弱节奏、古典音乐、强变速歌曲可能不稳定。可以通过手动调整 IN/OUT 点、排序策略、Deck EQ 和过渡策略来修正听感。

## 技术文档

更详细的技术方案、模块职责、API、算法原理和实现细节见：

[TECHNICAL_DESIGN.md](./TECHNICAL_DESIGN.md)

## 双曲重组 / Mashup Builder

SmartMix 现在提供 “双曲重组 / Mashup Builder” 第一版，用于把两首已上传歌曲自动拆成 8/16 小节段落，并基于 BPM、Camelot、energy、vocal density、timbre 和 transition candidates 生成 mashup / re-edit 拼接方案。它不训练模型，使用规则系统复用现有 `backend/analysis.py` 的节拍、小节、乐句、能量、调性和过渡候选分析。

工作流：

1. 在前端上传并分析至少两首歌。
2. 在 “双曲重组 / Mashup Builder” 面板选择 Song A / Song B、mode、8/16 bars 和 useStems。
3. 点击 “分析段落” 查看两首歌的 segment blocks。
4. 点击 “生成拼接方案” 查看新的 plan 时间线和兼容性警告。
5. 点击 “渲染试听/导出” 生成新的 WAV，下载链接由 `/api/exports/{filename}` 提供。

支持模式：

- `smooth_join`：A 的 intro/verse/chorus 接 B 的兼容 chorus/outro。
- `hook_swap`：A verse、B chorus、A breakdown、B chorus/outro。
- `a_vocal_b_instrumental`：A vocals 叠加 B instrumental。
- `b_vocal_a_instrumental`：B vocals 叠加 A instrumental。
- `energy_build`：低能量到高能量排序组合，确保两首歌都出现。
- `auto`：从以上模板中选择评分最高的方案。

如果 Demucs stems 缓存可用，layered 模式会使用 `vocals` stem 与另一首歌的 `drums + bass + other` instrumental 组合；否则会降级到 full mix 并在 warnings 中提示。渲染阶段会做 BPM 对齐，time-stretch 限制在 0.88x 到 1.12x，并进行 LUFS normalize。Camelot 不强制移调，第一版只输出调性兼容性 warnings。

新增 API：

- `POST /api/mashup/analyze`
- `POST /api/mashup/plan`
- `POST /api/mashup/render`
