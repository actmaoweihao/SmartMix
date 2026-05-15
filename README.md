# SmartMix

SmartMix 是一个本地运行的智能混音工作台。它面向音乐爱好者、活动策划者和轻量 DJ 场景：上传多首歌曲后，系统会自动分析 BPM、调性、Camelot 编码、能量、响度、节拍网格和可过渡片段，帮助你生成更顺耳的播放顺序，预览两两重叠的混音效果，并导出 MP3 或 WAV。

项目由 Vite 前端和 FastAPI 后端组成，核心音频分析与导出在本地完成，不需要云端账号。

## 主要功能

- 多音频上传：支持常见音频格式，包括 MP3、WAV、FLAC、M4A、OGG、AAC、AIFF、OPUS、WEBM 等。
- 自动音频分析：识别时长、BPM、Key、Camelot、多指标能量、LUFS、真峰值、beat grid、bar、phrase、波形峰值和推荐过渡点。
- 智能排序：支持综合推荐、谐和优先、BPM 升序、BPM 降序、能量弧线和原始顺序。
- 两首歌衔接评分：上传任意两首歌，按 Camelot、BPM、能量和结构可过渡性计算 A 到 B / B 到 A 的匹配分。
- Harmonic tuning 建议：当两首歌调性不够兼容时，给出可调到的 Camelot 目标、半音数和质量风险。
- 实时混音预览：使用 Web Audio API 在浏览器里预览淡入淡出、EQ、滤波扫频和动态 EQ。
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
3. 等待每首歌分析完成，列表中会显示时长、BPM、调性、能量和过渡信息。
4. 在左侧选择排序策略，然后点击“应用排序”。
5. 调整过渡时长、AI 精准小节混音、响度归一化、滤波和 EQ 设置。
6. 点击“预览”，在浏览器里试听当前混音。
7. 点击波形或时间线跳转播放位置；拖动 IN/OUT 手柄调整每首歌的入点和出点。
8. 在 Deck Mixer 里微调当前过渡两首歌的增益和三段 EQ。
9. 选择 MP3 或 WAV，点击“导出”。
10. 导出完成后点击下载链接保存混音文件。

## 功能说明

### 上传与分析

每个上传文件都会先在浏览器中解码，生成本地预览波形；同时上传到后端，由 `librosa` 完成更稳定的分析。后端返回：

- `duration`：歌曲时长
- `bpm`：估算 BPM
- `beats` / `bars` / `phrases`：节拍、小节和 phrase 时间点
- `key` / `camelot`：调性和 Camelot 编码
- `energy` / `energy_profile`：多指标能量画像，包括 LUFS、RMS 分位数、crest factor、低频比例、动态范围和 intro/outro 相对能量
- `intro_low` / `outro_low`：首尾低能量时长
- `loudness_lufs` / `true_peak_db`：响度指标
- `transition_candidates`：推荐入点、出点、人声密度和置信度
- `peaks`：用于波形绘制的峰值数组

如果后端暂时不可用，前端会尝试用浏览器本地算法做 fallback 分析，但精度会低于后端。

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

评分满分为 100。总分权重为：

```text
total = 0.45 * Camelot
      + 0.30 * BPM
      + 0.15 * Energy
      + 0.10 * Structure
```

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
pnpm test:backend     # 运行后端单元测试
```

## API 概览

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/health` | 后端健康检查 |
| `POST` | `/api/tracks` | 上传并分析单首歌 |
| `GET` | `/api/tracks/{track_id}/audio` | 获取已上传音频 |
| `POST` | `/api/tracks/{track_id}/tune` | 把曲目调到指定 Camelot |
| `POST` | `/api/match` | 计算两首歌衔接匹配分 |
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
    main.js                 # 前端状态、UI、上传、预览、排序、波形、导出
    styles.css              # 工作台样式
  backend/
    main.py                 # FastAPI 入口和 API 路由
    analysis.py             # librosa 音频分析、节拍/调性/能量/过渡候选
    matching.py             # 两歌匹配评分、Camelot 距离、调音建议
    transition.py           # 过渡计划计算
    mixing.py               # 后端混音、动态 EQ、节拍同步、MP3/WAV 导出
    loudness.py             # LUFS 测量和响度归一化
    tuning.py               # Camelot 调音、Demucs/Rubber Band/librosa fallback
    storage.py              # 本地上传、导出、项目 JSON 存储
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

### 分析结果不准

BPM、Key 和 Camelot 都是算法估计，复杂现场录音、弱节奏、古典音乐、强变速歌曲可能不稳定。可以通过手动调整 IN/OUT 点、排序策略、Deck EQ 和过渡策略来修正听感。

## 技术文档

更详细的技术方案、模块职责、API、算法原理和实现细节见：

[TECHNICAL_DESIGN.md](./TECHNICAL_DESIGN.md)
