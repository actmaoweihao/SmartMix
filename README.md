# SmartMix

SmartMix 是一个本地运行的智能混音工作台。它把 Vite 前端、FastAPI 后端和 Python 音频处理管线组合在一起，帮助你上传歌曲、分析 BPM/调性/能量/响度/结构，生成更顺耳的排序和过渡方案，预览混音效果，并导出 MP3 或 WAV。

适合这些场景：

- 音乐爱好者快速把一组歌做成顺滑串烧。
- 活动策划、店铺播放、派对暖场等轻量 DJ 工作流。
- 学习 DJ 接歌：查看推荐接法、cue 点、EQ/滤波/交叉推子步骤，并生成真实无缝试听。
- 调试音频分轨、节拍同步、Harmonic Mixing 和两首歌 Mashup。

项目默认完全在本机运行，不需要云端账号。上传音频、分析结果、导出文件和项目存档都会保存在本地 `backend/data/` 目录。

## 功能概览

- 多音频上传：支持 MP3、WAV、FLAC、M4A、OGG、AAC、AIFF、OPUS、WEBM 等常见格式。
- 自动音频分析：识别时长、BPM、Key、Camelot、LUFS、true peak、能量画像、beat grid、bar、phrase、波形峰值和过渡候选点。
- 智能排序：综合推荐、谐和优先、BPM 升序、BPM 降序、能量弧线、原始顺序。
- 浏览器混音预览：基于 Web Audio API 预览淡入淡出、EQ、滤波扫频、动态 EQ 和 Deck Mixer 参数。
- 后端高质量导出：按当前排序、IN/OUT、过渡、EQ、响度归一化和节拍同步导出 MP3/WAV。
- Pair Match：上传任意两首歌，计算 A 到 B / B 到 A 的衔接评分，并给出调性、BPM、能量、结构分项。
- 自动匹配修复：自动选择处理 A 或 B，生成调性、速度、能量修复方案并渲染新音频。
- DJ 教学入口：基于当前 Deck A 推荐下一首歌、接歌方法、操作步骤、风险提示和真实无缝试听。
- Demucs 分轨调试：可选生成 vocals / drums / bass / other 四路真实 stems；未完成时提供浏览器模拟分轨占位。
- 双曲 Mashup Builder：分析两首歌的 8/16 小节段落，生成拼接/叠加方案，支持 stems、groove bed、人声优先级、能量曲线和替代方案。
- Reference Mix：以一首参考歌为目标风格，为分轨后的歌曲生成参考混音。
- Harmonic Tuning：可选把歌曲调到指定 Camelot，优先使用 Demucs + Rubber Band，失败时回退到整曲移调。
- 项目保存与加载：保存曲目、排序、设置、混音参数和过渡状态。

## 技术栈

- 前端：Vite、原生 JavaScript、TypeScript 业务模块、Web Audio API、Vitest。
- 后端：FastAPI、Uvicorn、Pydantic。
- 音频分析与渲染：librosa、soundfile、numpy、scipy、pyloudnorm、imageio-ffmpeg。
- 可选高质量音频能力：Demucs、Torch/TorchCodec、Rubber Band CLI。

## 项目结构

```text
SmartMix/
  src/                         # 前端入口、UI、音频预览、TS 业务模块和测试
    analysis/                  # BPM、Key、Energy、Phrase、Vocal 等前端分析辅助
    audio/                     # 浏览器音频处理、crossfade、limiter、loudness
    transitions/               # 接歌策略与推荐逻辑
    seamless/                  # cue 对齐、tempo/pitch/stem automation
    practice/                  # DJ 练习计划
    explain/                   # 接歌解释文案
    __tests__/                 # Vitest 单元测试
  backend/                     # FastAPI 后端和 Python 音频管线
    api/                       # tracks/projects 路由
    services/                  # 曲目上传、读取、分轨响应等服务
    analysis.py                # 后端音频分析
    mixing.py                  # 歌单导出渲染
    matching.py                # 两首歌匹配评分
    repair.py                  # 匹配修复
    seamless.py                # 真实无缝过渡试听
    mashup.py                  # Mashup 分析、方案、渲染
    tuning.py                  # Camelot 调音和 Demucs/Rubber Band 管线
    reference_mix.py           # Reference Mix
    data/                      # 本地运行时数据，自动创建
  auto_mix/                    # 独立 auto mix 实验模块
  docs/                        # 架构审查等补充文档
  ref/                         # 论文和第三方参考实现
  package.json                 # 前端、后端和测试命令
```

运行时数据目录：

```text
backend/data/uploads/          # 上传音频和分析 JSON
backend/data/exports/          # 导出的 MP3/WAV/报告
backend/data/projects/         # 保存的项目 JSON
backend/data/stems/            # Demucs 分轨缓存
```

## 快速开始

### 1. 准备环境

建议版本：

- Node.js 20+
- pnpm
- Python 3.11+

如果没有 pnpm：

```bash
npm install -g pnpm
```

MP3 导出默认使用 `imageio-ffmpeg` 自带的 ffmpeg，一般不需要额外安装系统 ffmpeg。

### 2. 安装依赖

```bash
pnpm install
pnpm setup:backend
```

### 3. 启动开发服务

```bash
pnpm dev
```

启动后访问：

- 前端工作台：http://127.0.0.1:3000
- 后端健康检查：http://127.0.0.1:8002/api/health

`pnpm dev` 会同时启动：

- `pnpm frontend`：Vite，固定监听 `127.0.0.1:3000`
- `pnpm backend`：FastAPI/Uvicorn，固定监听 `127.0.0.1:8002`

也可以单独启动：

```bash
pnpm frontend
pnpm backend
```

## 可选高质量依赖

基础功能只需要 `pnpm setup:backend`。如果要使用 Demucs 真分轨、Harmonic Tuning、Reference Mix 或更高质量的 Mashup，安装可选依赖：

```bash
pnpm setup:tuning
```

NVIDIA GPU 用户可以先按 PyTorch 官方方式安装 CUDA 版 Torch，例如：

```bash
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pnpm setup:tuning
```

Rubber Band 不是 Python 包，需要单独安装 CLI，并确保 `rubberband-r3` 或 `rubberband` 在 `PATH` 中。

Windows 可选安装方式：

```bash
winget install BreakfastQuay.RubberBand
```

或：

```bash
choco install rubberband
```

没有安装 Demucs 或 Rubber Band 时，相关功能会降级或返回明确错误；基础上传、分析、排序、预览、导出仍可运行。

## 基础使用流程

1. 打开 http://127.0.0.1:3000。
2. 点击“选择音频”，或把音频文件拖入上传区域。
3. 等待歌曲分析完成，曲目会显示 BPM、调性、Camelot、能量、风格、响度和过渡候选点。
4. 选择排序策略，点击“应用排序”。
5. 调整过渡时长、AI 精准小节混音、响度归一化、滤波、EQ 和 Deck Mixer。
6. 点击“预览”在浏览器试听。
7. 拖动波形上的 IN/OUT 手柄，微调每首歌的入点和出点。
8. 需要教学时打开“教学入口”，选择推荐接法并生成真实无缝试听。
9. 需要 stems 时打开“分轨调试”，等待 Demucs 真分轨完成。
10. 选择 MP3 或 WAV，点击“导出”，完成后下载结果。
11. 点击“保存项目”可把当前曲目、排序和设置保存到本地。

## 核心工作流

### 上传与分析

前端会先在浏览器中解码音频，生成本地波形和预览；同时上传到后端 `/api/tracks`，由 `librosa` 完成更稳定的分析。后端会返回：

- `duration`：歌曲时长
- `bpm`：估算 BPM
- `beats` / `bars` / `phrases`：节拍、小节和 phrase 时间点
- `key` / `camelot`：调性和 Camelot 编码
- `style` / `style_label` / `style_profile`：风格分类和特征画像
- `energy` / `energy_profile`：多指标能量画像
- `loudness_lufs` / `true_peak_db`：响度和真峰值
- `intro_low` / `outro_low`：首尾低能量时长
- `transition_candidates`：推荐入点、出点、人声密度和置信度
- `peaks`：前端波形绘制数据

如果后端不可用，前端会提示连接错误；请确认 `pnpm backend` 或 `pnpm dev` 正在运行。

### 排序与过渡

SmartMix 会为相邻歌曲计算衔接成本。综合推荐会同时考虑 BPM、Camelot、能量和结构；谐和优先会提高 Camelot 权重；能量弧线会让歌单先升能量再回落。

过渡计划会结合：

- 上一首的 outro 候选点
- 下一首的 intro 候选点
- 4/8/16 小节长度
- BPM 差和节拍同步设置
- 人声密度和 vocal conflict
- 动态 EQ、低通/高通滤波和等功率淡化

### DJ 教学入口

教学入口会把当前选中曲目作为 Deck A，对其他候选曲目生成推荐：

- `recommendNextTracks()`：推荐下一首歌。
- `explainTransition()`：解释推荐原因、难度和风险。
- `stepByStep`：输出 cue、loop、EQ、filter、crossfader 操作步骤。
- `/api/transition-preview`：后端渲染真实无缝过渡试听。

点击“使用这个接法”后，前端会把返回的 cue 点、重叠时长和已渲染过渡音频写回时间线。后续浏览器预览和最终导出会优先复用这段过渡，避免教学试听和导出结果不一致。

### Demucs 分轨调试

团队 SDK 与外部 API 调用说明见 [`docs/guides/DEMUCS_STEM_SDK.md`](docs/guides/DEMUCS_STEM_SDK.md)。

安装 `pnpm setup:tuning` 后，上传歌曲并完成基础分析时，前端可以排队调用：

```http
POST /api/tracks/{track_id}/stems
```

生成并缓存：

- `vocals`
- `drums`
- `bass`
- `other`

缓存位置：

```text
backend/data/stems/{track_id}/demucs_api/
```

如果缓存已存在，默认直接复用；请求参数 `force: true` 可强制重新生成。

### Pair Match 与自动修复

Pair Match 用于单独比较两首歌：

- 计算 A 到 B 和 B 到 A 的衔接分。
- 分项展示 Camelot、BPM、Energy、Structure。
- 给出推荐方向和 harmonic tuning 建议。
- 可调用自动修复，把其中一首歌处理成更适合衔接的版本。

评分公式：

```text
total = 0.45 * Camelot
      + 0.30 * BPM
      + 0.15 * Energy
      + 0.10 * Structure
```

### Mashup Builder

Mashup Builder 用于两首歌的段落级重组。典型流程：

1. 上传并分析至少两首歌。
2. 在 Mashup 面板选择 Track A 和 Track B。
3. 选择 8 或 16 小节作为段落粒度。
4. 点击“分析段落”，查看每首歌的可用片段、能量、风险和结构标签。
5. 点击“生成拼接方案”，生成主方案和替代方案。
6. 点击“渲染试听/导出”，后端生成可试听文件。

可调参数包括：

- `mode`：自动、交替、清唱叠加等策略。
- `useStems`：是否优先使用 Demucs stems。
- `transitionStrictness`：过渡严格度。
- `stemUsage`：stem 使用偏好。
- `vocalPriority`：人声优先级。
- `energyCurve`：能量曲线。
- `bedPreference`：伴奏 bed 偏好。
- `allowHybridBed`：是否允许混合 bed。
- `allowVocalPitchShift`：是否允许人声移调。
- `maxVocalStretch`：最大人声拉伸比例。

### Harmonic Tuning

把某首歌调到指定 Camelot：

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

离线脚本示例：

```bash
python tune_quality.py "song.mp3" --source 9A --target 3A -o "song_3A.wav"
python tune_quality.py "song.mp3" --source 9A --target 3A --device cuda -o "song_3A.wav"
```

优先管线：

```text
原曲
  -> Demucs 分离 vocals / drums / bass / other
  -> vocals 使用 Rubber Band R3 保留 formant 后移调
  -> bass 和 other 移调
  -> drums 尽量不移调
  -> 合成 stems
  -> EQ 修饰与响度归一化
  -> 保存为新的 SmartMix 曲目
```

## 常用命令

```bash
pnpm dev              # 同时启动前端和后端
pnpm frontend         # 只启动 Vite 前端
pnpm backend          # 只启动 FastAPI 后端
pnpm setup:backend    # 安装后端基础依赖
pnpm setup:tuning     # 安装可选 Demucs/TorchCodec 依赖
pnpm check            # 检查 src/main.js 语法
pnpm typecheck        # TypeScript 类型检查
```

## API 概览

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/health` | 后端健康检查 |
| `POST` | `/api/tracks` | 上传并分析单首歌 |
| `GET` | `/api/tracks/{track_id}/audio` | 获取已上传音频 |
| `POST` | `/api/tracks/{track_id}/stems` | 生成或读取 Demucs stems |
| `GET` | `/api/tracks/{track_id}/stems/{stem_name}/audio` | 获取单个 stem 音频 |
| `POST` | `/api/tracks/{track_id}/reference-mix` | 按参考曲目渲染参考混音 |
| `POST` | `/api/tracks/{track_id}/tune` | 把曲目调到指定 Camelot |
| `POST` | `/api/match` | 计算两首歌衔接评分 |
| `POST` | `/api/match/repair` | 自动修复两首歌匹配 |
| `POST` | `/api/transition-preview` | 生成真实无缝过渡试听 |
| `POST` | `/api/mashup/analyze` | 分析两首歌的 Mashup 段落 |
| `POST` | `/api/mashup/plan` | 生成 Mashup 拼接/叠加方案 |
| `POST` | `/api/mashup/render` | 渲染 Mashup 方案 |
| `POST` | `/api/export` | 导出当前歌单混音 |
| `GET` | `/api/exports/{filename}` | 下载导出文件 |
| `POST` | `/api/projects` | 保存项目 |
| `GET` | `/api/projects` | 列出已保存项目 |
| `GET` | `/api/projects/{project_id}` | 加载项目 |

## 开发说明

### 前端

- 入口文件是 `src/main.js`，负责界面渲染、状态管理、事件绑定和 API 调用。
- API 基础地址在 `src/api/client.js`，前端会按当前页面 hostname 拼出 `http://{host}:8002`。
- TypeScript 模块集中在 `src/analysis/`、`src/transitions/`、`src/seamless/`、`src/audio/` 等目录。

### 后端

- 入口是 `backend/main.py`。
- `/api/tracks` 和 `/api/projects` 分别拆在 `backend/api/tracks.py`、`backend/api/projects.py`。
- 音频分析、导出、匹配、修复、调音、Mashup 都是独立 Python 模块，便于单测和替换。
- 后端启动时会自动创建 `backend/data/` 下的运行时目录。

### 数据和隐私

SmartMix 不会把音频上传到外部服务。运行时生成的数据都在本地：

- 删除 `backend/data/uploads/` 可清理上传曲目和分析结果。
- 删除 `backend/data/exports/` 可清理导出文件。
- 删除 `backend/data/stems/` 可清理 Demucs 缓存。
- 删除 `backend/data/projects/` 可清理保存的项目。

## 验证

建议在提交前运行：

```bash
pnpm check
pnpm typecheck
```

测试文件已从当前工程中移除，日常验证以启动项目和运行静态检查为主。

## 常见问题

### 前端提示无法连接后端

确认后端已启动：

```bash
pnpm backend
```

然后打开：

```text
http://127.0.0.1:8002/api/health
```

如果健康检查不可访问，优先检查 Python 依赖是否已安装：

```bash
pnpm setup:backend
```

### 端口被占用

默认前端使用 `3000`，后端使用 `8002`。如果端口被占用，可以先停止占用进程，或临时修改 `package.json` 中的 `frontend` / `backend` 脚本和 `src/api/client.js` 中的 API 端口。

### MP3/WAV 导出失败

先确认后端日志中的具体错误。常见原因包括：

- 某些上传音频解码失败。
- Python 音频依赖未安装完整。
- 导出目录没有写入权限。
- 曲目数据已经被删除但项目仍引用旧 track id。

可先重新运行：

```bash
pnpm setup:backend
```

### Demucs 分轨不可用

如果 `/api/tracks/{track_id}/stems` 返回 `Demucs is not available`，说明可选依赖还没安装：

```bash
pnpm setup:tuning
```

如果安装后仍失败，检查 Torch、torchaudio、TorchCodec 和本机 Python 环境是否一致。GPU 用户还需要确认 CUDA 版 Torch 与显卡驱动匹配。

### Rubber Band 不可用

Harmonic Tuning 会优先找 `rubberband-r3` 或 `rubberband`。如果找不到，调音质量会降级。请安装 Rubber Band CLI，并确认命令在终端中可直接运行：

```bash
rubberband-r3 --help
```

或：

```bash
rubberband --help
```

### 中文显示乱码

README 和后端 JSON 都使用 UTF-8。Windows PowerShell 旧环境如果显示乱码，可以尝试：

```powershell
chcp 65001
Get-Content README.md -Encoding utf8
```

## 相关文档

- `docs/product/SMARTMIX_PRODUCT_SPEC.md`：产品规格。
- `docs/product/MASHUP_UX_FLOW.md`：Mashup Builder 使用路径。
- `docs/architecture/TECHNICAL_DESIGN.md`：技术设计。
- `docs/architecture/ARCHITECTURE_REVIEW.md`：架构审查。
- `docs/algorithms/SORTING_ALGORITHM.md`：排序算法说明。
- `docs/algorithms/SONG_MATCHING_SCORING.md`：两首歌匹配评分。
- `docs/audio/ENERGY_SCORING.md`：能量评分。
- `docs/guides/DEMUCS_STEM_SDK.md`：Demucs 分轨 SDK 与接口调用。
- `docs/integrations/DIFF_MST_INTEGRATION.md`：Diff-MST 集成记录。
- `docs/tools/AUTO_MIX_README.md`：独立 auto mix 模块说明。
- `docs/reference/DIFF_MST_REFERENCE_README.md`：Diff-MST 参考项目说明。

## 当前状态

SmartMix 仍是本地优先的开发版本。核心上传、分析、排序、预览、导出、项目保存、Pair Match、教学试听、分轨调试、Mashup 和调音管线已经具备可运行路径；高质量 stems、GPU 加速和 Rubber Band 调音依赖本机环境，首次运行可能需要较长下载和处理时间。

Mashup Builder 的推荐使用路径见 [docs/product/MASHUP_UX_FLOW.md](docs/product/MASHUP_UX_FLOW.md)。
