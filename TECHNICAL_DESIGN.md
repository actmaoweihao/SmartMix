# SmartMix 技术方案文档

本文档面向开发者，说明 SmartMix 的整体架构、核心模块、主要 API、数据结构，以及音频分析、排序、两歌匹配、过渡规划、实时预览、后端导出和 Camelot 调音的实现原理。

## 1. 技术栈

### 前端

- Vite：本地开发服务器与前端构建工具。
- 原生 JavaScript：项目没有引入 React/Vue，主要逻辑集中在 `src/main.js`。
- Web Audio API：负责浏览器端音频解码、实时预览、GainNode 淡入淡出、BiquadFilter EQ 和滤波扫频。
- Canvas：负责选中曲目的波形绘制和 IN/OUT 手柄编辑。
- CSS：工作台布局、控制面板、时间线、Deck Mixer 和表格样式。

### 后端

- FastAPI：提供上传、分析、匹配、导出、项目保存和调音 API。
- librosa：音频解码、BPM、beat tracking、chroma、time stretch、pitch shift fallback。
- NumPy / SciPy：音频数组处理、滤波、包络和动态 EQ。
- soundfile：写 WAV。
- imageio-ffmpeg：提供 ffmpeg 二进制，用于格式 fallback 解码和 MP3 转码。
- pyloudnorm：可用时用于 LUFS 测量和响度归一化。
- Demucs：可选，用于高质量调音的 stem separation。
- Rubber Band：可选，用于高质量 pitch shifting。

## 2. 运行架构

```text
Browser
  |
  |  HTTP / JSON / multipart form
  v
FastAPI backend
  |
  |  local filesystem
  v
backend/data/
  uploads/   uploaded audio + analysis json
  exports/   rendered mix files
  projects/  saved project json
```

前端默认访问 `http://当前主机:8001`，后端 CORS 允许本地开发端口，包括 `3000` 和 `5173`。

启动命令来自 `package.json`：

```text
pnpm dev      -> concurrently 启动 backend 和 frontend
pnpm frontend -> vite --host 127.0.0.1 --port 3000
pnpm backend  -> uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
```

## 3. 目录与模块职责

```text
src/main.js
```

前端主模块，负责：

- 应用状态 `state`
- UI 模板和事件绑定
- 文件上传、拖拽上传和浏览器解码
- 本地 fallback 分析
- 曲目排序
- Web Audio 实时预览
- 时间线构建
- 波形绘制和 IN/OUT 编辑
- Deck Mixer
- 项目保存/加载
- 调用后端匹配、导出等 API

```text
backend/main.py
```

FastAPI 路由入口，负责：

- CORS 配置
- 上传文件校验
- 调用分析、匹配、导出、调音模块
- 项目 JSON 保存与读取
- 返回音频文件和导出文件

```text
backend/analysis.py
```

音频分析模块，负责：

- 解码为 44.1 kHz 单声道
- BPM 和 beat grid
- downbeat offset 粗估
- bar / phrase 时间线
- chroma 调性识别
- Camelot 映射
- 能量、intro/outro 低能量区
- LUFS 和 true peak
- 过渡候选点
- 波形 peaks

```text
backend/transition.py
```

过渡计划模块，负责把两首曲目的候选点、BPM、时长和设置合成为一个 `TransitionPlan`。

```text
backend/matching.py
```

两歌匹配评分模块，负责：

- Key 到 Camelot 的映射
- Camelot 距离计算
- BPM 匹配评分
- 能量匹配评分
- 结构可过渡性评分
- A -> B 和 B -> A 双向评分
- 调音建议

```text
backend/mixing.py
```

后端导出模块，负责：

- 读取上传音频为 stereo buffer
- 可选节拍同步
- 单曲 mixer 和全局 EQ
- 过渡重叠渲染
- 动态 EQ
- 响度归一化
- WAV 写出和 MP3 转码

```text
backend/tuning.py
```

高质量 Camelot 调音模块，负责：

- Camelot 到半音差计算
- 谐和目标推荐
- Demucs 分轨
- Rubber Band 或 librosa 移调
- stems 合成
- master polish
- 调音后重新分析并注册为新曲目

```text
backend/loudness.py
```

响度模块，优先使用 `pyloudnorm`，失败时退化为 RMS LUFS 近似。

```text
backend/storage.py
```

本地文件和 JSON 存储模块。

## 4. 核心数据结构

### 4.1 前端 Track

前端曲目对象由上传时创建，分析完成后合并后端返回字段。关键字段：

```js
{
  id,                 // 后端 track id
  localId,            // 前端本地 id，用于 UI 操作
  file,               // 原始 File，加载项目时为 null
  name,
  status,             // uploading | ready | error
  buffer,             // AudioBuffer，用于浏览器预览
  peaks,              // 波形峰值
  duration,
  bpm,
  key,
  camelot,
  key_index,
  mode,
  energy,
  intro_low,
  outro_low,
  transition_candidates,
  introPoint,         // 用户可编辑入点
  outroPoint,         // 用户可编辑出点
  mixer: {
    gain,
    eq: { low, mid, high }
  }
}
```

### 4.2 后端分析结果

`POST /api/tracks` 返回的核心字段：

```json
{
  "id": "track id",
  "name": "song.mp3",
  "path": "backend/data/uploads/xxx.mp3",
  "duration": 180.0,
  "bpm": 128,
  "beats": [0.51, 0.98],
  "bars": [0.51, 2.38],
  "phrases": [0.51, 8.0],
  "downbeat_offset": 0,
  "beat_confidence": 0.72,
  "key": "E Min",
  "camelot": "9A",
  "key_index": 4,
  "mode": "minor",
  "energy": 0.63,
  "intro_low": 7.2,
  "outro_low": 12.5,
  "loudness_lufs": -14.8,
  "true_peak_db": -1.2,
  "transition_candidates": {
    "intro": 16.0,
    "outro": 146.5,
    "confidence": 0.81,
    "intro_vocal_density": 0.21,
    "outro_vocal_density": 0.18,
    "method": "bar-vocal-energy"
  },
  "peaks": [0.1, 0.5, 0.3]
}
```

### 4.3 全局 Settings

前端默认设置位于 `state.settings`：

```js
{
  sortMode: "recommended",
  crossfade: 8,
  autoTransition: true,
  beatSync: false,
  aiPrecision: true,
  phraseBars: 8,
  loudnessNormalize: true,
  targetLufs: -16,
  equalPowerFade: true,
  mixStrategy: "auto",
  filterMode: "lowpassSweep",
  exportFormat: "mp3",
  eq: { low: 0, mid: 0, high: 0 }
}
```

## 5. API 设计

### 5.1 健康检查

```http
GET /api/health
```

返回：

```json
{"ok": true}
```

### 5.2 上传并分析曲目

```http
POST /api/tracks
Content-Type: multipart/form-data

file=<audio>
```

流程：

1. 校验 MIME 或后缀是否为音频。
2. 写入 `backend/data/uploads/{track_id}{suffix}`。
3. 调用 `analyze_audio(path)`。
4. 写入 `backend/data/uploads/{track_id}.json`。
5. 返回曲目元数据和分析结果。

### 5.3 获取曲目音频

```http
GET /api/tracks/{track_id}/audio
```

用于加载项目时恢复浏览器 `AudioBuffer`。

### 5.4 两歌匹配

```http
POST /api/match
Content-Type: multipart/form-data

file_a=<audio>
file_b=<audio>
```

后端会分别保存并分析两首歌，然后返回：

- `track_a` / `track_b` 摘要
- `overall_score`
- `overall_level`
- `recommended_direction`
- `directions.a_to_b`
- `directions.b_to_a`
- `tuning_recommendations`

### 5.5 导出混音

```http
POST /api/export
Content-Type: application/json

{
  "trackIds": ["id1", "id2"],
  "tracks": [/* 前端当前曲目状态 */],
  "settings": {/* 全局设置 */},
  "format": "mp3"
}
```

后端读取每个 track id 对应的上传文件，再合并前端传来的用户编辑字段，调用 `render_mix()` 导出文件。

### 5.6 下载导出文件

```http
GET /api/exports/{filename}
```

### 5.7 项目保存和加载

```http
POST /api/projects
GET /api/projects
GET /api/projects/{project_id}
```

项目以 JSON 存储，不包含音频二进制，只引用已经保存在 `uploads/` 中的曲目 id。

### 5.8 曲目调音

```http
POST /api/tracks/{track_id}/tune
Content-Type: application/json

{
  "targetCamelot": "3A",
  "sourceCamelot": "9A",
  "direction": "nearest",
  "format": "wav",
  "device": "auto"
}
```

返回一个新的曲目元数据，并附带 `tuning` 信息。该新曲目会写入 uploads 元数据目录，可以像普通曲目一样加载和导出。

## 6. 音频分析实现

入口为：

```python
analyze_audio(path: Path) -> dict
```

### 6.1 解码

`_load_audio()` 优先使用：

```python
librosa.load(path, sr=44100, mono=True)
```

如果失败，则用 `imageio_ffmpeg.get_ffmpeg_exe()` 把源文件转成临时 WAV，再交给 librosa 加载。这样可以覆盖部分 librosa 直接解码失败的格式。

### 6.2 BPM 和 beat grid

`_beat_grid()` 使用：

```python
onset = librosa.onset.onset_strength(y=y, sr=sr)
tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, units="frames")
```

然后把 beat frame 转成秒：

```python
beat_times = librosa.frames_to_time(beats, sr=sr)
```

如果 beat 数足够，会粗估 downbeat offset：

1. 计算 RMS。
2. 取每个 beat 上的能量。
3. 遍历 0、1、2、3 四种小节偏移。
4. 假设 downbeat 的平均能量应略高于其他拍。
5. 选择分数最高的 offset。

小节和 phrase：

```text
bars = beat_times[offset::4]
phrases = bars[::4]
```

也就是默认 4 拍一小节，4 小节一组 phrase。

如果 beat tracking 不可靠，会 fallback 到 `_estimate_bpm_from_envelope()`，用 RMS flux 在 70 到 180 BPM 范围内做自相关搜索。

### 6.3 调性和 Camelot

`_estimate_key()` 使用 `librosa.feature.chroma_cqt()` 计算 chroma，再与 Krumhansl-Schmuckler 大调/小调模板做匹配。

对每个 root：

```python
major_score = dot(roll(MAJOR_PROFILE, root), chroma_profile)
minor_score = dot(roll(MINOR_PROFILE, root), chroma_profile)
```

取得分最高的 root 和 mode，输出如：

```text
E Min
C Maj
```

随后由 `matching.key_label_to_camelot()` 映射到 Camelot：

```text
E Min -> 9A
C Maj -> 8B
```

### 6.4 能量、intro/outro 低能量区

`_energy_metrics()` 计算 RMS：

```python
rms = librosa.feature.rms(...)
avg = mean(rms)
peak = max(rms)
threshold = max(avg * 0.55, peak * 0.08)
```

整体能量：

```text
energy = min(1.0, avg * 7 + peak * 1.5)
```

intro/outro 低能量区：

- 只看歌曲前后 25% 的 RMS frame。
- 从开头或结尾向内扫描，直到 RMS 超过阈值。
- 连续低能量 frame 数换算为秒。

### 6.5 响度和峰值

`loudness_metrics()` 优先调用 `pyloudnorm.Meter.integrated_loudness()`。如果依赖不可用或计算失败，fallback 到 RMS dB 近似。

真峰值当前是 sample peak 近似：

```text
peak_db = 20 * log10(max(abs(audio)))
```

### 6.6 过渡候选点

`_transition_candidates()` 根据 bars、能量和人声密度估计适合进入/退出的位置。

如果没有 bar 信息：

- intro 取 `intro_low` 和 4 秒之间的合理值。
- outro 取 `duration - outro_low` 附近。
- 方法标记为 `energy-fallback`。

如果有 bar 信息：

1. `_bar_features()` 为每个小节计算：
   - 小节平均能量
   - vocal density
   - 位置比例
2. vocal density 来自 `_vocal_density_curve()`：
   - 先用 HPSS 取 harmonic 分量。
   - 做 STFT。
   - 计算 300 Hz 到 3400 Hz 频段能量占比。
   - 把这个占比视为人声/主体中频密度近似。
3. intro 候选：
   - 在 `intro_floor` 到歌曲 45% 以内选择。
   - 偏好人声密度低、能量低、位置靠前的 bar。
4. outro 候选：
   - 在歌曲后半段选择。
   - 偏好人声密度低、能量低、位置靠后的 bar。
5. 根据候选数量、人声密度和分析结果计算置信度。

### 6.7 波形 peaks

`_waveform_peaks(y, 720)` 把整首歌分成固定数量的 chunk，取每段绝对峰值并归一化，前端直接用这些值绘制 Canvas 波形。

## 7. 排序算法

排序在前端实现，入口为：

```js
sortTracks(tracks, mode)
```

### 7.1 原始、BPM、能量弧线

- `original`：按 `state.originalOrder` 恢复上传顺序。
- `bpmAsc` / `bpmDesc`：按 `safeBpm(track)` 排序，缺省 BPM 为 120。
- `energyArc`：先按能量升序，再把后 35% 高能量曲目反向接到结尾，形成升高后回落的弧线。

### 7.2 贪心最近邻

综合推荐和谐和优先使用：

```js
greedySort(tracks, mode)
```

流程：

1. 复制曲目列表，按能量升序。
2. 取能量最低的曲目作为起点。
3. 对剩余曲目逐一计算与当前曲目的 `transitionScore()`。
4. 选分数最低的作为下一首。
5. 重复直到全部排完。

### 7.3 综合推荐评分

```js
score = 0.55 * bpmDelta
      + 0.30 * harmonic
      + 0.15 * energyDelta
```

其中：

```text
bpmDelta = min(1, abs(BPM_A - BPM_B) / 60)
energyDelta = abs(energy_A - energy_B)
harmonic = min(1, camelotDistance(A, B) / 6)
```

分数越低，说明越适合相邻播放。

### 7.4 谐和优先评分

```js
score = 0.60 * harmonic
      + 0.30 * bpmDelta
      + 0.10 * energyDelta
```

它更重视 Camelot 兼容性，适合调性连续比速度连续更重要的场景。

### 7.5 Camelot 距离

`camelotDistance(codeA, codeB)` 规则：

- 同 Camelot 数字：距离 0，认为大小调关系兼容。
- 同 mode：按 12 格轮盘数字距离。
- 不同 mode 且数字不同：数字距离 + 2 惩罚。
- 纯五度关系：距离 - 1 奖励。
- 数字距离大于等于 6：额外 + 1 惩罚。

## 8. 两歌匹配评分

后端入口：

```python
evaluate_track_match(track_a, track_b)
```

系统分别计算：

```text
A -> B
B -> A
```

方向性主要来自结构评分，因为 A 的 outro 和 B 的 intro 是否合适，与反方向不同。

### 8.1 总分

```text
total_score = 0.45 * camelot_score
            + 0.30 * bpm_score
            + 0.15 * energy_score
            + 0.10 * structure_score
```

等级：

```text
>= 90  完美
>= 75  推荐
>= 60  可用
<  60  避坑
```

### 8.2 Camelot 分项

先计算 Camelot 距离，再映射为分数：

```text
distance 0 -> 100
distance 1 -> 92
distance 2 -> 84
distance 3 -> 72
distance 4 -> 62
distance 5+ -> 继续递减
```

### 8.3 BPM 分项

BPM 评分会自动考虑 half/double tempo：

```python
candidates = [bpm_b, bpm_b * 2, bpm_b / 2]
normalized_bpm_b = candidate closest to bpm_a
pct = abs(bpm_a - normalized_bpm_b) / bpm_a
bpm_score = 100 - pct * 420
```

最后限制到 0 到 100。

### 8.4 Energy 分项

```text
energy_score = 100 - abs(energy_A - energy_B) * 120
```

### 8.5 Structure 分项

先调用 `recommend_transition()`，也就是基于 `plan_transition()` 的 16 小节精准过渡建议。根据可承载的重叠时长给分：

```text
>=24s  96
>=16s  88
>=8s   76
>=4s   58
<4s    35
```

## 9. 过渡规划

核心函数：

```python
plan_transition(prev_track, next_track, settings) -> TransitionPlan
```

前端有对应的 `planClientTransition(prev, next)`，用于实时预览和时间线展示。

### 9.1 输入

关键输入：

- `crossfade`：用户请求的过渡秒数。
- `autoTransition`：是否根据首尾低能量区自动限制。
- `aiPrecision`：是否优先使用分析候选点和小节长度。
- `phraseBars`：希望重叠的 phrase 小节数。
- `prev_track.duration`
- `next_track.duration`
- `prev_track.transition_candidates.outro`
- `next_track.transition_candidates.intro`
- `prev_track.outro_low`
- `next_track.intro_low`
- `bpm`

### 9.2 最大过渡限制

为了避免短歌被过渡吞掉：

```text
max_by_length = min(prev_duration, next_duration) * 0.35
```

实际过渡不会超过它。

### 9.3 AI 精准小节过渡

开启 `aiPrecision` 时：

```text
phrase_seconds = phraseBars * 4 * (60 / avg_bpm)
```

例如平均 120 BPM、8 bars：

```text
8 * 4 * 0.5 = 16 秒
```

这让重叠区更接近音乐小节结构。

### 9.4 出入点对齐

后端计划：

```text
prev_overlap_start = prev_outro
next_overlap_start = next_intro - seconds
```

含义：

- 上一首从其 outro 候选点开始淡出。
- 下一首从 intro 前面提前 `seconds` 秒进入。
- 过渡结束时，下一首正好到达 intro 锚点附近。

### 9.5 自动过渡限制

开启 `autoTransition` 时：

```text
structural = prev.outro_low + next.intro_low + 2
requested = max(2, min(requested, structural))
```

这样可以避免在首尾低能量区很短时强行做过长重叠。

### 9.6 置信度

`_plan_confidence()` 会综合：

- 是否能容纳 4/8/16 小节重叠
- 可用重叠时长是否足够
- 上下曲候选点置信度
- intro/outro 人声密度是否较低

输出 0 到 0.95。

## 10. 实时预览实现

前端入口：

```js
previewMix(offset = 0)
```

流程：

1. `buildTimeline()` 根据当前曲目和过渡计划生成整段 mix timeline。
2. `scheduleMix()` 遍历 timeline item。
3. 对每首歌创建：
   - `AudioBufferSourceNode`
   - `GainNode`
   - Low shelf `BiquadFilterNode`
   - Mid peaking `BiquadFilterNode`
   - High shelf `BiquadFilterNode`
   - transition filter
4. `applyPreviewEnvelope()` 对 GainNode 和 filter 自动化。
5. `tickPlayback()` 每 80 ms 更新播放位置、进度条和波形 playhead。

### 10.1 预览 EQ

每首歌的 EQ 来自：

```text
track.mixer.eq + global settings.eq
```

映射到 Web Audio：

- Low：`lowshelf`，220 Hz
- Mid：`peaking`，1200 Hz，Q=0.9
- High：`highshelf`，3400 Hz

### 10.2 预览淡入淡出

上一首在 `fadeOutStart -> fadeOutEnd` 线性降低 gain。下一首在自己的 `fadeIn` 时间内从当前 offset 淡入。

### 10.3 预览滤波

- `lowpassSweep`：上一首淡出时从 16 kHz 扫到 900 Hz。
- `highpassLift`：下一首淡入时从 700 Hz 逐步降到 35 Hz。
- `dynamicEq`：不主要扫 filter，而是自动化三段 EQ 的 gain。

### 10.4 动态 EQ 策略

`resolveMixStrategy()` 自动模式：

```text
if outgoing/incoming vocal density > 0.55 -> vocalSafe
else if bpm_delta <= 4 and energy_lift > 0.08 -> bassSwap
else if bpm_delta > 12 -> smooth
else -> bassSwap
```

`strategyEqCurves()` 为 incoming/outgoing 的 low/mid/high 提供不同 EQ 偏移。例如 vocalSafe 会明显压低新歌中频，以避免主唱或旋律冲突。

## 11. 后端导出实现

入口：

```python
render_mix(tracks, settings, fmt) -> Path
```

### 11.1 读取和预处理

`_load_stereo()` 把每首歌读成：

```text
shape = (2, samples)
sample_rate = 44100
dtype = float32
```

单声道会复制为双声道，多声道只取前两个声道。

### 11.2 节拍同步

开启 `beatSync` 时：

1. 收集有效 BPM。
2. 取中位数作为目标 BPM。
3. 每首歌的 time stretch rate：

```text
rate = clamp(target_bpm / track_bpm, 0.88, 1.12)
```

4. 差异小于 1.5% 时跳过。
5. 调用 `librosa.effects.time_stretch()`。

### 11.3 响度归一化

开启 `aiPrecision` 或 `loudnessNormalize` 时：

- 先对每首歌归一化到 `targetLufs`。
- 渲染完成后对整段 mix 再归一化一次。
- 峰值超过 ceiling 时按比例降低，避免 clipping。

### 11.4 单曲 mixer 和全局 EQ

每首歌先应用自己的 mixer：

```text
track mixer EQ -> track gain
```

再应用全局 EQ。

后端 `_apply_static_eq()` 的三段分离：

- Low：lowpass 220 Hz
- High：highpass 3200 Hz
- Mid：原信号 - Low - High

输出：

```text
out = buffer + low_band * low + mid_band * mid + high_band * high
```

### 11.5 Crossfade 渲染

`_crossfade()` 从第一首开始逐首拼接。每次处理当前 rendered 和 incoming：

1. 调用 `plan_transition()`。
2. 找到上一首实际 overlap start。
3. 找到下一首 source overlap start。
4. 取双方相同长度片段。
5. 选择普通 fade 或动态 EQ overlap。
6. 拼接：

```text
head + overlap + incoming tail
```

普通 fade：

- 线性 fade
- 或 equal power fade：

```text
fade_out = cos(x * pi / 2)
fade_in  = sin(x * pi / 2)
```

### 11.6 动态 EQ overlap

`_dynamic_eq_overlap()` 把 outgoing 和 incoming 都拆成低中高三段：

```text
low  < 220 Hz
high > 3200 Hz
mid  = original - low - high
```

然后按策略生成不同曲线：

- `vocalSafe`
- `bassSwap`
- `smooth`
- `quickCut`

最终叠加：

```text
overlap =
  prev_low  * prev_low_curve
  + next_low  * next_low_curve
  + prev_mid  * prev_mid_curve
  + next_mid  * next_mid_curve
  + prev_high * prev_high_curve
  + next_high * next_high_curve
```

这个方案的目标不是做复杂源分离，而是在重叠区用频段让位减少低频和人声冲突。

### 11.7 写文件

先写 WAV：

```python
sf.write(wav_path, mix.T, 44100, subtype="PCM_16")
```

如果用户选择 MP3，再用 ffmpeg 转码：

```text
ffmpeg -i input.wav -codec:a libmp3lame -b:a 192k output.mp3
```

## 12. 项目保存与恢复

保存入口：

```js
saveProject()
```

前端发送：

- `name`
- `tracks: state.tracks.map(exportableTrack)`
- `settings`

后端 `POST /api/projects` 生成 uuid 并写入：

```text
backend/data/projects/{project_id}.json
```

加载入口：

```js
loadSelectedProject()
```

流程：

1. 获取项目 JSON。
2. 合并 settings。
3. 对每个 saved track 调用 `/api/tracks/{id}/audio`。
4. 浏览器重新 decode 为 `AudioBuffer`。
5. 恢复 peaks、mixer、状态和本地 id。

## 13. Camelot 调音实现

入口：

```python
render_harmonic_tune(input_path, source_camelot, target_camelot, ...)
```

### 13.1 Camelot 到半音差

`_CAMELOT_TO_TONIC` 把 Camelot 编码映射到 12 平均律 pitch class。

半音差：

```python
diff = (target_tonic - source_tonic) % 12
```

方向：

- `up`：使用 `diff`
- `down`：使用 `diff - 12`
- `nearest`：如果 diff >= 6 则向下，否则向上

### 13.2 谐和目标推荐

`harmonic_targets(reference_camelot)` 返回：

- 同号同 mode
- 前一个数字同 mode
- 后一个数字同 mode
- 同号另一 mode

例如 `9A` 的目标包括：

```text
9A, 8A, 10A, 9B
```

`recommend_pair_tuning()` 会尝试把 A 或 B 调到对方的 harmonic targets，并按半音数和关系给分，最多返回 6 个建议。

### 13.3 高质量调音管线

优先路径：

```text
input
  -> ffmpeg 统一转成 44.1k stereo wav
  -> Demucs htdemucs 分离 vocals/drums/bass/other
  -> drums 保持原调
  -> vocals 用 Rubber Band R3 + formant preservation
  -> bass/other 用 Rubber Band R3 pitch shift
  -> stems 按 gain 合成
  -> highpass 28 Hz
  -> gentle presence
  -> -14 LUFS 归一化
  -> WAV 或 MP3
```

### 13.4 fallback

如果 Demucs 不可用：

- 使用整首歌移调。

如果 Rubber Band 不可用：

- 使用 `librosa.effects.pitch_shift()`。

每次 fallback 都会写入 `warnings`，并在返回的 `tuning` 字段中展示。

### 13.5 调音结果注册为曲目

`POST /api/tracks/{track_id}/tune` 调音完成后调用：

```python
analyze_tuned_output(result, original_meta)
```

它会重新分析输出文件，并写入新的 `{new_track_id}.json`。因此调音产物可以直接参与排序、预览、保存和导出。

## 14. 前端 fallback 分析

当后端不可达或分析失败时，前端 `applyLocalFallbackAnalysis()` 尝试基于 `AudioBuffer` 做简化分析。

包括：

- `buildEnvelope()`：RMS 包络、能量、intro/outro 低能量区。
- `estimateLocalBpm()`：RMS flux 自相关，在 70 到 180 BPM 搜索。
- `estimateLocalKey()`：分块估计 pitch class，再用大/小调模板匹配。
- `localLoudness()`：RMS dB 近似。
- `localPeakDb()`：sample peak dB。

fallback 的目标是保证 UI 可继续使用，不追求专业精度。

## 15. 测试覆盖

后端测试位于：

```text
backend/test_audio_engine.py
```

覆盖点包括：

- 节拍同步会按目标 BPM 拉伸。
- 响度归一化会接近目标 LUFS 并限制峰值。
- 过渡计划会输出 phrase fit、候选点和 overlap anchor。
- 过渡候选包含人声密度字段。
- crossfade 会使用计划中的上一首和下一首锚点。
- track mixer gain 会影响导出 buffer。
- 自动混音策略会在人声密集时选择 vocalSafe。

运行：

```bash
pnpm test:backend
```

前端语法检查：

```bash
pnpm check
```

## 16. 当前边界与后续方向

### 当前边界

- Key/Camelot 识别基于 chroma 和模板匹配，复杂歌曲可能不准。
- downbeat offset 是能量启发式，不等同专业 downbeat detection。
- 人声密度是频段占比近似，不是真正 vocal separation。
- 前端和后端各有一份过渡规划逻辑，虽然保持相近，但未来需要抽象成共享规则或加强回归测试。
- 调音 fallback 到 librosa 时会有更明显音质损失。
- 项目保存依赖本地 `backend/data/uploads` 中的音频仍然存在。

### 可升级方向

- 引入更专业的 downbeat / structural segmentation 模型。
- 把排序从贪心最近邻升级为 beam search、2-opt 或 TSP 近似。
- 把 phrase compatibility、vocal density 和频谱冲突纳入排序评分。
- 支持多用户和任务队列，把长时间导出/调音改为异步 job。
- 增加项目删除、导出历史管理和 data 目录清理。
- 前后端共享 transition plan schema，减少预览与导出的行为偏差。
- 引入更完整的音频质量评估，给调音建议加试听风险评分。
