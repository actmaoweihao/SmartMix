# SmartMix 当前算法实现与后续路线梳理

本文参照 `C:\Users\Weiha\Downloads\梳理.pdf` 的 Auto-DJ 算法调研框架，对当前 SmartMix 项目已经落地的算法与功能、仍处于可选/实验状态的能力，以及下一阶段建议引入的算法工具做一次对齐。

一句话结论：SmartMix 当前已经超出最基础的播放器级 crossfade，进入了“结构感知规则型 Auto-DJ Engine + stem 辅助混音实验”的阶段。下一步不应急着做端到端生成模型，而应补强 beat/downbeat/cue/结构识别精度、把评分体系统一起来，并逐步收集用户修正数据，为学习型 cue 和转场参数推荐做准备。

## 1. 对照调研文档的总体状态

| 调研路线 | 当前状态 | 项目中对应实现 | 判断 |
| --- | --- | --- | --- |
| A. 播放器级 Crossfade / Gapless | 已实现 | 前端 Web Audio 预览、后端导出 crossfade、equal-power fade | 作为 fallback 和基础能力已具备 |
| B. Beatmatch + Phrase-aligned Crossfade | 部分实现 | `backend/analysis.py` beat/grid/bar/phrase，`backend/transition.py`，`backend/mixing.py`，`backend/auto_handoff.py` | 已有规则型雏形，但 downbeat 仍是启发式 |
| C. EQ Crossfade / Frequency-aware Mixing | 已实现 | Web Audio 三段 EQ、动态 EQ、后端 `_dynamic_eq_overlap()`、bassSwap/vocalSafe/smooth/quickCut | 已有频段让位能力 |
| D. 自动 Cue Point / Switch Point 检测 | 部分实现 | `transition_candidates`、`cue_candidates`、`cue_detr.py` 可选模型、Smart Beat Handoff cue 排序 | 规则 cue 已落地，CUE-DETR 是可选接入 |
| E. 完整 Auto-DJ 系统 | 部分实现 | 上传分析、排序、cue、转场计划、试听、导出、教学推荐、Auto Handoff | 已形成 v1 系统闭环，但还不是全局最优或学习型系统 |
| F. Highlight / Drop-based Mix Generation | 部分实现 | Mashup 段落识别、drop/build/break 标签、短片段重组能力 | 有基础，但还没有独立“高光串烧/短视频混剪”产品模式 |
| G. Stem-based Transition / Mashup | 已实现核心链路 | Demucs stems、stem debugger、vocal activity、Mashup layered render、Reference Mix | v1.5 方向已经开始落地 |
| H. 深度学习生成 Transition 控制曲线 | 未实现 | 仅参考 Diff-MST 的可解释 console 参数思想 | 不建议近期作为主线 |
| I. Time-stretch / Pitch-shift 渲染 | 已实现基础与高质量可选 | librosa time stretch/pitch shift、Rubber Band 可选、Harmonic Tuning | 可用，但需要质量风险评估和更稳定的渲染策略 |

## 2. 当前已经实现的算法

### 2.1 音频输入与预处理

已实现能力：

| 能力 | 实现位置 | 说明 |
| --- | --- | --- |
| 多格式音频上传与本地保存 | `backend/services/tracks.py`、`backend/storage.py` | 上传后保存到 `backend/data/uploads/`，分析结果写 JSON |
| 音频解码 | `backend/analysis.py` | 优先 `librosa.load`，失败后用 `imageio-ffmpeg` 转 WAV 再解码 |
| 波形峰值提取 | `backend/analysis.py`、`src/main.js` | 后端生成固定 bins peaks，前端用于 Canvas 波形和 cue 编辑 |
| 前端 fallback 分析 | `src/main.js` | 后端不可用时用浏览器 AudioBuffer 粗估 BPM、调性、能量和响度 |

当前意义：这部分对应调研文档里的“输入层：曲库与音频预处理”，已经能把歌曲转成结构化元数据。

### 2.2 MIR 基础分析

已实现能力：

| 算法/特征 | 实现位置 | 当前方法 |
| --- | --- | --- |
| BPM 检测 | `backend/analysis.py` | `librosa.beat.beat_track` + onset strength；失败时 RMS flux 自相关 |
| beat grid | `backend/analysis.py` | librosa beat frames 转秒 |
| downbeat offset 粗估 | `backend/analysis.py` | 四拍偏移枚举，假设 downbeat 能量更强 |
| bars / phrases | `backend/analysis.py` | 按 4 拍一小节、4 小节一 phrase 从 beat grid 推导 |
| key / mode | `backend/analysis.py` | `chroma_cqt` + Krumhansl-Schmuckler 大小调模板 |
| Camelot 映射 | `backend/matching.py` | 传统 key 映射到 DJ harmonic mixing 的 Camelot code |
| LUFS / true peak | `backend/loudness.py` | 优先 `pyloudnorm`，失败时 RMS 近似；true peak 当前是 sample peak 近似 |
| 能量画像 | `backend/analysis.py`、`docs/audio/ENERGY_SCORING.md` | LUFS、RMS 分位数、crest factor、低频占比、动态范围、intro/outro 相对能量 |
| 风格粗分类 | `backend/analysis.py` | 基于 BPM、低频、percussive ratio、vocal density、brightness 等启发式分类 |

当前边界：

- 没有接入 Essentia RhythmExtractor2013、madmom downbeat/meter tracking。
- downbeat 不是模型级检测，复杂节奏、现场版、弱鼓点歌曲会不稳。
- key 检测仍是 chroma 模板法，对转调、无明确和声中心、强噪声/强采样音乐会偏粗。

### 2.3 结构、段落与 cue 候选

已实现能力：

| 能力 | 实现位置 | 说明 |
| --- | --- | --- |
| intro/outro 候选 | `backend/analysis.py` | 结合低能量区、小节位置、人声密度、能量选择入点/出点 |
| vocal density curve | `backend/analysis.py` | HPSS harmonic 分量中 300-3400Hz 频段占比，作为人声/主体密度近似 |
| energy curve | `backend/analysis.py` | 小节级 RMS 能量曲线 |
| cue candidates v2 | `backend/analysis.py` | 生成 `mix_in`、`mix_out`、`drop`、`bridge`、`drum_loop`、`vocal_safe` 等角色 cue |
| cue 评分 | `backend/analysis.py` | phrase alignment、vocal safety、groove stability、novelty、section boundary、duration room |
| CUE-DETR 可选接入 | `backend/cue_detr.py` | 通过 `SMARTMIX_ENABLE_CUE_DETR` 开关和 Hugging Face checkpoint 预测 cue，再合并到规则 cue |
| MSAF hybrid 段落分析 | `backend/services/segment_analysis.py`、`backend/segmentation.py` | 可选真实 `msaf` 包，结合 SmartMix bar/stem 指标输出结构段落 |
| 段落标签精修 | `backend/services/section_labeler.py` | 标注 Intro、Verse、Build、Drop / Chorus、Break、Transition、Outro |

当前边界：

- CUE-DETR 不是默认主路径，依赖 `torch`、`transformers`、checkpoint 和环境变量。
- 规则 cue 已可用，但还没有基于用户修正数据持续学习。
- MSAF 是可选依赖，没安装时会退回 SmartMix 自有逻辑。

### 2.4 排序与可接性评分

已实现能力：

| 能力 | 实现位置 | 当前算法 |
| --- | --- | --- |
| 综合推荐排序 | `src/main.js`、`docs/algorithms/SORTING_ALGORITHM.md` | BPM、Camelot、能量加权评分 + 贪心最近邻 |
| 谐和优先排序 | `src/main.js` | 提高 Camelot 权重 |
| BPM 升/降序 | `src/main.js` | 直接排序 |
| 能量弧线 | `src/main.js` | 先升能量，再让高能段回落 |
| Pair Match | `backend/matching.py`、`docs/algorithms/SONG_MATCHING_SCORING.md` | Camelot、BPM、Energy、Structure 四项双向评分 |
| Smart Beat Handoff 排序 | `backend/auto_handoff.py` | tempo、cue、rhythm bed、vocal safety、bass safety、harmonic、energy flow 综合评分 |

当前关键公式：

```text
前端综合推荐：
score = 0.55 * BPM差
      + 0.30 * Camelot距离
      + 0.15 * 能量差

后端 Pair Match：
total = 0.45 * Camelot
      + 0.30 * BPM
      + 0.15 * Energy
      + 0.10 * Structure

Smart Beat Handoff：
score = 0.24 * tempo
      + 0.18 * cue
      + 0.16 * rhythmBed
      + 0.14 * vocalSafety
      + 0.12 * bassSafety
      + 0.10 * harmonic
      + 0.06 * energyFlow
```

当前边界：

- 前端歌单排序仍是贪心，不保证全局最优。
- 前端综合排序和后端 Auto Handoff 的评分体系并未完全统一。
- 排序层还没有充分使用 cue_candidates、vocal density、bass safety、phrase fit 等后端已有信息。

### 2.5 转场计划与渲染

已实现能力：

| 能力 | 实现位置 | 说明 |
| --- | --- | --- |
| TransitionPlan | `backend/transition.py` | 根据 intro/outro 候选、BPM、phraseBars、crossfade 设置计算 overlap |
| 前端时间线计划 | `src/main.js` | `buildTimeline()`、`planClientTransition()` 用于实时预览 |
| Web Audio 实时预览 | `src/main.js` | AudioBufferSource、GainNode、BiquadFilter、动态 EQ 自动化 |
| 后端导出 | `backend/mixing.py` | stereo buffer 拼接、beatSync、EQ、动态 EQ、LUFS normalize、MP3/WAV |
| Equal-power fade | `backend/mixing.py`、`src/audio/crossfade.ts` | 使用 cos/sin 曲线降低线性 fade 的响度凹陷 |
| 动态 EQ overlap | `backend/mixing.py` | low/mid/high 分频后按 vocalSafe、bassSwap、smooth、quickCut 等曲线让位 |
| 真实转场试听 | `backend/seamless.py` | 生成 pair transition audio，支持 cue 对齐、tempo/pitch 计划、stem-like overlap、echo/glue/rhythm bridge |
| DJ 教学推荐 | `src/transitions/*`、`src/explain/*`、`src/practice/*` | beatmix、bassSwap、breakdownSwitch、echoOut、quickCut、wideBpmLoop 等模板和解释 |

当前边界：

- 前后端各有一份转场计划逻辑，长期需要共享 schema 或增加一致性测试。
- BeatSync 主要使用 librosa time stretch，质量不如 Rubber Band 稳定。
- 对 drop swap / double drop 这类高风险转场，已有模板和 mashup 能力，但 cue 精度与低频控制还需要继续加强。

### 2.6 Stem 分轨、冲突控制与 Mashup

已实现能力：

| 能力 | 实现位置 | 说明 |
| --- | --- | --- |
| Demucs 四轨分离 | `backend/services/stem_separation.py`、`backend/api/tracks.py` | 输出 vocals、drums、bass、other，并缓存到 `backend/data/stems/` |
| stem 下载 | `backend/api/tracks.py` | 每个 stem 可独立下载 WAV |
| vocal activity | `backend/vocal_activity.py` | 基于 vocals stem 提取人声区域、入口、释放点和 activity curve |
| Stem Debugger | `src/main.js` | 前端 stem 播放、solo/mute/gain/EQ/compressor/pan/master 参数 |
| Diff-MST 风格参数借鉴 | `docs/integrations/DIFF_MST_INTEGRATION.md`、`backend/mixing.py` | 借鉴可解释 mixing-console 参数，不加载 Diff-MST 训练栈 |
| Reference Mix | `backend/reference_mix.py`、`auto_mix/auto_mix.py` | 用参考曲特征推导 stems gain/EQ/compressor/master 参数，并可做小规模优化 |
| Mashup 段落分析 | `backend/mashup.py`、`backend/segmentation.py` | 分析两首歌的 8/16 小节段落、能量、人声、鼓、贝斯、重复结构 |
| Mashup 方案生成 | `backend/mashup.py` | 拼接、叠加、groove bed、vocal priority、energy curve、alternatives |
| Mashup 渲染 | `backend/mashup.py` | layered render、stem-aware full mix、automation、duck、filter sweep、echo/reverb tail |

当前意义：调研文档里的 v1.5 “Stem-assisted Mixing” 已经有可运行链路，不只是路线图。

当前边界：

- Demucs 是可选依赖，首次运行成本高。
- stem-based transition 的质量高度依赖分离质量和节拍/结构准确性。
- Reference Mix 当前是规则和随机/局部优化，不是训练好的 Diff-MST 模型。

### 2.7 Harmonic Tuning 与自动修复

已实现能力：

| 能力 | 实现位置 | 说明 |
| --- | --- | --- |
| Camelot 半音差计算 | `backend/tuning.py` | 根据 source/target Camelot 计算 nearest/up/down 移调半音数 |
| 谐和目标推荐 | `backend/tuning.py` | 推荐同号、前后相邻、相对大小调等 harmonic targets |
| 高质量调音管线 | `backend/tuning.py` | Demucs 分轨后 vocals 用 Rubber Band formant，bass/other 移调，drums 尽量不移调 |
| fallback 调音 | `backend/tuning.py` | Demucs 或 Rubber Band 不可用时退回整曲/librosa pitch shift |
| Pair Match Repair | `backend/repair.py` | 自动选择 A 或 B，生成 pitch、tempo、energy 修复计划并渲染新音频 |

当前边界：

- Rubber Band CLI 是外部依赖，未安装时音质下降。
- 调音/变速还缺少统一的质量评分，例如 artifacts、formant risk、stretch risk。

## 3. 当前已经实现的产品功能

| 功能 | 当前实现 | 对应算法基础 |
| --- | --- | --- |
| 多歌曲上传与分析 | 已实现 | 解码、BPM/key/energy/loudness/style/cue 分析 |
| 曲目表与指标展示 | 已实现 | 后端分析结果 + 前端 fallback |
| 推荐排序 | 已实现 | 贪心路径 + BPM/Camelot/能量评分 |
| 手动排序/删除/选择 | 已实现 | 前端状态管理 |
| 波形显示与 IN/OUT cue 编辑 | 已实现 | peaks、Canvas、用户手动覆盖 cue |
| 实时预览 | 已实现 | Web Audio 调度、EQ、filter、gain automation |
| 后端导出 MP3/WAV | 已实现 | 后端 DSP 渲染、LUFS normalize、ffmpeg MP3 |
| Pair Match 双曲匹配 | 已实现 | Camelot/BPM/Energy/Structure 双向评分 |
| Pair Match 自动修复 | 已实现 | pitch/tempo/energy 处理 |
| Smart Beat Handoff | 已实现 | cue 排序、rhythm bed、vocal/bass safety、转场类型选择 |
| DJ 教学推荐与试听 | 已实现 | 多模板 transition recommendation + seamless preview |
| Demucs 分轨调试 | 已实现，可选依赖 | vocals/drums/bass/other 分离 |
| Stem 下载 | 已实现 | cached stems |
| Reference Mix | 已实现，可选 stems | 参考曲特征驱动 stem mixer |
| Harmonic Tuning | 已实现，可选高质量依赖 | Camelot 移调、Demucs、Rubber Band |
| Mashup Builder | 已实现实验区 | MSAF/SmartMix 段落、stem-aware plan、layered render |
| 项目保存/加载 | 已实现 | 本地 JSON + 上传音频引用 |

## 4. 需要继续补强的算法工具与功能

### 4.1 v1 优先：让规则型 Auto-DJ 更稳定

| 要改进的功能 | 建议算法/工具 | 原因 | 优先级 |
| --- | --- | --- | --- |
| beat/downbeat 精度 | `madmom` DBNDownBeatTrackingProcessor 或 Essentia RhythmExtractor2013/BeatTrackerDegara | 当前 downbeat 是能量启发式，drop mix 和 16/32 小节对齐会受影响 | P0 |
| 结构边界稳定性 | MSAF 默认安装路径优化；可评估 `scluster`、`foote`、`cnmf` 的场景差异 | Mashup 和 cue 都依赖结构边界 | P0 |
| cue 选择一致性 | 统一 `transition_candidates`、Auto Handoff cue、教学 cue 的数据结构和评分 | 现在多个模块各算一套 cue 或评分 | P0 |
| 预览/导出一致性 | 共享 transition plan schema，增加 golden fixture 测试 | 避免浏览器听到和导出文件不一致 | P0 |
| 排序全局优化 | beam search、2-opt、TSP 近似 | 贪心排序容易陷入局部最优 | P1 |
| 频谱冲突评分 | 低/中/高频 overlap、低频 masking、vocal overlap | 排序阶段提前避免“低频撞车/人声打架” | P1 |
| 质量评测 | transition score、beat drift、LUFS jump、spectral jump、vocal conflict、人工 A/B 打分 | 没有评测就难迭代 | P1 |

建议产物：

```text
一个统一的 Mixability(A, B) 评分服务：
tempo + beat_grid + phrase + key + energy + vocal + bass + style + novelty
```

它应该同时服务：

- 歌单排序
- Pair Match
- Smart Beat Handoff
- 教学推荐
- Mashup plan ranking

### 4.2 v1.5：把 stem-assisted transition 做成稳定卖点

| 要实现/改进的功能 | 建议算法/工具 | 原因 |
| --- | --- | --- |
| 人声冲突处理 | Demucs vocals + vocal activity regions + vocal duck automation | 比频段人声近似更可靠 |
| 低频冲突处理 | Demucs bass/drums + low-frequency ratio + bass swap automation | Double drop / stem bridge 必须控制 bass |
| drums bridge | 使用 outgoing 或 incoming drums stem 作为 8/16 bars rhythm bed | 对应调研里的 Stem Bridge / Rolling Transition |
| acapella-over-intro | vocals stem + incoming instrumental intro + key/tempo guard | 形成 AI remix 差异化 |
| stem 分离任务队列 | 后端异步 job，例如 RQ/Celery/本地队列 | Demucs 运行时间长，不应阻塞 API |
| stem 质量风险提示 | vocals bleed、bass leakage、drum energy、phase/correlation 检测 | 分离质量差时自动降级到 full-track transition |

建议规则：

```text
如果 stems 可用：
  优先用 vocals activity 判断人声安全区
  用 drums stem 判断 groove bed 稳定性
  用 bass stem 判断低频冲突和 bass swap 时机
否则：
  回退到 vocal_density_curve + low_frequency_ratio + dynamic EQ
```

### 4.3 v2：学习型 Cue Point 与转场参数推荐

| 要实现的功能 | 建议算法/工具 | 数据来源 |
| --- | --- | --- |
| 学习型 cue detector | CUE-DETR 或轻量 CNN/Transformer cue classifier | EDM-CUE、用户手动 cue 修正、内部标注 |
| transition type classifier | XGBoost/LightGBM/小型 MLP | 规则系统生成候选 + 用户选择/跳过行为 |
| EQ/fader 参数预测 | 回归模型或 sequence model | 用户编辑的 EQ、filter、crossfader 曲线 |
| 个性化接歌偏好 | contextual bandit / learning-to-rank | 用户对推荐的采纳、试听时长、手动修改 |
| transition dataset | 保存 pair features、cue、render settings、用户反馈 | 作为后续模型训练基础 |

不建议近期直接做：

- 端到端音频生成 transition。
- 直接训练 DJtransGAN 类模型作为主路径。
- 大规模真实 DJ set 对齐系统。

更稳的路径是：

```text
规则系统生成 baseline
-> 记录用户修正 cue / transition type / EQ fader 参数
-> 训练模型预测参数
-> 模型只控制可解释参数，不直接生成最终音频
```

### 4.4 v2.5 / v3：高光串烧与数据驱动 DJ 风格

| 功能方向 | 建议算法/工具 | 说明 |
| --- | --- | --- |
| 高光派对串烧 | highlight/drop/hook detection + filter-and-rank | 可做“30 秒短视频音乐混剪”或“高能派对串烧” |
| Drop Swap / Double Drop | drop boundary detector + bass/vocal conflict guard | 适合 EDM/Bass/DnB，但失败成本高 |
| DJ 风格学习 | mix-to-track alignment + transition metadata extraction | 需要真实 DJ set 数据和版权策略 |
| 风格模式 | club/chill/aggressive/smooth/TikTok short mix | 先用规则参数 profile，后续用学习模型微调 |

## 5. 推荐开发顺序

### 阶段 1：统一评分与转场计划

目标：让已有功能不再各自为政。

当前进展：

- 已新增 `backend/mixability.py`，作为后端统一可接性评分入口。
- `Pair Match` 已改为通过 `evaluate_mixability(..., profile="pair_match")` 计算总分与组件分。
- `Smart Beat Handoff` 已改为通过 `evaluate_mixability(..., profile="handoff")` 计算 pair score，并继续保留原有 tempo/cue/rhythm/vocal/bass/harmonic/energy 组件字段。
- 已新增 `POST /api/mixability/order`，用于后端统一可接性排序。
- 前端 `recommended` / `harmonic` 排序已优先调用后端 Mixability 排序；后端不可用时回退原本地排序。
- Mixability 排序结果已返回并写回每段推荐的 outgoing/incoming cue，排序后会直接影响时间线预览和导出锚点。
- 时间线当前转场和 Deck Mixer 已显示 Mixability 分数及低分风险组件，方便调试排序原因。
- 已新增 `POST /api/mixability/recommend-next`，教学推荐会用后端 Mixability 结果重排候选歌曲。
- 教学卡片、无缝试听生成和“使用这个接法”已复用 Mixability 推荐的 cue 与 overlap，让推荐分数、试听和时间线应用保持一致。
- 后端单曲分析已新增 `analysis_quality`，输出 beatGrid、downbeat、structure、cue、key、loudness 组件质量与整体等级。
- Mixability 的 rhythm/structure 评分已参考 `analysis_quality`，避免在 beat grid 或结构识别不可靠时过度推荐长过渡。
- Deck Mixer 已展示当前曲目的 Analysis 等级和整体分数，便于判断推荐是否建立在可靠分析上。
- Mixability 已新增 `transitionQuality`，输出 beat grid drift、tempo stretch、人声/低频冲突、分析质量下限等转场质量指标。
- 排序结果、教学推荐和 Deck Mixer 已展示 `transitionQuality` 等级及首条风险提示，作为后续 cue accuracy / beat drift 评测入口。
- Mixability 已新增 `degradedTransition`，当 beat drift、tempo stretch、人声/低频冲突或分析质量风险过高时，会建议降级为 `echo_out`、`quick_cut`、`breakdown_switch` 或 `bass_swap`。
- Smart Beat Handoff 和教学推荐已读取 `degradedTransition`，低质量长混音会自动改用更安全的过渡类型和更短 overlap。
- 已新增 `POST /api/mixability/evaluate-transitions`，可批量评估当前相邻转场的 cue drift、transitionQuality 和自动降级结果。
- 新模块保留旧 API 字段名，例如 `phrase_bars`、`overlap_seconds`、`normalized_bpm_b`，避免前端 Pair Match 面板被重构影响。

建议任务：

1. 抽出统一 `MixabilityScore` schema。
2. 将 Pair Match、Smart Beat Handoff、排序、教学推荐的评分字段对齐。
3. 增加 transition plan fixture 测试，校验前端预览和后端导出 cue/overlap 一致。
4. 把 `cue_candidates` 作为排序和教学推荐的主输入。

### 阶段 2：补强 beat/downbeat/structure

目标：让 Smooth Blend、Drop Mix、Double Drop 的基础更准。

建议任务：

1. 接入 madmom 或 Essentia 做 downbeat/meter tracking 对比实验。
2. 建立小型本地评测集，保存人工标注的 beat/downbeat/cue。
3. 对 MSAF 算法做自动 benchmark，按 EDM/Pop/Hip-hop/Lo-fi 分类选择默认参数。
4. 给每首歌输出 `analysis_quality`，包括 beat grid confidence、downbeat confidence、structure confidence。

### 阶段 3：产品化 stem-assisted transition

目标：把“AI DJ”差异化从实验区推到主流程。

建议任务：

1. 将 Demucs stems 的 vocal activity 写入所有 cue/transition 评分。
2. 做 `Stem Bridge`、`Drums Bridge`、`Bass Swap` 的正式转场模板。
3. 给转场渲染增加 stem 可用性检查与自动降级。
4. 将 stem 分离改为异步任务，并在 UI 中显示队列进度。

### 阶段 4：开始收集学习数据

目标：为 v2 学习型 cue 和参数推荐铺路。

建议任务：

1. 记录用户手动移动 IN/OUT cue、修改过渡时长、EQ、filter、stem gain 的行为。
2. 记录用户采纳/拒绝推荐转场的事件。
3. 为每次 transition preview/export 保存匿名化 feature payload。
4. 先训练轻量 ranker/classifier，而不是生成式模型。

## 6. 当前项目的关键边界

- 当前是“规则型 + 可选模型/可选高质量依赖”，不是全学习型 Auto-DJ。
- CUE-DETR、MSAF、Demucs、Rubber Band 都依赖本地环境，必须做好 fallback。
- 大多数评分权重是手写规则，优点是可解释，缺点是风格适配有限。
- 前端和后端都有部分重复逻辑，后续需要共享数据结构或加强回归测试。
- 评测体系还没有系统化，建议优先补 `beat drift`、`cue accuracy`、`LUFS jump`、`vocal conflict`、`bass conflict`、人工评分等指标。

## 7. 结论

按调研文档的路线来看，SmartMix 当前已经完成了 v1 的大部分骨架，并且提前进入 v1.5 的 stem-assisted mixing 和 mashup 实验。最值得投入的下一步不是堆更多转场特效，而是把已有算法统一成稳定的 Auto-DJ 决策链：

```text
MIR 分析
+ 结构/ cue 检测
+ Mixability 统一评分
+ Transition Planner
+ EQ/fader/stem automation
+ 可验证的渲染与评测体系
+ 用户反馈数据闭环
```

这样可以先把“听起来真的能接”做稳，再逐步升级到“越用越像 DJ”的学习型系统。
