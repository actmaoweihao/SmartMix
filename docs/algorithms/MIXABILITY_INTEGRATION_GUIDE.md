# Mixability 调用与集成指南

更新日期：2026-06-15

本文面向 SmartMix 团队内其他开发任务：接歌推荐、自动排序、转场评测、教学解释、批量质检等功能都应优先调用这里定义的稳定入口。

## 推荐调用层级

### 1. 后端 Python 业务代码

优先使用 service 门面：

```python
from backend.services.mixability_service import MixabilityOptions, mixability_service

result = mixability_service.evaluate_pair(
    previous_track,
    incoming_track,
    options=MixabilityOptions(profile="handoff", settings={"targetEnergy": "keep"}),
)
```

不要在新业务里直接调用 `tempo_compatibility()`、`cue_compatibility()`、`_adjust_score()` 这类内部函数。它们属于算法实现细节，后续权重和风险门槛会继续调整。

### 2. 团队脚本 / 离线任务

优先使用 SDK：

```python
from backend.sdk import MixabilityConfig, evaluate_song_pair, order_songs_for_mix, recommend_next_song

config = MixabilityConfig(profile="handoff", mode="recommended", limit=5)

pair = evaluate_song_pair(track_a_metadata, track_b_metadata, config=config)
ordered = order_songs_for_mix([track_a_metadata, track_b_metadata, track_c_metadata], config=config)
next_songs = recommend_next_song(track_a_metadata, [track_b_metadata, track_c_metadata], config=config)
```

SDK 也可以直接传本地音频路径：

```python
pair = evaluate_song_pair("A.wav", "B.wav")
```

默认会先跑 `analyze_audio()`，再做可接性判断。若传入的 metadata 已经包含分析结果，会直接使用 metadata。

### 3. 前端 / 外部调用

继续使用现有 HTTP API：

- `POST /api/mixability/order`
- `POST /api/mixability/recommend-next`
- `POST /api/mixability/evaluate-transitions`

这些接口现在内部也走 `mixability_service`，所以前端和后端任务拿到的判断逻辑一致。

## Service 能力

`backend/services/mixability_service.py` 提供四个稳定方法：

| 方法 | 用途 | 典型场景 |
| --- | --- | --- |
| `evaluate_pair(prev, next)` | 判断两首歌能否接 | Pair Match、单个转场预检 |
| `recommend_next(current, candidates)` | 从候选里推荐下一首 | Deck Mixer、教学推荐、Auto DJ |
| `order_tracks(tracks)` | 对一组歌做可接性排序 | 歌单排序、自动编排 |
| `evaluate_sequence(tracks)` | 评测已有顺序的相邻转场 | cue accuracy / beat drift 评测、批量质检 |

另外提供：

- `resolve_tracks(track_ids, track_overrides)`：按 ID 从上传缓存读取 metadata，并合并调用方传来的临时 metadata。
- `resolve_track_map(track_ids, track_overrides)`：返回 `{track_id: metadata}`，适合推荐接口。

## 常用参数

`MixabilityOptions` / `MixabilityConfig`：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `profile` | `handoff` | `pair_match`、`handoff`、`sort_recommended`、`sort_harmonic` |
| `mode` | `recommended` | 排序模式：`recommended` 或 `harmonic` |
| `limit` | `5` | 推荐下一首时返回数量 |
| `settings` | `{}` | 业务设置，例如 `{"targetEnergy": "up"}` |

## 结果字段怎么用

核心字段：

- `score`：0-100 的最终可接性分数，已经包含硬门槛封顶。
- `rawScore`：纯加权平均分，便于调试。
- `level`：`perfect`、`recommended`、`usable`、`avoid`。
- `summary`：面向 UI 的短解释。
- `components`：tempo、cue、alignment、rhythm、vocal、bass、harmonic、energy、structure、style 的详细分。
- `transitionQuality`：cue drift、tempo stretch、vocal/bass conflict、analysis quality 等质量指标。
- `methodScores`：beatmix、bass_swap、quick_cut、echo_out、breakdown_switch 的方法适配分与阈值。
- `degradedTransition`：是否建议降级转场，以及建议方法。

接歌功能里推荐这样用：

```python
if result["level"] in {"perfect", "recommended"}:
    method = result["degradedTransition"]["method"]
else:
    method = "skip_or_manual_review"
```

如果功能需要直接选择转场方法，优先使用 `degradedTransition.method`；如果需要解释或调参，再读取 `methodScores`。

## 学习型 cue 接入

上传分析阶段会自动调用 `backend.learned_cues.collect_learned_cue_points()`。学习型 cue 会作为候选进入 `transition_candidates.cue_candidates`，再由规则层重新做安全评分。

支持三种 provider：

| Provider | 开关/配置 | 说明 |
| --- | --- | --- |
| CUE-DETR | `SMARTMIX_ENABLE_CUE_DETR=1` | 使用 `SMARTMIX_CUE_DETR_CHECKPOINT` 指定 checkpoint |
| Sidecar JSON | `SMARTMIX_LEARNED_CUE_DIR` 或 `SMARTMIX_LEARNED_CUE_FILE` | 自训练模型离线生成 cue JSON |
| Command | `SMARTMIX_CUE_MODEL_COMMAND` | 本地命令输出 cue JSON，命令里可用 `{path}` 和 `{stem}` |

Sidecar / Command 输出格式：

```json
{
  "cues": [
    {"time": 16.0, "role": "mix_in", "confidence": 0.88},
    {"time": 192.0, "role": "mix_out", "score": 91}
  ]
}
```

`role` 可选；不传时 SmartMix 会根据歌曲位置和局部特征推断 `mix_in` / `mix_out` / `drop` / `bridge` 等角色。

## 约定

- 新功能需要“判断两首歌能不能接”时，优先从 `mixability_service.evaluate_pair()` 开始。
- 新功能需要“选下一首”时，不要自己写排序公式，使用 `recommend_next()`。
- 新功能需要“整条歌单排序”时，使用 `order_tracks()`。
- 新功能需要“评测一串歌哪里接得差”时，使用 `evaluate_sequence()`。
- 算法内部函数仍可用于研究和单元测试，但不作为跨团队稳定接口。
