# 两首歌可接性算法调研与本次改进

## 结论

目前没有一个被行业和学术界共同认可的“单一完善公式”可以直接判断任意两首歌是否可接。更可靠的做法是分层判断：

1. 音频分析层：BPM、beat grid、bar/phrase、key、能量、vocal/bass 密度。
2. Cue 层：为上一首选择 mix-out cue，为下一首选择 mix-in cue，并评估 cue 是否落在 beat/bar/phrase anchor 附近。
3. 可接性评分层：综合 tempo、harmonic、energy、structure、style、vocal、bass、rhythm、cue 等维度。
4. 硬门槛层：当 cue 漂移、tempo stretch、vocal clash、低置信度等问题严重时，不能靠其他高分抵消。
5. 转场策略层：根据风险自动选择 beatmix、breakdown switch、bass swap、quick cut、echo out 等方式。

这和 SmartMix 现在的架构是一致的。本次改进重点补上第 2/4 层：把 cue/phrase 对齐独立成评分维度，并为严重风险设置总分上限。

## 调研依据

- Len Vande Veire 和 Tijl De Bie 的 2018 论文 [From raw audio to a seamless mix: creating an automated DJ system for Drum and Bass](https://doi.org/10.1186/s13636-018-0134-8) 描述了完整 Auto-DJ 系统：从原始音频分析、结构/cue 推断、相邻曲目选择，到最终生成 DJ mix。它不是用一个公式解决问题，而是用多个音乐信息检索特征和规则/优化步骤组合。
- 该论文引用了早期自动 DJ 系统和 “user discomfort” 相关研究，说明 tempo adjustment、vocal、energy、chord/key 信息都会影响可接性，单一 BPM 或 Camelot 匹配不足以判断。
- Mixxx 官方手册把 [beatmatching、cue、Auto DJ](https://manual.mixxx.org/2.5/en/chapters/djing_with_mixxx.html) 作为不同工作流处理，也说明成熟 DJ 软件会把可播放、可对拍、可自动接歌分层建模。
- CUE-DETR / EDM-CUE 方向的研究把 cue point estimation 单独建模，说明 cue 质量本身值得从“辅助字段”升级为显式评分输入。SmartMix 已有可选 `backend/cue_detr.py` 接入点，后续可以继续接入模型 cue。

## 本次算法改进

### 1. 新增 alignment 维度

新增 `alignment_compatibility()`，它会评估：

- outgoing cue 到上一首最近 beat/bar/phrase anchor 的漂移；
- incoming cue 到下一首最近 beat/bar/phrase anchor 的漂移；
- cue 自身的 `phraseAlignment`；
- 双方 beat-grid 分析质量。

输出字段会进入每个 transition 的 `components.alignment`，同时进入 `transitionQuality.metrics.alignmentScore`。

### 2. 调整权重

旧版排序主要依赖 harmonic、tempo、energy、structure 等加权平均。新版把 alignment 加入各 profile：

- `pair_match`：保留 harmonic/tempo 主导，但加入 cue 和 alignment，避免只看调性/速度。
- `handoff`：提高 cue 和 alignment 权重，更贴近真实接歌安全性。
- `sort_recommended`：在推荐排序中显式考虑 cue 落点。
- `sort_harmonic`：仍然偏调性，但不允许忽略节拍落点。

### 3. 加入硬门槛评分上限

新版 `_adjust_score()` 会在加权平均后根据质量风险封顶：

- transition quality 低于阈值时封顶；
- cue grid drift 超过 180ms/260ms 时封顶；
- tempo stretch 超过 6%/8% 时封顶；
- vocal conflict 过高时封顶；
- analysis quality 过低时封顶；
- alignment 单项低于 60 时封顶。

这样可以避免“其他维度很好，但实际会明显跑拍/撞人声”的候选被评为推荐。

## 已补齐的两个增强项

### 1. 学习型 cue 接入

已新增 `backend/learned_cues.py`，把学习型 cue 抽成 provider：

- CUE-DETR：继续通过 `SMARTMIX_ENABLE_CUE_DETR=1` 和 `SMARTMIX_CUE_DETR_CHECKPOINT` 启用。
- 自训练 sidecar：支持 `SMARTMIX_LEARNED_CUE_DIR` 下的 `{track_stem}.cues.json`，或 `SMARTMIX_LEARNED_CUE_FILE` 指向单个 JSON。
- 自训练命令：支持 `SMARTMIX_CUE_MODEL_COMMAND`，命令 stdout 输出 JSON cue 列表。

JSON cue 格式：

```json
{
  "cues": [
    {"time": 32.0, "role": "mix_in", "confidence": 0.91},
    {"time": 188.5, "role": "mix_out", "score": 84}
  ]
}
```

模型 cue 不会直接覆盖规则 cue。它们会先被转换为候选，再通过 SmartMix 当前的 phrase alignment、vocal safety、groove stability、duration room 等规则重新打分，最后并入 `transition_candidates.cue_candidates`。分析结果会新增：

- `transition_candidates.learned_cue`
- 兼容字段 `transition_candidates.cue_detr`

### 2. Mix Method Aware Scoring

已在 `backend/mixability.py` 新增 `mix_method_scores()`，每个 pair 会输出：

- `beatmix`
- `bass_swap`
- `quick_cut`
- `echo_out`
- `breakdown_switch`

每个方法都有：

- `score`：该方法的适配分；
- `threshold`：当前可校准阈值；
- `usable`：是否达到阈值。

`degradedTransition` 现在会参考这些方法分来选择转场方式，并返回：

- `methodScore`
- `methodScores`

这样同一对歌不再只有一个“总可接性分”，而是可以判断“适合长 beatmix，还是更适合 bass swap / quick cut / echo out”。

## 下一步建议

1. 增加离线评测集：收集人工标注的可接/不可接 pair，记录 cue drift、beat drift、transition quality、DJ 评分。
2. 做 A/B 校准：记录用户手动调整 cue、切换转场类型、跳过推荐的行为，把权重和硬门槛从经验值升级为数据校准值。
3. 把 `methodScores` 和人工评分对齐，分别校准 beatmix、bass_swap、quick_cut、echo_out、breakdown_switch 的阈值。
