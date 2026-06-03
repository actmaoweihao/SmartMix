# MSAF-style 双曲重组段落分析

SmartMix 的 Mashup 段落分析现在在原有 novelty / stem-aware 逻辑上增加了 MSAF-style 结构分组。

## 入口

- API：`POST /api/mashup/analyze`
- 后端入口：`backend/mashup.py::analyze_mashup_tracks`
- 核心分析：`backend/segmentation.py::analyze_track_segmentation`

## 分析流程

1. 按小节切分整曲。
2. 为每个小节提取特征：
   - chroma
   - MFCC
   - spectral contrast
   - spectral centroid
   - onset strength
   - energy
   - vocal density
   - drum activity
   - bass energy
3. 构建四类自相似矩阵：
   - harmonic
   - timbre
   - rhythm
   - energy
4. 融合为 `fused` self-similarity matrix。
5. 在 fused SSM 上做 4/8/16 小节多尺度 novelty 检测。
6. 构建 recurrence affinity。
7. 使用 normalized graph Laplacian 做 spectral embedding。
8. 用轻量 k-means 得到 bar-level 结构状态。
9. 将 spectral cluster change 融入边界候选，边界类型为 `msaf_cluster`。
10. 对最终 `sections` 和 `minorSections` 做结构分组。

## 新增输出字段

每个 section 会新增：

```json
{
  "sectionGroup": "A",
  "groupConfidence": 0.82,
  "repetitionScore": 0.91,
  "similarSectionIds": ["A_sec_003"]
}
```

字段含义：

- `sectionGroup`：结构重复分组，按首次出现顺序命名为 `A`、`B`、`C` 等。
- `groupConfidence`：该 section 与同组 section 的相似置信度。
- `repetitionScore`：重复结构强度，用于判断副歌/主歌复现。
- `similarSectionIds`：最相似的其他段落 ID。

分析报告还会返回：

```json
{
  "structuralGroups": [
    {
      "group": "A",
      "count": 2,
      "sectionIds": ["A_sec_001", "A_sec_003"],
      "labels": ["chorus_like"],
      "meanRepetition": 0.88
    }
  ]
}
```

## Stems 的作用

如果已有 Demucs stems，分析会使用：

- `vocals`：修正人声密度、人声入口、人声出口、vocal phrase。
- `drums`：修正鼓活跃度和 groove bed。
- `bass`：修正低频能量和 bass stability。
- `other`：辅助 groove bed 与伴奏判断。

如果 stems 缺失，MSAF-style 分组仍然会运行，但人声和 groove 判断置信度会下降。

## 与原有逻辑的关系

MSAF-style 分析不是替代原有逻辑，而是补强：

- 原有 novelty 检测擅长找变化点。
- MSAF-style spectral grouping 擅长识别重复结构。
- Demucs stems 擅长修正人声安全边界和伴奏 bed。

最终段落边界由 novelty、transition candidate、phrase boundary、vocal entry/exit、drum/bass change 和 `msaf_cluster` 共同决定。
