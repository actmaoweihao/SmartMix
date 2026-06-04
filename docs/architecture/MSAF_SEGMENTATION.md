# 真实 MSAF 双曲重组段落分析

SmartMix 的段落识别已经封装为独立功能，并支持真实 `msaf` Python 包。默认推荐使用 `hybrid` 模式：MSAF 负责专业结构边界，SmartMix 负责人声、鼓、贝斯、groove bed 和安全切点等制作侧指标。

## 安装

```bash
pnpm setup:segmentation
```

`msaf==0.1.80` 在新版 SciPy 环境中会访问旧的 `scipy.inf`。SmartMix 会在导入 MSAF 前自动做运行时兼容补丁，不需要修改 site-packages。

## 入口

- 独立单曲 API：`POST /api/segmentation/tracks/{track_id}`
- MSAF 算法列表：`GET /api/segmentation/msaf/algorithms`
- Mashup API：`POST /api/mashup/analyze`
- 服务封装：`backend/services/segment_analysis.py`
- SmartMix stem-aware 细化：`backend/segmentation.py`

## 单曲调用

```http
POST /api/segmentation/tracks/{track_id}
Content-Type: application/json
```

```json
{
  "analyzer": "hybrid",
  "boundariesId": "scluster",
  "labelsId": "scluster",
  "feature": "pcp",
  "useStems": true,
  "nJobs": 1
}
```

字段说明：

- `analyzer`：`smartmix`、`msaf`、`hybrid`。
- `boundariesId`：MSAF 边界算法，支持 `cnmf`、`foote`、`olda`、`scluster`、`sf`、`vmo` 等。
- `labelsId`：MSAF 标签算法，支持 `cnmf`、`fmc2d`、`scluster`、`vmo`，也可以为 `null`。
- `feature`：MSAF 特征，支持 `pcp`、`mfcc`、`tonnetz`。
- `useStems`：是否读取 Demucs stems 修正人声和 groove 判断。
- `nJobs`：传给 MSAF 的并行任务数。

## Mashup 调用

`/api/mashup/analyze` 支持：

```json
{
  "trackAId": "A",
  "trackBId": "B",
  "barsPerSegment": 16,
  "useStems": true,
  "segmentationAnalyzer": "hybrid"
}
```

`segmentationAnalyzer` 可选：

- `hybrid`：默认，真实 MSAF + SmartMix stem-aware 指标。
- `msaf`：只使用真实 MSAF 包生成结构段落。
- `smartmix`：只使用 SmartMix 原有 novelty / stem-aware 逻辑。

## 分析流程

1. 调用真实 `msaf.process(audio_file, boundaries_id=..., labels_id=..., feature=...)`。
2. 清洗 MSAF 输出的 boundaries 和 labels。
3. 将 MSAF boundary 映射到 SmartMix 小节网格。
4. 使用 SmartMix bar features 计算每个 MSAF section 的能量、人声、鼓、贝斯、loopability、mix-in/out。
5. 使用 SmartMix structural grouping 补充 `sectionGroup`、`repetitionScore` 和 `similarSectionIds`。
6. 在 `hybrid` 模式下，保留 SmartMix 的 vocal phrases、groove bed candidates 和 safe cut points。

## 新增输出字段

每个 section 会新增：

```json
{
  "sectionType": "drop_chorus",
  "sectionLabel": "Drop / Chorus",
  "sectionSubLabel": "Chorus / Hook",
  "arrangementLevel": "peak",
  "layerProfile": {
    "energy": 0.82,
    "vocal": 0.63,
    "drums": 0.76,
    "bass": 0.71,
    "brightness": 0.42,
    "density": 0.67,
    "tension": 0.31,
    "repetition": 0.88
  },
  "labelReasons": ["repeated high-energy hook"],
  "sectionGroup": "A",
  "groupConfidence": 0.82,
  "repetitionScore": 0.91,
  "similarSectionIds": ["A_sec_003"]
}
```

字段含义：

- `sectionType`：标准段落类型，供程序筛选和排序。
- `sectionLabel`：标准展示标签，供 UI 和团队标注使用。
- `sectionSubLabel`：更细的层次标签，例如 `Verse - Sparse`、`Verse - Full`、`Build - Rising`、`Chorus / Hook`。
- `arrangementLevel`：编曲层次，可能是 `sparse`、`medium`、`rising`、`falling`、`peak`。
- `layerProfile`：当前段的人声、鼓、贝斯、能量、亮度、密度和张力画像。
- `labelReasons`：标签判定原因，方便调试“为什么这里是 Build / Break / Chorus”。
- `sectionGroup`：结构重复分组，按首次出现顺序命名为 `A`、`B`、`C` 等。
- `groupConfidence`：该 section 与同组 section 的相似置信度。
- `repetitionScore`：重复结构强度，用于判断副歌/主歌复现。
- `similarSectionIds`：最相似的其他段落 ID。

标准展示标签包括：

```text
Intro
Verse
Build
Drop / Chorus
Break
Transition
Outro
```

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

真实 MSAF 分析不是完全替代原有逻辑，而是补强：

- MSAF 擅长结构边界和重复结构。
- Demucs stems 擅长修正人声安全边界和伴奏 bed。
- SmartMix 原有逻辑擅长给 Mashup 渲染提供制作侧指标。

默认 `hybrid` 模式会优先使用 MSAF sections 作为段落，同时保留 SmartMix 的 stems 语义和风险提示。

## 标签精修

为避免“高能段全部被标成 Chorus”，SmartMix 在 MSAF 边界之后增加 `section_labeler`：

- Chorus / Drop 不再只看高能量，而是结合重复分组、hook 复现、是否接在 Build 后、鼓和贝斯是否同时进入。
- Build 更看重当前段内部能量/张力上升，以及下一段是否明显更强。
- Break 更看重从前一段抽空、鼓/贝斯减少、能量下降。
- Verse 会区分 `Verse - Sparse` 和 `Verse - Full`。
- 当前不会把同一类型的段落按内部 layer change 再切成小片段；边界只来自 MSAF/SmartMix 段落识别结果，标签器会在标注后合并相邻同类段落。
