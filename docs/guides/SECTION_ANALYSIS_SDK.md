# SmartMix 段落分析 / 标注 / 拆分 SDK 调用文档

这份文档给团队成员说明如何调用 SmartMix 的歌曲段落分析能力。这个功能会完成三件事：

- 段落识别：识别歌曲的结构边界。
- 段落标注：给每段打上 `Intro`、`Verse`、`Build`、`Drop / Chorus`、`Break`、`Transition`、`Outro` 等标签。
- 段落拆分结果输出：返回每段的开始时间、结束时间、能量、人声密度、鼓/贝斯强度、风险标记等信息。

推荐优先使用 Python SDK。如果是外部工具或其他服务调用，可以使用 HTTP API。

## 1. 安装依赖

在项目根目录执行：

```bash
pnpm setup:backend
pnpm setup:segmentation
```

如果要结合 Demucs stems 做更细的人声/鼓/贝斯判断，还需要：

```bash
pnpm setup:tuning
```

## 2. Python SDK 调用

### 最简单调用

```python
from backend.sdk import analyze_song_sections


result = analyze_song_sections("songs/demo.mp3")

for section in result["sections"]:
    print(
        section["sectionSubLabel"],
        section["start"],
        section["end"],
        section["labelConfidence"],
    )
```

### 指定算法配置

```python
from backend.sdk import SectionAnalysisConfig, analyze_song_sections


config = SectionAnalysisConfig(
    analyzer="hybrid",       # 可选：hybrid / msaf / smartmix
    boundaries_id="scluster",
    labels_id="scluster",
    feature="pcp",           # 可选：pcp / mfcc / tonnetz
    use_stems=True,
    n_jobs=1,
    include_report=True,
)

result = analyze_song_sections(
    "songs/demo.mp3",
    track_id="demo-song",
    name="Demo Song",
    config=config,
)
```

### 传入已有 Demucs stems

如果已经提前分好四轨，可以把 stem 路径传进去，段落标注会更容易区分人声段、Drop、Break 等层次。

```python
from backend.sdk import analyze_song_sections


result = analyze_song_sections(
    "songs/demo.mp3",
    stems={
        "vocals": "stems/demo/vocals.wav",
        "drums": "stems/demo/drums.wav",
        "bass": "stems/demo/bass.wav",
        "other": "stems/demo/other.wav",
    },
)
```

### 传入已有元数据

如果调用方已经有 BPM、调性、bars、beats 等分析结果，可以通过 `metadata` 传入。传入的字段会覆盖 SDK 自动分析出的字段。

```python
result = analyze_song_sections(
    "songs/demo.mp3",
    metadata={
        "bpm": 128,
        "key": "A Minor",
        "camelot": "8A",
        "duration": 214.2,
        "bars": [0.0, 1.875, 3.75],
    },
)
```

## 3. 返回结构

SDK 返回一个字典，核心字段如下：

```python
{
    "track": {...},
    "config": {...},
    "analyzer": "hybrid",
    "method": "hybrid_msaf_package_smartmix_stem_aware",
    "sections": [...],
    "sectionCount": 8,
    "warnings": [],
    "msaf": {...},
    "availableAlgorithms": {...},
    "report": {...},
}
```

团队业务通常只需要使用 `sections`。

单个 section 示例：

```json
{
  "id": "A_seg_001",
  "trackId": "demo-song",
  "source": "A",
  "start": 0.0,
  "end": 26.4,
  "duration": 26.4,
  "sectionType": "intro",
  "sectionLabel": "Intro",
  "sectionSubLabel": "Intro",
  "rawLabel": "intro_like",
  "arrangementLevel": "sparse",
  "labelConfidence": 0.86,
  "labelReasons": ["first low-vocal section", "early song position"],
  "energy": 0.38,
  "vocalDensity": 0.12,
  "drumActivity": 0.34,
  "bassEnergy": 0.28,
  "riskFlags": []
}
```

## 4. 标签说明

| 字段 | 说明 |
| --- | --- |
| `sectionType` | 程序内部标准类型，例如 `intro`、`verse`、`drop_chorus` |
| `sectionLabel` | 标准展示标签，例如 `Intro`、`Verse`、`Drop / Chorus` |
| `sectionSubLabel` | 更细的展示标签，例如 `Chorus / Hook`、`Build - Rising`、`Break - Low Layer` |
| `rawLabel` | 底层算法原始标签，例如 `chorus_like`、`drop_like` |
| `arrangementLevel` | 编曲层次，可能是 `sparse`、`medium`、`peak`、`rising`、`falling` |
| `labelConfidence` | 标签置信度，范围 0 到 1 |
| `labelReasons` | 为什么打这个标签 |
| `riskFlags` | 风险标记，例如 `short_section`、`coverage_gap`、`low_loopability` |

## 5. 时间轴完整性

SDK 对外输出的 `sections` 会做时间轴完整性修复：

- 不会因为段落短于 4 小节就删除。
- 相邻同类标签会合并，避免同一段落被展示成很多碎片。
- 如果上游算法漏掉某段时间，会补一个 `coverage_gap` 段，保证歌曲时间轴不被挖空。

如果看到 `riskFlags` 里有 `coverage_gap`，说明这段是为了补全时间轴生成的占位段，需要后续继续优化上游边界识别。

## 6. HTTP API 调用

先启动后端：

```bash
pnpm backend
```

默认地址：

```text
http://127.0.0.1:8002
```

### 第一步：上传歌曲

```bash
curl -X POST "http://127.0.0.1:8002/api/tracks" \
  -F "file=@songs/demo.mp3"
```

返回结果里会有 `id`，后续作为 `track_id` 使用。

### 第二步：可选，生成 Demucs stems

```bash
curl -X POST "http://127.0.0.1:8002/api/tracks/{track_id}/stems" \
  -H "Content-Type: application/json" \
  -d "{\"device\":\"auto\",\"force\":false}"
```

### 第三步：调用段落分析 SDK API

```bash
curl -X POST "http://127.0.0.1:8002/api/segmentation/tracks/{track_id}/sections" \
  -H "Content-Type: application/json" \
  -d "{\"analyzer\":\"hybrid\",\"useStems\":true}"
```

请求体字段：

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

## 7. PowerShell 示例

```powershell
$upload = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8002/api/tracks" `
  -Form @{ file = Get-Item ".\songs\demo.mp3" }

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8002/api/tracks/$($upload.id)/stems" `
  -ContentType "application/json" `
  -Body '{"device":"auto","force":false}'

$sections = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8002/api/segmentation/tracks/$($upload.id)/sections" `
  -ContentType "application/json" `
  -Body '{"analyzer":"hybrid","useStems":true}'

$sections.sections | Select-Object sectionSubLabel,start,end,labelConfidence
```

## 8. 常见错误

`Audio file not found`

音频路径不存在。检查传给 `analyze_song_sections()` 的路径是否正确。

`MSAF segmentation failed`

真实 MSAF 包运行失败。如果使用 `analyzer="hybrid"`，SDK 会自动回退到 SmartMix；如果使用 `analyzer="msaf"`，会直接抛错。

`coverage_gap`

这不是异常，而是时间轴补全标记。说明上游段落边界没有覆盖完整歌曲，SDK 为了不丢时间段补了一个可见段。

`stemsUsed` 为 `false`

说明没有传入 stems，或传入的四轨路径不完整。没有 stems 也能分析，但人声/鼓/贝斯层次判断会弱一些。
