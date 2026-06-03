# SmartMix Demucs 分轨 SDK 与接口调用文档

这份文档给团队成员说明如何调用 SmartMix 的 Demucs 分轨能力。支持两种方式：

- Python SDK：适合在本项目内写 Python 脚本或服务代码时直接调用。
- HTTP API：适合外部工具、前端、测试脚本或其他服务调用。

## 1. 安装分轨依赖

基础后端依赖只能支持上传、分析、预览等功能；如果要使用真实 Demucs 分轨，需要安装可选依赖：

```bash
pnpm setup:backend
pnpm setup:tuning
```

如果机器有 NVIDIA GPU，建议先安装 CUDA 版 PyTorch，再安装分轨依赖：

```bash
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pnpm setup:tuning
```

## 2. Python SDK 调用方式

适用场景：团队成员在本仓库里写 Python 代码，或者脚本运行环境可以 import `backend` 包。

```python
from pathlib import Path

from backend.services.stem_separation import separate_demucs_stems


result = separate_demucs_stems(
    input_path=Path("songs/demo.mp3"),
    workspace=Path("backend/data/stems/manual-demo"),
    device="auto",  # 可选："auto"、"cuda"、"cpu"
)

print(result.engine)  # demucs
print(result.device)  # cuda 或 cpu
print(result.stems["vocals"])
print(result.stems["drums"])
print(result.stems["bass"])
print(result.stems["other"])
```

SDK 会把四路分轨文件写到：

```text
{workspace}/demucs_api/vocals.wav
{workspace}/demucs_api/drums.wav
{workspace}/demucs_api/bass.wav
{workspace}/demucs_api/other.wav
```

当前公开 SDK 入口：

```python
from backend.services.stem_separation import (
    StemSeparationResult,
    demucs_available,
    prepare_demucs_input,
    resolve_torch_device,
    separate_demucs_stems,
    separate_prepared_demucs_input,
)
```

推荐默认使用 `separate_demucs_stems(...)`。`separate_prepared_demucs_input(...)` 是高级入口，只适合已经把原始音频转换成 SmartMix Demucs 输入 WAV 的调用方。

## 3. 启动后端服务

```bash
pnpm backend
```

默认本地 API 地址：

```text
http://127.0.0.1:8002
```

## 4. 外部 HTTP 调用流程

### 第一步：上传音频

```bash
curl -X POST "http://127.0.0.1:8002/api/tracks" \
  -F "file=@songs/demo.mp3"
```

返回结果里会有 `id` 字段。后续把这个 `id` 当作 `track_id` 使用。

### 第二步：生成或复用 Demucs 分轨

```bash
curl -X POST "http://127.0.0.1:8002/api/tracks/{track_id}/stems" \
  -H "Content-Type: application/json" \
  -d "{\"device\":\"auto\",\"force\":false}"
```

请求体：

```json
{
  "device": "auto",
  "force": false
}
```

字段说明：

- `device`：运行设备，可选 `auto`、`cuda`、`cpu`。
- `force`：是否强制重新分轨。`false` 表示如果缓存已存在就直接复用；`true` 表示重新生成。

返回示例：

```json
{
  "trackId": "track-1",
  "engine": "demucs",
  "device": "cuda",
  "cached": false,
  "stems": {
    "vocals": {
      "url": "/api/tracks/track-1/stems/vocals/audio",
      "path": "backend/data/stems/track-1/demucs_api/vocals.wav"
    },
    "drums": {
      "url": "/api/tracks/track-1/stems/drums/audio",
      "path": "backend/data/stems/track-1/demucs_api/drums.wav"
    },
    "bass": {
      "url": "/api/tracks/track-1/stems/bass/audio",
      "path": "backend/data/stems/track-1/demucs_api/bass.wav"
    },
    "other": {
      "url": "/api/tracks/track-1/stems/other/audio",
      "path": "backend/data/stems/track-1/demucs_api/other.wav"
    }
  }
}
```

### 第三步：下载单路分轨音频

```bash
curl -L "http://127.0.0.1:8002/api/tracks/{track_id}/stems/vocals/audio" \
  -o vocals.wav
```

可下载的 stem 名称：

```text
vocals
drums
bass
other
```

## 5. PowerShell 调用示例

Windows 下可以直接使用下面这段：

```powershell
$upload = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8002/api/tracks" `
  -Form @{ file = Get-Item ".\songs\demo.mp3" }

$stems = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8002/api/tracks/$($upload.id)/stems" `
  -ContentType "application/json" `
  -Body '{"device":"auto","force":false}'

Invoke-WebRequest `
  -Uri "http://127.0.0.1:8002$($stems.stems.vocals.url)" `
  -OutFile ".\vocals.wav"
```

## 6. 缓存规则

SmartMix 会把 API 生成的分轨缓存到：

```text
backend/data/stems/{track_id}/demucs_api/
```

只有下面四个文件都存在时，API 才认为缓存可用：

```text
vocals.wav
drums.wav
bass.wav
other.wav
```

如果需要强制重新生成，调用分轨接口时传：

```json
{
  "device": "auto",
  "force": true
}
```

## 7. 常见错误

`Demucs is not available. Install backend/requirements-tuning.txt first.`

说明 Demucs 依赖还没装，执行：

```bash
pnpm setup:tuning
```

`CUDA was requested, but PyTorch cannot see a CUDA GPU.`

说明请求里指定了 `"device": "cuda"`，但当前 PyTorch 没检测到可用 CUDA。可以改成 `"device": "auto"` 或 `"device": "cpu"`，也可以重新安装 CUDA 版 PyTorch。

`Track not found`

说明还没有上传这首歌，或者 `track_id` 写错了。先调用 `POST /api/tracks` 上传音频，再使用返回的 `id`。

`Stem not found`

说明该路分轨文件不存在。先确认分轨接口已经成功返回，再请求 `vocals`、`drums`、`bass` 或 `other` 其中之一。
