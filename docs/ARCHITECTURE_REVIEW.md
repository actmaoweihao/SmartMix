# SmartMix 软件架构梳理与改进建议

## 当前架构概览

SmartMix 是一个本地运行的智能 DJ / 混音工作台，整体采用“浏览器工作台 + 本地 FastAPI 音频引擎”的架构。

```text
Browser / Vite
  - src/main.js UI、状态、事件、预览、项目操作
  - src/analysis/* 前端分析与评分辅助
  - src/transitions/* 接歌推荐与教学步骤
  - src/seamless/* 无缝过渡计划
  - Web Audio API / Canvas 实时试听与波形
        |
        | HTTP JSON / multipart / audio files
        v
FastAPI backend
  - backend/main.py API 路由入口
  - backend/analysis.py 音频分析
  - backend/matching.py 两歌匹配评分
  - backend/mixing.py 导出渲染
  - backend/seamless.py 无缝过渡试听
  - backend/tuning.py Camelot 调音
  - backend/mashup.py Mashup 编排与渲染
  - backend/reference_mix.py 参考曲风格混音
        |
        v
backend/data/
  uploads/   上传音频与分析 JSON
  stems/     Demucs 分轨缓存
  exports/   导出音频
  projects/  项目 JSON
```

## 核心模块职责

| 层级 | 模块 | 当前职责 |
|---|---|---|
| 前端应用层 | `src/main.js` | UI 模板、状态管理、事件绑定、文件上传、实时试听、项目保存、Mashup 面板、教学面板、分轨调试 |
| 前端算法层 | `src/analysis/*`、`src/transitions/*`、`src/seamless/*` | BPM/Key/能量/人声辅助分析、接歌推荐、过渡策略、教学解释 |
| 前端基础设施层 | `src/api/client.js` | 后端 API 地址、URL 拼接、JSON 请求与错误处理 |
| 后端 API 层 | `backend/main.py` | FastAPI 路由、请求校验、调用领域模块、返回文件 |
| 后端 API 子路由 | `backend/api/projects.py` | 项目保存、项目列表、项目加载 |
| 后端 API 子路由 | `backend/api/tracks.py` | 曲目上传、音频读取、Demucs 分轨、参考曲混音、Camelot 调音 |
| 后端服务层 | `backend/services/tracks.py` | 曲目上传分析、曲目元数据读取、stem 路径和响应结构 |
| 后端领域层 | `backend/analysis.py`、`matching.py`、`mixing.py`、`seamless.py`、`mashup.py`、`tuning.py` | 音频分析、匹配评分、混音导出、无缝试听、Mashup、调音 |
| 后端基础设施层 | `backend/storage.py`、本地文件系统 | 上传、导出、项目、分轨缓存的目录管理和 JSON 读写 |

## 主要数据流

1. 上传歌曲：前端选择文件，浏览器本地解码用于波形和预览，同时 `POST /api/tracks` 上传到后端。
2. 音频分析：后端用 `librosa`、`numpy`、`pyloudnorm` 等生成 BPM、Camelot、能量、响度、结构和候选过渡点。
3. 排序与推荐：前端结合后端分析结果，用 TypeScript 业务模块计算排序、下一首推荐和接法解释。
4. 试听：浏览器用 Web Audio API 根据当前时间线做实时混音。
5. 重型渲染：导出、无缝过渡、Mashup、分轨、参考混音由后端生成真实音频文件。
6. 持久化：项目状态保存为本地 JSON，音频文件保存在 `backend/data` 下。

## 当前架构问题

| 问题 | 影响 | 建议 |
|---|---|---|
| `src/main.js` 过大 | UI、状态、API、音频播放、业务流程混在一起，后续功能难维护 | 逐步拆成 `state/`、`views/`、`api/`、`audio/`、`features/` |
| `backend/main.py` 路由集中 | API 入口越来越长，领域模块边界不够明显 | 按功能拆 FastAPI routers，如 `tracks`、`projects`、`mix`、`mashup` |
| API 契约缺少统一类型 | 前后端字段靠约定，容易出现字段漂移 | 引入 Pydantic response model，并生成/维护前端类型 |
| 长任务同步执行 | Demucs、Mashup、导出可能阻塞请求和 UI 等待 | 引入后台任务队列和 job 状态查询接口 |
| 本地文件存储缺少索引 | 上传、导出、项目之间靠 JSON 和路径关联 | 增加轻量 SQLite 元数据索引，保留音频文件落盘 |
| 文档偏功能说明 | 架构决策、模块边界和演进路线不足 | 保留本文件，并对关键技术选择写 ADR |

## 已落地改进

本次先完成一个低风险的前端解耦：

- 新增 `src/api/client.js`
- 将 API host、协议、端口、URL 拼接和 `fetchJson()` 错误处理从 `src/main.js` 移出
- `src/main.js` 改为通过 `API_BASE_URL`、`apiUrl()`、`fetchJson()` 访问后端

这样后续如果要切换端口、改为环境变量、增加鉴权、统一超时、埋点或重试，不需要继续修改庞大的 `main.js`。

第二步完成一个低风险的后端路由解耦：

- 新增 `backend/api/projects.py`
- 将 `/api/projects` 的保存、列表、加载路由从 `backend/main.py` 移出
- `backend/main.py` 改为通过 `app.include_router(projects_router)` 装配项目路由

这保留了现有 API 路径和响应结构，但让后端开始形成“app 装配层 + 功能 router + 领域模块”的形状。后续拆 `tracks`、`mashup`、`exports` 时可以沿用同一模式。

第三步继续拆出曲目相关 API：

- 新增 `backend/api/tracks.py`
- 新增 `backend/services/tracks.py`
- 将 `/api/tracks` 上传、音频文件读取、stem 分离、stem 音频读取、参考曲混音和调音接口从 `backend/main.py` 移出
- `backend/main.py` 仍保留 match、export、transition-preview、mashup 等尚未拆分的路由，但通过 `services.tracks` 读取曲目元数据

这一步让“曲目资源”成为独立 API 模块，并把可复用的曲目元数据和 stem 路径逻辑放进服务层。后续拆 match、transition、mashup 时不需要从 router 互相导入实现细节。

## 后续推荐改进顺序

1. 前端继续拆分 `src/main.js`：优先抽 `api`、`audioPreview`、`projectStore`、`stemDebugger`、`mashupPanel`。
2. 后端拆分 router：先把 `tracks/stems`、`projects`、`mashup` 从 `backend/main.py` 移出。
3. 给后端响应补 Pydantic model：从 `TrackAnalysis`、`ProjectPayload`、`ExportResult` 开始。
4. 给长任务加 job 化：`POST` 返回 job id，前端轮询 `/api/jobs/{id}`。
5. 增加架构测试：检查 API client、路由可用性、关键 payload 字段稳定性。
