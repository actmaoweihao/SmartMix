# SmartMix

SmartMix 是一个本地运行的智能混音工作台。前端由 pnpm + Vite 管理，后端使用 FastAPI + librosa 分析音频，并通过 imageio-ffmpeg 导出 MP3/WAV。

## 启动

```bash
pnpm install
pnpm setup:backend
pnpm dev
```

访问：

- 前端：http://127.0.0.1:3000
- 后端：http://127.0.0.1:8001/api/health

## 已实现

- 多音频上传，后端保存并用 librosa 分析 BPM、调性、能量、首尾低能量区和波形峰值。
- AI 精准混音分析：输出 beat grid、4/4 小节线、16 拍 phrase 线、自动入点/出点候选和响度指标。
- 曲目排序：综合推荐、谐和优先、BPM 升/降序、能量弧线、原始顺序。
- 波形视图：选择曲目后显示波形，拖动 IN/OUT 手柄调整过渡点。
- 播放进度条：拖动全局混音进度条可跳转预览；点击波形可跳到选中曲目的位置。
- Web Audio 预览：支持交叉淡入淡出、EQ 调整、低通扫尾/高通抬入滤波过渡。
- 后端导出：按当前曲目顺序、过渡点、EQ、节拍同步开关导出 MP3 或 WAV。
- AI Precision 导出：按 4/8/16 小节计算重叠区间，使用等功率淡化、动态 EQ 频率避让和响度归一化。
- 项目保存/加载：保存曲目顺序、过渡点和混音设置到本地后端数据目录。

## 项目结构

```text
backend/
  analysis.py      # librosa 分析
  main.py          # FastAPI API
  mixing.py        # 后端混音、节拍同步、MP3/WAV 导出
  storage.py       # 本地文件与项目 JSON 存储
src/
  main.js          # 前端状态、上传、波形、播放、导出
  styles.css       # 工作台 UI
```

## 说明

Essentia 没有默认纳入依赖，因为它在 Windows 本地安装成本较高。当前后端使用 librosa 完成核心分析；以后可以在 `backend/analysis.py` 中增加 Essentia 可选分支。

导出的 MP3 使用 `imageio-ffmpeg` 自带的 ffmpeg 二进制，不要求系统提前安装 ffmpeg。

当前“AI”能力采用可本地运行的算法流水线实现：librosa beat tracking/chroma、规则化音乐结构候选、tempo stretch、动态 EQ、响度归一化。后续如需更接近工业级深度学习方案，可继续接入 madmom、Demucs、Essentia 或 PyTorch 结构分割模型。
