# SmartMix

SmartMix 是一个浏览器端智能混音 MVP。它可以上传多首音频，估算 BPM、调性和能量，按策略重新排序，预览交叉淡入淡出混音，并导出 WAV。

## 快速开始

直接用浏览器打开 `index.html` 即可使用，不需要安装依赖。

建议使用最新版 Chrome、Edge 或 Firefox。浏览器能否解码某种音频格式取决于本机浏览器支持情况。

## 当前文件

- `SMARTMIX_PRODUCT_SPEC.md`：基于原 PDF 完善后的产品与技术规格。
- `index.html`：应用入口。
- `styles.css`：工作台界面样式。
- `app.js`：上传、分析、排序、预览和导出逻辑。

## 已实现

- 多音频上传和拖拽上传。
- 浏览器端音频解码。
- BPM、能量、粗略调性、首尾低能量区估算。
- 原始顺序、BPM 升序、BPM 降序、谐和优先、能量弧线、综合推荐排序。
- 手动上移、下移、删除曲目。
- Web Audio 实时预览混音。
- OfflineAudioContext 离线渲染并导出 WAV。

## 下一步建议

- 加入波形视图和可拖动过渡点。
- 将分析任务迁移到 FastAPI + librosa/essentia 后端。
- 增加 MP3 导出、项目保存、节拍同步和 EQ/filter 过渡。

