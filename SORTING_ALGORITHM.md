# SmartMix 综合排序算法说明

## 1. 当前实现位置

SmartMix 目前的“综合推荐”排序在前端实现，核心代码位于：

- `src/main.js`
- `sortTracks()`
- `greedySort()`
- `transitionScore()`
- `keyDistance()`

后端 `backend/analysis.py` 负责给每首歌提供排序所需的分析字段，包括 BPM、调性、大小调模式、能量等；前端拿到这些字段后完成排序。

## 2. 算法类型

当前综合排序不是深度学习模型，也不是全局最优求解器，而是一个基于加权评分函数的贪心最近邻算法。

它的目标是：从一首能量较低的歌开始，每一步都从剩余曲目中选择“和当前曲目过渡成本最低”的下一首歌。

可以把它理解成 DJ 自动排歌里的轻量版路径搜索：

```text
起点 = 能量最低的歌曲
while 还有未排序歌曲:
    从剩余歌曲里找一首和当前歌曲最接近的
    把它放到队列末尾
```

它追求的是局部连续性，不保证全局最优。

## 3. 输入依赖

每首曲目排序时主要依赖以下字段。

### 3.1 BPM

字段：

```js
track.bpm
```

来源：

- 优先来自后端 `backend/analysis.py`
- 后端使用 librosa 的 onset/tempo 分析
- 如果后端失败，前端会使用本地能量包络自相关做 fallback

用途：

- 计算两首歌速度差
- BPM 越接近，过渡评分越低

缺省值：

```js
function safeBpm(track) {
  return track.bpm || 120;
}
```

如果某首歌没有 BPM，系统暂时按 120 BPM 处理。

### 3.2 调性

字段：

```js
track.key_index
track.mode
```

`key_index` 是 0-11 的半音编号：

```text
0 = C
1 = C#
2 = D
...
11 = B
```

`mode` 是：

```text
major
minor
```

来源：

- 后端使用 chroma 特征和大小调模板匹配
- 前端 fallback 会用简化 pitch class 估算

用途：

- 计算两首歌的调性距离
- 调性越接近，过渡评分越低
- 大调/小调不同会增加少量惩罚

### 3.3 能量

字段：

```js
track.energy
```

来源：

- 后端根据 RMS 能量均值和峰值估算
- 前端 fallback 使用类似 RMS 能量包络

用途：

- 作为排序起点依据
- 作为相邻歌曲能量差评分的一部分

当前算法会先按能量升序排列，把能量最低的一首作为起点。

## 4. 综合评分公式

综合推荐模式使用 `transitionScore(a, b, "recommended")`。

代码：

```js
function transitionScore(a, b, mode) {
  const bpmDelta = Math.min(1, Math.abs(safeBpm(a) - safeBpm(b)) / 60);
  const harmonic = keyDistance(a, b);
  const energyDelta = Math.abs(a.energy - b.energy);
  if (mode === "harmonic") return harmonic * 0.6 + bpmDelta * 0.3 + energyDelta * 0.1;
  return bpmDelta * 0.55 + harmonic * 0.3 + energyDelta * 0.15;
}
```

综合推荐的评分公式：

```text
score = 0.55 * bpmDelta
      + 0.30 * harmonicDistance
      + 0.15 * energyDelta
```

分数越低，代表两首歌越适合相邻播放。

权重含义：

```text
BPM 差异：55%
调性距离：30%
能量差异：15%
```

这说明当前算法更重视节奏连续，其次重视谐和混音，最后才考虑能量衔接。

## 5. BPM 差异计算

代码：

```js
const bpmDelta = Math.min(1, Math.abs(safeBpm(a) - safeBpm(b)) / 60);
```

计算逻辑：

```text
bpmDelta = min(1, abs(BPM_A - BPM_B) / 60)
```

例子：

```text
120 BPM -> 128 BPM
abs(120 - 128) = 8
bpmDelta = 8 / 60 = 0.133
```

如果两首歌 BPM 差超过 60，会被截断为 1。

这个归一化让 BPM 差异落在 0-1 之间，方便和调性距离、能量差一起加权。

## 6. 调性距离计算

代码：

```js
function keyDistance(a, b) {
  if (a.key_index === null || b.key_index === null) return 0.5;
  const raw = Math.abs(a.key_index - b.key_index);
  const circular = Math.min(raw, 12 - raw) / 6;
  const modePenalty = a.mode === b.mode ? 0 : 0.15;
  return Math.min(1, circular + modePenalty);
}
```

计算逻辑：

```text
raw = abs(keyA - keyB)
circular = min(raw, 12 - raw) / 6
modePenalty = 0 if modeA == modeB else 0.15
keyDistance = min(1, circular + modePenalty)
```

这里使用的是 12 平均律的环形距离。

例如 C 到 B：

```text
C = 0
B = 11
raw = 11
circular = min(11, 12 - 11) / 6 = 1 / 6 = 0.166
```

因为 C 和 B 在 12 个半音环上只差 1 个半音，所以距离较小。

如果任意一首歌没有调性，算法返回中性距离：

```text
0.5
```

注意：当前算法不是 Camelot Wheel 完整实现，只是用半音环距离近似谐和程度。

## 7. 能量差计算

代码：

```js
const energyDelta = Math.abs(a.energy - b.energy);
```

能量值通常在 0-1 之间，因此能量差也天然接近 0-1。

能量差越小，说明两首歌强弱更接近，过渡评分越低。

当前能量差权重只有 15%，所以它更多是辅助项，不会强行压过 BPM 和调性。

## 8. 贪心排序流程

代码：

```js
function greedySort(tracks, mode) {
  const remaining = [...tracks].sort((a, b) => a.energy - b.energy);
  const result = [remaining.shift()];
  while (remaining.length) {
    const current = result[result.length - 1];
    let bestIndex = 0;
    let bestScore = Infinity;
    remaining.forEach((candidate, index) => {
      const score = transitionScore(current, candidate, mode);
      if (score < bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    });
    result.push(remaining.splice(bestIndex, 1)[0]);
  }
  return result;
}
```

流程拆解：

1. 复制曲目数组，避免直接修改原始输入。
2. 按能量从低到高排序。
3. 取能量最低的曲目作为第一首。
4. 以当前队尾曲目为基准，遍历所有剩余曲目。
5. 对每个候选曲目计算 `transitionScore`。
6. 选择分数最低的曲目作为下一首。
7. 重复，直到所有曲目排完。

## 9. 举例说明

假设当前队尾曲目是 A：

```text
A: 120 BPM, C Maj, energy 0.50
```

候选 B：

```text
B: 124 BPM, D Maj, energy 0.55
```

候选 C：

```text
C: 145 BPM, F# Min, energy 0.90
```

算法会分别计算：

```text
score(A, B)
score(A, C)
```

如果 B 的 BPM 更接近、调性距离更小、能量也更接近，那么 B 会被选为下一首。

算法不会提前考虑“选了 B 以后后面是否更难排”，这就是贪心算法的特征。

## 10. 和其他排序模式的关系

`sortTracks()` 根据用户选择决定排序方式：

```js
function sortTracks(tracks, mode) {
  const copy = [...tracks];
  if (mode === "original") return copy.sort((a, b) => state.originalOrder.indexOf(a.localId) - state.originalOrder.indexOf(b.localId));
  if (mode === "bpmAsc") return copy.sort((a, b) => safeBpm(a) - safeBpm(b));
  if (mode === "bpmDesc") return copy.sort((a, b) => safeBpm(b) - safeBpm(a));
  if (mode === "energyArc") return energyArc(copy);
  return greedySort(copy, mode === "harmonic" ? "harmonic" : "recommended");
}
```

不同模式：

```text
original    原始上传顺序
bpmAsc      BPM 升序
bpmDesc     BPM 降序
energyArc   能量弧线
harmonic    谐和优先
recommended 综合推荐
```

其中 `harmonic` 和 `recommended` 都使用 `greedySort()`，区别是评分权重不同。

谐和优先权重：

```text
score = 0.60 * harmonicDistance
      + 0.30 * bpmDelta
      + 0.10 * energyDelta
```

综合推荐权重：

```text
score = 0.55 * bpmDelta
      + 0.30 * harmonicDistance
      + 0.15 * energyDelta
```

## 11. 当前算法优点

- 实现简单，运行很快。
- 不需要服务端排序，前端交互即时响应。
- 对小型歌单足够直观。
- 可以同时考虑速度、调性和能量。
- 对后端分析失败的曲目也有 fallback 值，不容易整体崩溃。

## 12. 当前算法局限

### 12.1 不是全局最优

贪心算法每一步只选当前最优，不回溯。因此它可能错过整体更好的排列。

更高级方案可以把歌曲看成图，用 TSP/动态规划/beam search 寻找全局路径。

### 12.2 调性算法偏粗

当前 `keyDistance()` 只是半音环距离，没有完整考虑：

- Camelot Wheel
- relative major/minor
- perfect fifth
- energy-compatible harmonic mixing

因此它能提供近似参考，但不等于专业 DJ 调性排序。

### 12.3 没有使用 beat grid 和小节结构

虽然后端现在已经分析了 beats、bars、phrases，但综合排序目前还没有把它们纳入评分。

也就是说，排序只决定“哪首接哪首”，不直接判断“这两首是否能按 4/8/16 小节自然过渡”。

### 12.4 没有考虑人声冲突

当前排序没有分析 vocal density，也没有判断两首歌过渡段是否同时有人声。

更高级的排序可以加入：

- 人声检测
- 结构段落
- intro/outro 是否纯伴奏
- 过渡区频谱遮蔽程度

### 12.5 权重是固定的

目前综合推荐权重写死在代码里：

```text
BPM 0.55
调性 0.30
能量 0.15
```

不同风格音乐可能需要不同权重。例如：

- EDM 更重视 BPM 和 phrase
- Hip-hop 更重视 groove 和 vocal
- Pop 串烧更重视调性和副歌结构

## 13. 后续升级建议

### 13.1 引入全局路径优化

把每首歌作为节点，把 `transitionScore(a, b)` 作为边权重，然后寻找总成本最低的播放路径。

可选算法：

- nearest neighbor + 2-opt
- beam search
- dynamic programming
- genetic algorithm
- simulated annealing

### 13.2 加入 Camelot Wheel

把调性映射到 Camelot 编号，例如：

```text
8A, 8B, 9A, 7A
```

然后按 DJ harmonic mixing 规则计算调性兼容度。

### 13.3 加入 phrase compatibility

使用后端分析出的：

```js
track.bars
track.phrases
track.transition_candidates
```

增加评分项：

```text
phraseScore = 两首歌能否形成 4/8/16 小节重叠
structureScore = songA outro 是否适合出，songB intro 是否适合入
```

### 13.4 加入频谱冲突评分

对候选过渡区计算低频、中频、高频能量重叠，避免低频撞车或人声打架。

### 13.5 支持用户偏好权重

在 UI 中开放：

```text
BPM 连续性
调性和谐
能量曲线
结构稳定
随机探索
```

让用户按场景调整排序。

## 14. 总结

当前 SmartMix 的综合排序算法可以概括为：

```text
基于 BPM、调性、能量三项特征的加权评分 + 贪心最近邻路径构造
```

它的实际评分公式是：

```text
score = 0.55 * normalized_bpm_delta
      + 0.30 * harmonic_distance
      + 0.15 * energy_delta
```

它适合快速生成一个“听起来更顺”的初始歌单，但还不是专业级自动 DJ 排序。下一步应把后端已经具备的 beat grid、小节线、transition candidates、响度和频谱特征纳入排序评分，这样排序才能真正服务于 4/8/16 小节精准混音。

