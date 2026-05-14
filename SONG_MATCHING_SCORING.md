# 两首歌衔接匹配评分算法

## 1. 实现位置

后端：

- `backend/matching.py`
- `backend/main.py` 的 `POST /api/match`

前端：

- `src/main.js` 的 `calculatePairMatch()`
- 页面里的 “Pair Match / 两首歌衔接匹配评分” 面板

## 2. API 用法

接口：

```text
POST http://localhost:8001/api/match
```

表单字段：

```text
file_a: 第一首音频
file_b: 第二首音频
```

返回内容包括：

- 两首歌的 BPM、Key、Camelot、Energy、Duration
- A → B 方向评分
- B → A 方向评分
- 推荐衔接方向
- 总分和评级
- Camelot、BPM、Energy、Structure 四个分项

## 3. 总分公式

当前总分为 0-100，越高越适合衔接。

```text
total_score = 0.45 * camelot_score
            + 0.30 * bpm_score
            + 0.15 * energy_score
            + 0.10 * structure_score
```

权重解释：

- Camelot 调性：45%
- BPM 节奏：30%
- 能量：15%
- 结构可过渡性：10%

## 4. Camelot 调性评分

附件方案已落地到 `backend/matching.py`。

核心规则：

- 同 Camelot 数字 A/B 互转：距离 0，完美
- 同 mode 相邻：按轮盘数字距离评分
- 不同 mode 且数字不同：距离加 2 惩罚
- 纯五度关系：距离减 1 加分
- 数字距离大于等于 6：距离加 1 惩罚

距离越低越兼容。

评级：

```text
distance = 0      完美
distance = 1-2    推荐
distance = 3-4    可用
distance >= 5     避坑
```

再把距离换算成 0-100 分：

```text
0 -> 100
1 -> 92
2 -> 84
3 -> 72
4 -> 62
>=5 -> 继续递减
```

## 5. BPM 评分

BPM 评分会自动考虑 half/double tempo：

```text
候选 BPM_B = BPM_B, BPM_B * 2, BPM_B / 2
选择最接近 BPM_A 的候选值
```

然后计算差值百分比：

```text
pct = abs(BPM_A - normalized_BPM_B) / BPM_A
bpm_score = 100 - pct * 420
```

分数会限制在 0-100。

## 6. 能量评分

能量来自后端分析的 RMS/峰值综合值，通常在 0-1。

```text
energy_score = 100 - abs(energy_A - energy_B) * 120
```

能量越接近，分数越高。

## 7. 结构可过渡性评分

结构评分会尝试估算从上一首接到下一首时，最多能容纳几小节重叠：

- 上一首可用尾段：`duration - outro_candidate`
- 下一首可用头段：`intro_candidate`
- 歌曲长度限制：较短歌曲的 35%

取三者最小值作为可用重叠时间。

然后按 BPM 推算是否能容纳：

- 16 小节
- 8 小节
- 4 小节

评分：

```text
>=24s  96
>=16s  88
>=8s   76
>=4s   58
<4s    35
```

## 8. 方向性

衔接是有方向的，所以接口会分别计算：

```text
A → B
B → A
```

区别主要来自结构分项：

- A 的尾奏是否适合出
- B 的前奏是否适合入
- 反过来可能完全不同

接口会自动选择总分更高的方向作为推荐方向。

## 9. 当前排序也已改进

前端综合排序里的 `keyDistance()` 也已经从旧的半音距离改为 Camelot Wheel 距离。

因此“综合推荐”排序现在的调性评分和两歌匹配评分使用同一套 DJ harmonic mixing 逻辑。

