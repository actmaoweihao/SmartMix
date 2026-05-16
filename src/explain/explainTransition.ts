import type { TransitionRecommendation } from "../analysis/types";

export function explainTransition(rec: TransitionRecommendation): string {
  const conflict = rec.debug?.vocalConflictScore ?? 0;
  const why = conflict > 0.42
    ? `${rec.reason} 这段衔接里两首歌都有明显人声，长时间叠加容易让歌词打架，所以更适合缩短 overlap 或改用效果器过渡。`
    : rec.reason;
  const exit = `A 歌建议从 ${formatTime(rec.outgoingCue.time)} 的${sectionName(rec.outgoingCue.sectionType)}位置退出。`;
  const entry = `B 歌建议从 ${formatTime(rec.incomingCue.time)} 的${sectionName(rec.incomingCue.sectionType)}强拍进入。`;
  const operations = rec.stepByStep
    .map((step) => `${formatOffset(step.atBeatOffset)}，${step.targetDeck} Deck ${actionName(step.action)}，${step.explanation}`)
    .slice(0, 5)
    .join(" ");
  return `推荐用「${methodName(rec.method)}」。为什么这样接：${why}${exit}${entry}具体操作：${operations}${riskHint(rec)}`;
}

function methodName(method: string): string {
  return {
    fade: "渐隐",
    end_to_end: "首尾切换",
    quick_cut: "快切",
    beatmix: "对拍混音",
    bass_swap: "低频替换",
    echo_out: "Echo Out",
    filter_sweep: "滤波扫频",
    breakdown_switch: "空拍切换",
    wide_bpm_loop: "大 BPM 差 Loop",
    loop_build: "Loop Build",
    instrumental_bridge: "伴奏过桥",
    acapella_mashup: "清唱叠加",
  }[method] ?? method;
}

function riskHint(rec: TransitionRecommendation): string {
  if (rec.method === "beatmix" || rec.method === "bass_swap") return "翻车风险：两边低频不要同时全开；先关 B 歌低频，到下一个乐句边界再交换。";
  if (rec.method === "quick_cut") return "翻车风险：不要提前放 B 歌；一定要在乐句第一拍切，避免两段人声重叠。";
  if (rec.method === "echo_out") return "翻车风险：Echo 尾巴不要太长；B 歌一进来就把 A 歌音量收干净。";
  if (rec.method === "breakdown_switch") return "翻车风险：空拍里不要叠太多旋律；滤波要收干净再让 B 歌进入。";
  return "翻车风险：听到两首歌主唱同时出现时，缩短 overlap 或改用快切。";
}

function formatTime(time: number): string {
  const min = Math.floor(time / 60);
  const sec = Math.round(time % 60).toString().padStart(2, "0");
  return `${min}:${sec}`;
}

function sectionName(section: string): string {
  return {
    intro: "前奏",
    verse: "主歌",
    chorus: "副歌",
    bridge: "桥段",
    breakdown: "空拍/弱化段",
    drop: "Drop",
    outro: "尾奏",
  }[section] ?? section;
}

function actionName(action: string): string {
  return {
    press_play: "按播放",
    set_cue: "设好 Cue",
    start_loop: "打开 Loop",
    halve_loop: "把 Loop 减半",
    increase_filter: "加滤波",
    decrease_filter: "收滤波",
    filter_sweep: "做滤波扫频",
    enable_echo: "打开 Echo",
    disable_echo: "关闭 Echo",
    fade_out: "把音量推低",
    fade_in: "把音量推上来",
    eq_low_cut: "切掉低频",
    eq_low_swap: "交换低频",
    crossfader_move: "移动 crossfader",
    stop_track: "停掉这一轨",
  }[action] ?? action;
}

function formatOffset(beats: number): string {
  if (beats === 0) return "在切换点这一拍";
  return beats < 0 ? `提前 ${Math.abs(beats)} 拍` : `切换后 ${beats} 拍`;
}
