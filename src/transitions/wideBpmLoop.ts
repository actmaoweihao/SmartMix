import type { TrackAnalysis, TransitionContext, TransitionRecommendation } from "../analysis/types";
import { buildRecommendation } from "./common";

export function buildWideBpmLoop(outgoing: TrackAnalysis, incoming: TrackAnalysis, context: TransitionContext = {}): TransitionRecommendation {
  return buildRecommendation({
    outgoing,
    incoming,
    context,
    method: "wide_bpm_loop",
    difficulty: 5,
    outgoingRoles: ["breakdown", "outro", "exit"],
    incomingRoles: ["entry", "drop"],
    overlapDuration: 8,
    methodFit: context.beginnerMode ? 0.45 : 0.8,
    reason: "大 BPM 差 Loop 用无人声片段缓慢变速，难度较高，但适合跨风格过渡。",
    steps: [
      { atBeatOffset: -16, atTimeOffset: -8, action: "start_loop", targetDeck: "A", value: "4 beats", explanation: "在 A 歌无人声片段设置 4 拍 loop。" },
      { atBeatOffset: -8, atTimeOffset: -4, action: "halve_loop", targetDeck: "A", value: "2 beats", explanation: "逐步收紧 loop，制造转场张力。" },
      { atBeatOffset: 0, atTimeOffset: 0, action: "press_play", targetDeck: "B", explanation: "当 loop 节奏接近 B 歌时，从 B 歌乐句第一拍进入。" },
      { atBeatOffset: 4, atTimeOffset: 2, action: "eq_low_swap", targetDeck: "B", value: 1, explanation: "打开 B 歌低频并释放 A 歌 loop。" },
    ],
  });
}
