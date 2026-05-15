import type { TrackAnalysis, TransitionContext, TransitionRecommendation } from "../analysis/types";
import { buildRecommendation, phraseSeconds } from "./common";

export function buildBassSwap(outgoing: TrackAnalysis, incoming: TrackAnalysis, context: TransitionContext = {}): TransitionRecommendation {
  return buildRecommendation({
    outgoing,
    incoming,
    context,
    method: "bass_swap",
    difficulty: 3,
    outgoingRoles: ["outro", "breakdown", "exit"],
    incomingRoles: ["entry", "drop"],
    overlapDuration: phraseSeconds(outgoing, 8),
    methodFit: 0.9,
    reason: "低频替换适合鼓点稳定但低频会撞的两首歌。",
    steps: [
      { atBeatOffset: 0, atTimeOffset: 0, action: "press_play", targetDeck: "B", explanation: "让 B 歌从乐句第一拍进入。" },
      { atBeatOffset: 0, atTimeOffset: 0, action: "eq_low_cut", targetDeck: "B", value: 0, explanation: "B 歌低频先关掉，只露出中高频和节奏提示。" },
      { atBeatOffset: 24, atTimeOffset: phraseSeconds(outgoing, 6), action: "eq_low_swap", targetDeck: "A", value: 0, explanation: "下一段前快速切掉 A 歌低频。" },
      { atBeatOffset: 32, atTimeOffset: phraseSeconds(outgoing, 8), action: "eq_low_swap", targetDeck: "B", value: 1, explanation: "同一乐句边界打开 B 歌低频。" },
    ],
  });
}
