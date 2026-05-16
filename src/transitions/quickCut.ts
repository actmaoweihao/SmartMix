import type { TrackAnalysis, TransitionContext, TransitionRecommendation } from "../analysis/types";
import { buildRecommendation } from "./common";

export function buildQuickCut(outgoing: TrackAnalysis, incoming: TrackAnalysis, context: TransitionContext = {}): TransitionRecommendation {
  return buildRecommendation({
    outgoing,
    incoming,
    context,
    method: "quick_cut",
    difficulty: 2,
    outgoingRoles: ["chorus", "breakdown", "outro", "exit"],
    incomingRoles: ["drop", "chorus", "entry"],
    overlapDuration: 0.5,
    methodFit: 0.9,
    reason: "快切适合两边都有人声、调性有风险，或 BPM 差不适合长混的场景。",
    steps: [
      { atBeatOffset: -4, atTimeOffset: -2, action: "set_cue", targetDeck: "B", explanation: "B 歌 cue 到 drop、chorus 或乐句第一拍。" },
      { atBeatOffset: 0, atTimeOffset: 0, action: "crossfader_move", targetDeck: "B", value: 1, explanation: "在 A 歌乐句边界瞬间推到 B。" },
      { atBeatOffset: 0, atTimeOffset: 0, action: "press_play", targetDeck: "B", explanation: "同一拍启动 B 歌，避免两个人声叠唱。" },
      { atBeatOffset: 1, atTimeOffset: 60 / incoming.bpm, action: "stop_track", targetDeck: "A", explanation: "确认 B 歌稳定后停掉 A。" },
    ],
  });
}
