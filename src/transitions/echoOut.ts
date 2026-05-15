import type { TrackAnalysis, TransitionContext, TransitionRecommendation } from "../analysis/types";
import { buildRecommendation, phraseSeconds } from "./common";

export function buildEchoOut(outgoing: TrackAnalysis, incoming: TrackAnalysis, context: TransitionContext = {}): TransitionRecommendation {
  return buildRecommendation({
    outgoing,
    incoming,
    context,
    method: "echo_out",
    difficulty: 2,
    outgoingRoles: ["chorus", "breakdown", "outro", "exit"],
    incomingRoles: ["entry", "drop", "chorus"],
    overlapDuration: 2,
    methodFit: 0.88,
    reason: "Echo Out 用回声尾巴掩盖调性、人声或 BPM 冲突，适合流行歌救场。",
    steps: [
      { atBeatOffset: -4, atTimeOffset: -4 * (60 / outgoing.bpm), action: "enable_echo", targetDeck: "A", value: "1 beat", explanation: "A 歌乐句尾打开 1 拍回声。" },
      { atBeatOffset: -2, atTimeOffset: -2 * (60 / outgoing.bpm), action: "fade_out", targetDeck: "A", value: 0.2, explanation: "迅速降低 A 歌，让回声尾巴留下空间。" },
      { atBeatOffset: 0, atTimeOffset: 0, action: "press_play", targetDeck: "B", explanation: "在回声尾巴里从 B 歌乐句强拍进入。" },
      { atBeatOffset: 4, atTimeOffset: 4 * (60 / incoming.bpm), action: "disable_echo", targetDeck: "A", explanation: "B 歌站稳后关闭 A 歌回声。" },
    ],
  });
}
