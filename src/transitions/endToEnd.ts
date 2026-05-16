import type { TrackAnalysis, TransitionContext, TransitionRecommendation } from "../analysis/types";
import { buildRecommendation } from "./common";

export function buildEndToEnd(outgoing: TrackAnalysis, incoming: TrackAnalysis, context: TransitionContext = {}): TransitionRecommendation {
  return buildRecommendation({
    outgoing,
    incoming,
    context,
    method: "end_to_end",
    difficulty: 1,
    outgoingRoles: ["outro", "exit", "chorus"],
    incomingRoles: ["entry", "drop", "chorus"],
    overlapDuration: 0,
    methodFit: 0.82,
    reason: "首尾切换适合 A 歌结尾干净、B 歌开头强拍清楚的场景，几乎不依赖 BPM 或调性。",
    steps: [
      { atBeatOffset: -4, atTimeOffset: -2, action: "set_cue", targetDeck: "B", explanation: "把 B 歌准备在第一拍。" },
      { atBeatOffset: 0, atTimeOffset: 0, action: "press_play", targetDeck: "B", explanation: "A 歌乐句结束后的强拍启动 B 歌。" },
      { atBeatOffset: 0, atTimeOffset: 0, action: "stop_track", targetDeck: "A", explanation: "停止 A 歌，保持强拍感，不做长时间叠加。" },
    ],
  });
}
