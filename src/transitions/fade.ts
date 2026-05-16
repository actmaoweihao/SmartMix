import type { TrackAnalysis, TransitionContext, TransitionRecommendation } from "../analysis/types";
import { buildRecommendation, phraseSeconds } from "./common";

export function buildFade(outgoing: TrackAnalysis, incoming: TrackAnalysis, context: TransitionContext = {}): TransitionRecommendation {
  return buildRecommendation({
    outgoing,
    incoming,
    context,
    method: "fade",
    difficulty: 1,
    outgoingRoles: ["chorus", "outro", "exit"],
    incomingRoles: ["entry", "chorus", "drop"],
    overlapDuration: Math.min(4, phraseSeconds(outgoing, 4)),
    methodFit: context.beginnerMode ? 0.98 : 0.78,
    reason: "渐隐适合新手、BPM 或调性不稳定，以及不想让两首歌长时间叠在一起的场景。",
    steps: [
      { atBeatOffset: -32, atTimeOffset: -phraseSeconds(outgoing, 8), action: "set_cue", targetDeck: "B", explanation: "先把 B 歌 cue 到下一个乐句强拍。" },
      { atBeatOffset: -16, atTimeOffset: -phraseSeconds(outgoing, 4), action: "fade_out", targetDeck: "A", value: 0.45, explanation: "A 歌开始在 4 小节内慢慢降低音量。" },
      { atBeatOffset: 0, atTimeOffset: 0, action: "press_play", targetDeck: "B", explanation: "到乐句第一拍按下 B 歌播放。" },
      { atBeatOffset: 4, atTimeOffset: 4 * (60 / incoming.bpm), action: "crossfader_move", targetDeck: "B", value: 1, explanation: "把交叉推子推到 B，完成切换。" },
    ],
  });
}
