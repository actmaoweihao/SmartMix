import type { TrackAnalysis, TransitionContext, TransitionRecommendation } from "../analysis/types";
import { buildRecommendation, phraseSeconds } from "./common";

export function buildBeatmix(outgoing: TrackAnalysis, incoming: TrackAnalysis, context: TransitionContext = {}): TransitionRecommendation {
  return buildRecommendation({
    outgoing,
    incoming,
    context,
    method: "beatmix",
    difficulty: 3,
    outgoingRoles: ["outro", "breakdown", "exit"],
    incomingRoles: ["entry"],
    overlapDuration: phraseSeconds(outgoing, 8),
    methodFit: 0.92,
    reason: "对拍混音适合 BPM 接近、调性兼容且人声不密集的 intro/outro。",
    steps: [
      { atBeatOffset: -32, atTimeOffset: -phraseSeconds(outgoing, 8), action: "set_cue", targetDeck: "B", explanation: "把 B 歌前奏第一拍对准 A 歌尾奏乐句。" },
      { atBeatOffset: 0, atTimeOffset: 0, action: "press_play", targetDeck: "B", explanation: "A 歌尾奏开始时播放 B 歌。" },
      { atBeatOffset: 0, atTimeOffset: 0, action: "eq_low_cut", targetDeck: "B", value: 0, explanation: "先切掉 B 歌低频，避免两个底鼓打架。" },
      { atBeatOffset: 32, atTimeOffset: phraseSeconds(outgoing, 8), action: "eq_low_swap", targetDeck: "B", value: 1, explanation: "到下一个乐句交换低频，B 歌接管节奏。" },
    ],
  });
}
