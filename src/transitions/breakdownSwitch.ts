import type { TrackAnalysis, TransitionContext, TransitionRecommendation } from "../analysis/types";
import { buildRecommendation, phraseSeconds } from "./common";

export function buildBreakdownSwitch(outgoing: TrackAnalysis, incoming: TrackAnalysis, context: TransitionContext = {}): TransitionRecommendation {
  return buildRecommendation({
    outgoing,
    incoming,
    context,
    method: "breakdown_switch",
    difficulty: 3,
    outgoingRoles: ["breakdown", "outro", "exit"],
    incomingRoles: ["breakdown", "entry", "drop"],
    overlapDuration: Math.min(phraseSeconds(outgoing, 4), 8),
    methodFit: 0.84,
    reason: "空拍切换利用 A 歌能量下降的窗口，让跨 BPM 或跨风格衔接更自然。",
    steps: [
      { atBeatOffset: -16, atTimeOffset: -phraseSeconds(outgoing, 4), action: "filter_sweep", targetDeck: "A", value: "highpass", explanation: "A 歌进入空拍时做高通扫频，减少低频和旋律占用。" },
      { atBeatOffset: 0, atTimeOffset: 0, action: "press_play", targetDeck: "B", explanation: "在空拍乐句边界启动 B 歌。" },
      { atBeatOffset: 8, atTimeOffset: phraseSeconds(outgoing, 2), action: "fade_in", targetDeck: "B", value: 1, explanation: "B 歌逐渐成为主声部。" },
      { atBeatOffset: 16, atTimeOffset: phraseSeconds(outgoing, 4), action: "fade_out", targetDeck: "A", value: 0, explanation: "空拍后移出 A 歌。" },
    ],
  });
}
