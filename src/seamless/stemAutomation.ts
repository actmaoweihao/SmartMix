import type { TransitionRecommendation, VocalConflictScore } from "../analysis/types";
import type { StemAutomationPlan, StemPaths } from "../audio/types";
import { createBassSwapAutomation } from "./bassSwapAutomation";
import { reduceVocalConflictInMix } from "./vocalConflictReduction";

export function planStemMixAutomation(
  outgoingStems: StemPaths,
  incomingStems: StemPaths,
  vocalConflict: VocalConflictScore,
  rec: TransitionRecommendation,
): StemAutomationPlan {
  void outgoingStems;
  void incomingStems;
  const overlap = Math.max(0.5, rec.overlapDuration || 4);
  const bass = createBassSwapAutomation(rec.outgoingCue, rec.incomingCue, overlap);
  const base: StemAutomationPlan = {
    vocals: [
      { stem: "vocals", startTime: 0, endTime: overlap * 0.45, startGain: 1, endGain: 0.35, curve: "smoothstep" },
      { stem: "vocals", startTime: overlap * 0.45, endTime: overlap, startGain: 0.35, endGain: 1, curve: "smoothstep" },
    ],
    drums: [
      { stem: "drums", startTime: 0, endTime: overlap, startGain: 1, endGain: rec.method === "quick_cut" ? 0.1 : 0.75, curve: "equal_power" },
      { stem: "drums", startTime: 0, endTime: overlap, startGain: rec.method === "quick_cut" ? 0.1 : 0.55, endGain: 1, curve: "equal_power" },
    ],
    bass: [...bass.outgoingLowGain, ...bass.incomingLowGain],
    other: [
      { stem: "other", startTime: 0, endTime: overlap, startGain: 1, endGain: 0.2, curve: "equal_power" },
      { stem: "other", startTime: 0, endTime: overlap, startGain: 0.25, endGain: 1, curve: "equal_power" },
    ],
    filters: [
      { target: "incoming", type: "lowshelf", startTime: 0, endTime: overlap * 0.62, startValue: -18, endValue: -12 },
      { target: "outgoing", type: "highpass", startTime: overlap * 0.65, endTime: overlap, startValue: 60, endValue: 220 },
    ],
    effects: rec.method === "echo_out" ? [{ type: "echo", target: "outgoing", startTime: Math.max(0, overlap - 2), endTime: overlap, value: "1 beat" }] : [],
  };
  return reduceVocalConflictInMix(base, vocalConflict);
}
