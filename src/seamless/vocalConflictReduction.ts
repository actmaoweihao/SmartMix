import type { VocalConflictScore } from "../analysis/types";
import type { StemAutomationPlan } from "../audio/types";

export function reduceVocalConflictInMix(plan: StemAutomationPlan, vocalConflict: VocalConflictScore): StemAutomationPlan {
  if (vocalConflict.score < 0.35) return plan;
  const overlapEnd = Math.max(...plan.vocals.map((item) => item.endTime), 4);
  const outgoingEnd = Math.min(overlapEnd, vocalConflict.score > 0.65 ? overlapEnd * 0.18 : overlapEnd * 0.35);
  return {
    ...plan,
    vocals: [
      { stem: "vocals", startTime: 0, endTime: outgoingEnd, startGain: 1, endGain: 0.08, curve: "smoothstep" },
      { stem: "vocals", startTime: outgoingEnd, endTime: overlapEnd, startGain: 0, endGain: 1, curve: "smoothstep" },
    ],
    effects: [
      ...plan.effects,
      ...(vocalConflict.score > 0.62 ? [{ type: "echo" as const, target: "outgoing" as const, startTime: outgoingEnd * 0.5, endTime: outgoingEnd + 1, value: "1/2 beat" }] : []),
    ],
  };
}
