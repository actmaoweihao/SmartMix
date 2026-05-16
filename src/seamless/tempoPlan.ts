import { normalizedBpmDiff } from "../analysis/bpm";
import type { SeamlessTransitionOptions, TempoAdjustmentPlan } from "../audio/types";

export function computeTempoAdjustment(
  outgoingBpm: number,
  incomingBpm: number,
  options: SeamlessTransitionOptions = {},
): TempoAdjustmentPlan {
  const targetBpm = outgoingBpm || incomingBpm || 120;
  const normalizedDiff = normalizedBpmDiff(targetBpm, incomingBpm);
  const percent = Math.abs(normalizedDiff / Math.max(targetBpm, 1)) * 100;
  const maxPercent = options.maxTempoChangePercent ?? 10;
  const stretchRatio = round(targetBpm / compatibleIncomingBpm(targetBpm, incomingBpm));
  if (percent <= 6 && percent <= maxPercent) {
    return { shouldStretch: percent > 0.25, targetBpm, stretchRatio, tempoChangePercent: round(percent), risk: "low" };
  }
  if (percent <= 10 && percent <= maxPercent && options.targetMode === "quality") {
    return { shouldStretch: true, targetBpm, stretchRatio, tempoChangePercent: round(percent), risk: "medium" };
  }
  return { shouldStretch: false, targetBpm, stretchRatio: 1, tempoChangePercent: round(percent), risk: "high" };
}

function compatibleIncomingBpm(targetBpm: number, incomingBpm: number): number {
  const variants = [incomingBpm, incomingBpm / 2, incomingBpm * 2].filter((value) => value > 0);
  return variants.reduce((best, value) => (Math.abs(value - targetBpm) < Math.abs(best - targetBpm) ? value : best), variants[0] ?? targetBpm);
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
