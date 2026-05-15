import type { BpmScore, TrackAnalysis } from "./types";

export function scoreBpmCompatibility(a: TrackAnalysis, b: TrackAnalysis): BpmScore {
  const bpmDiff = normalizedBpmDiff(a.bpm, b.bpm);
  if (bpmDiff <= 3) return { score: 1 - bpmDiff / 24, bpmDiff, category: "same", suggestedMethod: "beatmix" };
  if (bpmDiff <= 8) return { score: 0.86 - (bpmDiff - 3) / 25, bpmDiff, category: "close", suggestedMethod: "quick_cut" };
  if (bpmDiff <= 20) return { score: 0.62 - (bpmDiff - 8) / 36, bpmDiff, category: "medium", suggestedMethod: "echo_out" };
  return { score: Math.max(0.08, 0.25 - (bpmDiff - 20) / 80), bpmDiff, category: "wide", suggestedMethod: "wide_bpm_loop" };
}

export function normalizedBpmDiff(a: number, b: number): number {
  const variants = [b, b / 2, b * 2].filter((value) => value > 0);
  return Math.min(...variants.map((value) => Math.abs(a - value)));
}
