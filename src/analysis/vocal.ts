import { averageRange } from "./energy";
import type { TrackAnalysis, VocalConflictScore } from "./types";

export function detectVocalConflict(
  outgoing: TrackAnalysis,
  incoming: TrackAnalysis,
  outgoingStartTime: number,
  incomingStartTime: number,
  overlapDuration: number,
): VocalConflictScore {
  const step = 2;
  const conflictRegions: VocalConflictScore["conflictRegions"] = [];
  let total = 0;
  let count = 0;

  for (let offset = 0; offset < overlapDuration; offset += step) {
    const outgoingDensity = averageRange(outgoing, outgoingStartTime + offset, step, "vocal");
    const incomingDensity = averageRange(incoming, incomingStartTime + offset, step, "vocal");
    const severity = Math.sqrt(outgoingDensity * incomingDensity);
    total += severity;
    count += 1;
    if (severity >= 0.45) {
      conflictRegions.push({ start: offset, end: Math.min(overlapDuration, offset + step), severity: round(severity) });
    }
  }

  const score = round(count ? total / count : 0);
  return { score, conflictRegions: mergeRegions(conflictRegions), recommendation: recommendationFor(score, incomingStartTime, outgoingStartTime) };
}

function recommendationFor(score: number, incomingStartTime: number, outgoingStartTime: number): VocalConflictScore["recommendation"] {
  if (score < 0.22) return "safe_to_blend";
  if (score < 0.42) return "use_eq_cut";
  if (score < 0.62) return incomingStartTime < 16 || outgoingStartTime > 60 ? "shorten_overlap" : "use_echo_out";
  if (score < 0.78) return "use_quick_cut";
  return "use_instrumental_bridge";
}

function mergeRegions(regions: VocalConflictScore["conflictRegions"]): VocalConflictScore["conflictRegions"] {
  const merged: VocalConflictScore["conflictRegions"] = [];
  for (const region of regions) {
    const last = merged.at(-1);
    if (last && region.start <= last.end) {
      last.end = region.end;
      last.severity = round(Math.max(last.severity, region.severity));
    } else {
      merged.push({ ...region });
    }
  }
  return merged;
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}
