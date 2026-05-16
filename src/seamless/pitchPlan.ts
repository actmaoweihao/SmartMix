import { scoreKeyCompatibility } from "../analysis/key";
import type { TrackAnalysis } from "../analysis/types";
import type { PitchShiftPlan, SeamlessTransitionOptions } from "../audio/types";

const CAMELOT_TO_PC: Record<string, number> = {
  "8A": 9, "8B": 0, "9A": 4, "9B": 7, "10A": 11, "10B": 2, "11A": 6, "11B": 9,
  "12A": 1, "12B": 4, "1A": 8, "1B": 11, "2A": 3, "2B": 6, "3A": 10, "3B": 1,
  "4A": 5, "4B": 8, "5A": 0, "5B": 3, "6A": 7, "6B": 10, "7A": 2, "7B": 5,
};

export function computePitchShiftForHarmonicMixing(
  outgoingKey: string,
  incomingKey: string,
  options: SeamlessTransitionOptions = {},
): PitchShiftPlan {
  const maxShift = options.maxPitchShiftSemitones ?? 2;
  const outgoing = trackForKey("out", outgoingKey);
  const incoming = trackForKey("in", incomingKey);
  const current = scoreKeyCompatibility(outgoing, incoming);
  if (current.score >= 0.82) {
    return { shouldPitchShift: false, targetKey: incoming.camelotKey ?? incoming.key, semitones: 0, expectedCamelotRelation: current.relation, risk: "low" };
  }
  const outPc = CAMELOT_TO_PC[outgoing.camelotKey ?? ""];
  const inPc = CAMELOT_TO_PC[incoming.camelotKey ?? ""];
  if (!Number.isFinite(outPc) || !Number.isFinite(inPc)) {
    return { shouldPitchShift: false, targetKey: null, semitones: 0, expectedCamelotRelation: "unknown", risk: "high" };
  }
  const candidates = [-2, -1, 1, 2].filter((shift) => Math.abs(shift) <= maxShift);
  const best = candidates
    .map((shift) => {
      const shiftedPc = mod12(inPc + shift);
      const target = Object.entries(CAMELOT_TO_PC).find(([, pc]) => pc === shiftedPc)?.[0] ?? null;
      const relation = target ? scoreKeyCompatibility(outgoing, trackForKey("shifted", target)) : current;
      return { shift, target, relation };
    })
    .sort((a, b) => b.relation.score - a.relation.score || Math.abs(a.shift) - Math.abs(b.shift))[0];
  if (!best || !best.target || best.relation.score < 0.72 || Math.abs(best.shift) > maxShift) {
    return { shouldPitchShift: false, targetKey: null, semitones: 0, expectedCamelotRelation: current.relation, risk: "high" };
  }
  return {
    shouldPitchShift: true,
    targetKey: best.target,
    semitones: best.shift,
    expectedCamelotRelation: best.relation.relation,
    risk: Math.abs(best.shift) <= 1 ? "low" : "medium",
  };
}

function trackForKey(id: string, key: string): TrackAnalysis {
  return {
    id,
    title: id,
    duration: 1,
    bpm: 120,
    key,
    camelotKey: key.match(/^\d{1,2}[AB]$/i) ? key.toUpperCase() : undefined,
    energyCurve: [],
    sections: [],
    beatGrid: [],
  };
}

function mod12(value: number): number {
  return ((value % 12) + 12) % 12;
}
