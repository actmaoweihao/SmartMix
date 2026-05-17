import { scoreBpmCompatibility } from "../analysis/bpm";
import { energyFlowScore } from "../analysis/energy";
import { scoreKeyCompatibility } from "../analysis/key";
import { bestCue, phraseAlignmentScore } from "../analysis/phrase";
import { scoreStyleCompatibility } from "../analysis/style";
import { detectVocalConflict } from "../analysis/vocal";
import type { CuePoint, DJActionStep, TrackAnalysis, TransitionContext, TransitionMethod, TransitionRecommendation } from "../analysis/types";

export type BuildInput = {
  outgoing: TrackAnalysis;
  incoming: TrackAnalysis;
  context: TransitionContext;
  method: TransitionMethod;
  difficulty: 1 | 2 | 3 | 4 | 5;
  outgoingRoles: CuePoint["role"][];
  incomingRoles: CuePoint["role"][];
  overlapDuration: number;
  methodFit: number;
  reason: string;
  steps: DJActionStep[];
};

export function buildRecommendation(input: BuildInput): TransitionRecommendation {
  const outgoingCue = bestCue(input.outgoing, input.outgoingRoles);
  const incomingCue = bestCue(input.incoming, input.incomingRoles);
  const bpm = scoreBpmCompatibility(input.outgoing, input.incoming);
  const key = scoreKeyCompatibility(input.outgoing, input.incoming, input.context);
  const phrase = phraseAlignmentScore(outgoingCue, incomingCue);
  const vocal = detectVocalConflict(input.outgoing, input.incoming, outgoingCue.time, incomingCue.time, input.overlapDuration);
  const energy = energyFlowScore(input.outgoing, input.incoming, input.context.targetEnergy ?? "keep");
  const style = scoreStyleCompatibility(input.outgoing, input.incoming);
  const beginner = beginnerSuitability(input.difficulty, input.context);
  const section = sectionSuitability(input.method, outgoingCue, incomingCue, input.context);
  const weighted =
    0.23 * (1 - vocal.score) +
    0.18 * phrase +
    0.13 * beginner +
    0.14 * energy +
    0.1 * bpm.score +
    0.1 * key.score +
    0.07 * style.score +
    0.05 * section;
  const complexityPenalty = input.context.maxComplexity && input.difficulty > input.context.maxComplexity ? 0.35 : 0;
  const score = clamp01(weighted * input.methodFit - complexityPenalty);
  const debugReasons = [
    `bpm=${bpm.category}`,
    `key=${key.relation}`,
    `style=${style.relation}`,
    `keyScore=${Math.round(key.score * 100)}%`,
    `styleScore=${Math.round(style.score * 100)}%`,
    `vocalConflict=${Math.round(vocal.score * 100)}%`,
    `phrase=${Math.round(phrase * 100)}%`,
    `section=${Math.round(section * 100)}%`,
  ];
  return {
    method: input.method,
    score: round(score),
    difficulty: input.difficulty,
    outgoingCue,
    incomingCue,
    overlapDuration: input.overlapDuration,
    reason: `${input.reason} BPM: ${bpm.category}, 调性: ${key.relation}, 风格: ${style.relation}, 人声冲突: ${Math.round(vocal.score * 100)}%。`,
    stepByStep: input.steps,
    debug: {
      method: input.method,
      finalScore: round(score),
      bpmScore: round(bpm.score),
      keyScore: round(key.score),
      vocalConflictScore: round(vocal.score),
      phraseScore: round(phrase),
      energyScore: round(energy),
      styleScore: round(style.score),
      beginnerScore: round(beginner),
      beginnerPenalty: round(complexityPenalty),
      sectionSuitability: round(section),
      methodFit: round(input.methodFit),
      reasons: debugReasons,
    },
  };
}

export function beginnerSuitability(difficulty: number, context: TransitionContext): number {
  if (!context.beginnerMode) return 1;
  return ({ 1: 1, 2: 0.92, 3: 0.72, 4: 0.42, 5: 0.15 }[difficulty] ?? 0.5);
}

export function phraseSeconds(track: TrackAnalysis, bars: number): number {
  return bars * 4 * (60 / Math.max(1, track.bpm));
}

export function clamp01(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

export function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function sectionSuitability(
  method: TransitionMethod,
  outgoingCue: CuePoint,
  incomingCue: CuePoint,
  context: TransitionContext,
): number {
  const outgoingClean = ["outro", "breakdown", "exit"].includes(outgoingCue.role) || ["outro", "breakdown", "bridge"].includes(outgoingCue.sectionType);
  const incomingClean = incomingCue.sectionType === "intro" || incomingCue.role === "entry";
  const incomingImpact = ["drop", "chorus"].includes(incomingCue.role) || ["drop", "chorus"].includes(incomingCue.sectionType);

  if (method === "beatmix" || method === "bass_swap") {
    return clamp01(0.35 + (outgoingClean ? 0.28 : -0.12) + (incomingClean ? 0.28 : -0.18) + (incomingImpact ? -0.08 : 0));
  }
  if (method === "quick_cut") {
    return clamp01(0.55 + (incomingImpact ? 0.25 : 0.05) + (["chorus", "breakdown", "outro"].includes(outgoingCue.role) ? 0.12 : 0));
  }
  if (method === "echo_out") {
    return clamp01(0.62 + (["chorus", "breakdown", "outro"].includes(outgoingCue.role) ? 0.18 : 0) + (incomingImpact ? 0.08 : 0));
  }
  if (method === "breakdown_switch") {
    return clamp01(0.45 + (outgoingCue.sectionType === "breakdown" ? 0.38 : 0) + (incomingClean || incomingImpact ? 0.12 : 0));
  }
  if (method === "wide_bpm_loop") {
    return clamp01(0.35 + (outgoingClean ? 0.2 : -0.18) - (context.beginnerMode ? 0.2 : 0));
  }
  if (method === "fade" || method === "end_to_end") {
    return clamp01(0.68 + (outgoingClean ? 0.12 : 0) + (incomingClean || incomingImpact ? 0.08 : 0));
  }
  return 0.5;
}
