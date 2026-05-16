import { scoreBpmCompatibility } from "../analysis/bpm";
import { energyAt, trackAverageEnergy } from "../analysis/energy";
import { scoreKeyCompatibility } from "../analysis/key";
import { detectVocalConflict } from "../analysis/vocal";
import type { TrackAnalysis, TransitionContext, TransitionRecommendation } from "../analysis/types";
import { buildBassSwap } from "./bassSwap";
import { buildBeatmix } from "./beatmix";
import { buildBreakdownSwitch } from "./breakdownSwitch";
import { buildEchoOut } from "./echoOut";
import { buildEndToEnd } from "./endToEnd";
import { buildFade } from "./fade";
import { buildQuickCut } from "./quickCut";
import { clamp01, round } from "./common";
import { buildWideBpmLoop } from "./wideBpmLoop";

export async function analyzeTrack(audioFile: File | Blob | TrackAnalysis): Promise<TrackAnalysis> {
  if (isTrackAnalysis(audioFile)) return audioFile;
  throw new Error("真实音频分析尚未接入。请先传入 mock metadata 或后端分析结果。");
}

export function recommendTransition(
  outgoing: TrackAnalysis,
  incoming: TrackAnalysis,
  context: TransitionContext = {},
): TransitionRecommendation[] {
  const recs = [
    buildFade(outgoing, incoming, context),
    buildEndToEnd(outgoing, incoming, context),
    buildQuickCut(outgoing, incoming, context),
    buildBeatmix(outgoing, incoming, context),
    buildBassSwap(outgoing, incoming, context),
    buildEchoOut(outgoing, incoming, context),
    buildBreakdownSwitch(outgoing, incoming, context),
    buildWideBpmLoop(outgoing, incoming, context),
  ].map((rec) => adjustForMethod(rec, outgoing, incoming, context));

  return recs
    .filter((rec) => !context.maxComplexity || rec.difficulty <= context.maxComplexity)
    .sort((a, b) => b.score - a.score);
}

export function recommendNextTracks(
  currentTrack: TrackAnalysis,
  candidates: TrackAnalysis[],
  options: { targetEnergy?: "up" | "down" | "keep"; beginnerMode?: boolean; maxComplexity?: 1 | 2 | 3 | 4 | 5; maxResults?: number } = {},
) {
  return candidates
    .filter((track) => track.id !== currentTrack.id)
    .map((track) => {
      const bestTransition = recommendTransition(currentTrack, track, options)[0];
      const bpm = scoreBpmCompatibility(currentTrack, track);
      const key = scoreKeyCompatibility(currentTrack, track, options);
      const energyDirection = trackAverageEnergy(track) >= trackAverageEnergy(currentTrack) ? "能量可上扬" : "能量会回落";
      return {
        track,
        totalScore: bestTransition.score,
        bestTransition,
        reasons: [`BPM ${bpm.category}`, key.explanation, energyDirection, bestTransition.reason],
      };
    })
    .sort((a, b) => b.totalScore - a.totalScore)
    .slice(0, options.maxResults ?? 5);
}

function adjustForMethod(
  rec: TransitionRecommendation,
  outgoing: TrackAnalysis,
  incoming: TrackAnalysis,
  context: TransitionContext,
): TransitionRecommendation {
  const bpm = scoreBpmCompatibility(outgoing, incoming);
  const key = scoreKeyCompatibility(outgoing, incoming, context);
  const vocal = detectVocalConflict(outgoing, incoming, rec.outgoingCue.time, rec.incomingCue.time, rec.overlapDuration || 4);
  const rawBpmDiff = Math.abs(outgoing.bpm - incoming.bpm);
  const isWideWithoutHalfTime = rawBpmDiff > 12 && bpm.bpmDiff > 3;
  const longBlend = ["beatmix", "bass_swap", "wide_bpm_loop", "loop_build", "acapella_mashup"].includes(rec.method);
  const shortOrEffect = ["fade", "end_to_end", "quick_cut", "echo_out"].includes(rec.method);
  const incomingCueEnergy = energyAt(incoming, rec.incomingCue.time);
  const outgoingCueEnergy = energyAt(outgoing, rec.outgoingCue.time);
  const reasons = [...(rec.debug?.reasons ?? [])];
  let score = rec.score;
  let beginnerPenalty = rec.debug?.beginnerPenalty ?? 0;

  if (longBlend && vocal.score > 0.35) {
    const penalty = vocal.score > 0.65 ? 0.5 : 0.3 + vocal.score * 0.22;
    score -= penalty;
    reasons.push("两首歌在叠加区都有人声，长时间叠加会让歌词打架");
  }
  if (["beatmix", "bass_swap"].includes(rec.method)) {
    if (bpm.category === "same" && vocal.score < 0.3) score += 0.08;
    if (key.score >= 0.82 && vocal.score < 0.3) score += 0.04;
    if (context.beginnerMode && bpm.category === "same" && key.score >= 0.82 && vocal.score < 0.22) {
      score += 0.38;
      reasons.push("BPM 和调性都稳且人声少，新手可以练简单对拍或低频替换");
    }
  }
  if (rec.method === "quick_cut") {
    if (vocal.score > 0.42) {
      score += 0.32;
      reasons.push("人声冲突高，快切能避免两段人声重叠");
    }
    if (bpm.category === "medium" || key.relation === "clash") score += 0.12;
    if (rec.outgoingCue.sectionType === "breakdown" && vocal.score < 0.25 && isWideWithoutHalfTime) score -= 0.18;
  }
  if (rec.method === "echo_out") {
    if (bpm.category === "medium" || key.relation === "clash") score += 0.2;
    if (vocal.score > 0.42) {
      score += 0.18;
      reasons.push("用 Echo 尾巴遮住人声或调性冲突");
    }
  }
  if (rec.method === "wide_bpm_loop") {
    if (bpm.category === "wide" && !context.beginnerMode) score += 0.9;
    if (context.beginnerMode) {
      score -= 0.48;
      beginnerPenalty += 0.48;
      reasons.push("新手模式中 Loop 变速难度太高，已降权");
    }
  }
  if (context.beginnerMode) {
    if (shortOrEffect) score += 0.12;
    if (rec.difficulty >= 4) {
      const penalty = rec.difficulty === 5 ? 0.28 : 0.18;
      score -= penalty;
      beginnerPenalty += penalty;
    }
  }
  if (isWideWithoutHalfTime) {
    if (["beatmix", "bass_swap"].includes(rec.method)) {
      score -= 0.4;
      reasons.push("BPM 差超过 12 且不是 half-time/double-time，不适合长时间对拍");
    }
    if (["echo_out", "quick_cut", "breakdown_switch"].includes(rec.method)) {
      score += 0.2;
      reasons.push("BPM 差较大，短切或效果器退出更稳定");
    }
  }
  if (key.relation === "clash") {
    reasons.push("调性不兼容，不建议长时间叠加");
    if (longBlend) score -= 0.36;
    if (shortOrEffect) {
      score += 0.14;
      reasons.push("调性不兼容时短过渡更稳");
    }
  }
  if (key.relation === "unknown") {
    reasons.push("未能识别调性，因此更建议使用短切或效果器过渡");
    if (longBlend) score -= 0.18;
    if (shortOrEffect) score += 0.08;
  }
  if (rec.method === "breakdown_switch" && (bpm.category === "medium" || bpm.category === "wide" || isWideWithoutHalfTime)) score += 0.2;
  if (rec.method === "breakdown_switch" && rec.outgoingCue.sectionType === "breakdown" && vocal.score < 0.25 && isWideWithoutHalfTime) {
    score += 0.5;
    reasons.push("A 歌 breakdown 人声少、能量下降，适合在空拍里切换");
  }
  if (rec.method === "echo_out" && rec.outgoingCue.sectionType === "breakdown" && isWideWithoutHalfTime) score += 0.12;
  if (rec.method === "fade" && context.beginnerMode && bpm.category === "same" && key.score >= 0.82 && vocal.score < 0.22) score -= 0.14;
  if (context.targetEnergy === "up" && ["drop", "chorus"].includes(rec.incomingCue.role)) {
    score += 0.16;
    reasons.push("B 歌从 chorus/drop 进入，适合提升现场能量");
  }
  if ((context.targetEnergy ?? "keep") === "keep" && ["drop", "chorus"].includes(rec.incomingCue.role) && incomingCueEnergy - outgoingCueEnergy > 0.25) {
    score -= 0.18;
    reasons.push("目标是保持能量，直接切入高能 drop/chorus 会太突然");
  }
  if (context.targetEnergy === "down" && ["fade", "end_to_end", "breakdown_switch"].includes(rec.method)) score += 0.1;

  const finalScore = round(clamp01(score));
  return {
    ...rec,
    score: finalScore,
    reason: enrichReason(rec.reason, reasons),
    debug: rec.debug
      ? {
          ...rec.debug,
          finalScore,
          beginnerPenalty: round(beginnerPenalty),
          reasons: [...new Set(reasons)],
        }
      : rec.debug,
  };
}

function enrichReason(reason: string, reasons: string[]): string {
  const important = reasons.filter(
    (item) =>
      item.includes("人声") ||
      item.includes("BPM 差") ||
      item.includes("调性") ||
      item.includes("能量"),
  );
  if (!important.length) return reason;
  return `${reason} ${[...new Set(important)].join("；")}。`;
}

function isTrackAnalysis(value: unknown): value is TrackAnalysis {
  return Boolean(value && typeof value === "object" && "bpm" in value && "sections" in value && "beatGrid" in value);
}
