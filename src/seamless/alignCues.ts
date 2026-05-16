import type { TrackAnalysis, TransitionRecommendation } from "../analysis/types";
import type { TransitionAlignment } from "../audio/types";

export function alignTransitionCues(
  outgoing: TrackAnalysis,
  incoming: TrackAnalysis,
  rec: TransitionRecommendation,
): TransitionAlignment {
  const outgoingSnap = snapCue(outgoing, rec.outgoingCue.time, "exit");
  const incomingSnap = snapCue(incoming, rec.incomingCue.time, "entry");
  const phraseAligned = outgoingSnap.isPhrase && incomingSnap.isPhrase;
  const overlapDuration = Math.max(0.5, rec.overlapDuration || phraseDuration(outgoing, 4));
  const driftPenalty = Math.min(0.3, (Math.abs(outgoingSnap.time - rec.outgoingCue.time) + Math.abs(incomingSnap.time - rec.incomingCue.time)) / 16);
  return {
    outgoingExitTime: outgoingSnap.time,
    incomingEntryTime: incomingSnap.time,
    outgoingDownbeatTime: outgoingSnap.downbeatTime,
    incomingDownbeatTime: incomingSnap.downbeatTime,
    overlapDuration,
    phraseAligned,
    alignmentConfidence: round(Math.max(0.25, (phraseAligned ? 0.9 : 0.62) - driftPenalty)),
  };
}

function snapCue(track: TrackAnalysis, time: number, role: "entry" | "exit") {
  const boundaries = track.beatGrid.filter((point) => point.beatIndex % 4 === 0);
  const phraseBoundaries = boundaries.filter((point) => point.barIndex % 8 === 0 || point.barIndex % 16 === 0);
  const candidates = phraseBoundaries.length ? phraseBoundaries : boundaries;
  const sectionBonus = (candidateTime: number) => {
    const section = track.sections.find((item) => candidateTime >= item.startTime && candidateTime <= item.endTime);
    if (!section) return 0;
    if (role === "exit" && ["outro", "breakdown", "chorus"].includes(section.type)) return -2.5;
    if (role === "entry" && ["intro", "drop", "chorus"].includes(section.type)) return -2.5;
    return 0;
  };
  const best = (candidates.length ? candidates : [{ time, beatIndex: 0, barIndex: 0, phraseIndex: 0 }]).reduce((winner, point) => {
    const currentScore = Math.abs(point.time - time) + sectionBonus(point.time);
    const winnerScore = Math.abs(winner.time - time) + sectionBonus(winner.time);
    return currentScore < winnerScore ? point : winner;
  });
  const downbeat = boundaries.reduce((winner, point) => (Math.abs(point.time - best.time) < Math.abs(winner.time - best.time) ? point : winner), best);
  return {
    time: round(best.time),
    downbeatTime: round(downbeat.time),
    isPhrase: best.barIndex % 8 === 0 || best.barIndex % 16 === 0,
  };
}

function phraseDuration(track: TrackAnalysis, bars: number): number {
  return bars * 4 * (60 / Math.max(1, track.bpm));
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
