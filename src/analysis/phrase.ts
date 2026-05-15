import { averageRange, energyAt, sectionAtTime } from "./energy";
import type { CuePoint, SongSection, SongSectionType, TrackAnalysis } from "./types";

const ENTRY_SECTIONS: SongSectionType[] = ["intro", "verse", "drop", "chorus", "breakdown"];
const EXIT_SECTIONS: SongSectionType[] = ["outro", "chorus", "breakdown", "bridge"];

export function findPhraseAlignedCuePoints(track: TrackAnalysis): CuePoint[] {
  const cues: CuePoint[] = [];
  for (const section of track.sections) {
    if (ENTRY_SECTIONS.includes(section.type)) {
      cues.push(sectionCue(track, section, entryRole(section.type), "entry"));
    }
    if (EXIT_SECTIONS.includes(section.type)) {
      cues.push(sectionCue(track, section, exitRole(section.type), "exit", section.type === "chorus"));
    }
  }
  if (!cues.length && track.beatGrid.length) {
    const firstBar = track.beatGrid.find((beat) => beat.beatIndex % 4 === 0) ?? track.beatGrid[0];
    cues.push({
      time: firstBar.time,
      beatIndex: firstBar.beatIndex,
      barIndex: firstBar.barIndex,
      phraseIndex: firstBar.phraseIndex,
      sectionType: sectionAtTime(track, firstBar.time)?.type ?? "intro",
      role: "entry",
      confidence: 0.35,
    });
  }
  return dedupeCues(cues).sort((a, b) => b.confidence - a.confidence || a.time - b.time);
}

export function phraseAlignmentScore(outgoingCue: CuePoint, incomingCue: CuePoint): number {
  const samePhrasePosition = outgoingCue.barIndex % 8 === incomingCue.barIndex % 8 || outgoingCue.barIndex % 16 === incomingCue.barIndex % 16;
  const phraseStart = incomingCue.barIndex % 8 === 0 || incomingCue.barIndex % 16 === 0;
  return Math.min(1, 0.45 + (samePhrasePosition ? 0.3 : 0) + (phraseStart ? 0.2 : 0) + (outgoingCue.confidence + incomingCue.confidence) * 0.05);
}

export function bestCue(track: TrackAnalysis, roles: CuePoint["role"][]): CuePoint {
  const cues = findPhraseAlignedCuePoints(track);
  return cues.find((cue) => roles.includes(cue.role)) ?? cues[0] ?? fallbackCue(track, roles[0] ?? "entry");
}

function sectionCue(track: TrackAnalysis, section: SongSection, role: CuePoint["role"], base: "entry" | "exit", useEnd = false): CuePoint {
  const rawTime = useEnd ? section.endTime : section.startTime;
  const grid = nearestPhraseGrid(track, rawTime);
  const vocal = averageRange(track, grid.time, Math.min(16, Math.max(2, section.endTime - section.startTime)), "vocal");
  const energy = energyAt(track, grid.time);
  const phraseBonus = grid.barIndex % 16 === 0 ? 0.18 : grid.barIndex % 8 === 0 ? 0.12 : -0.1;
  const vocalPenalty = base === "entry" && section.type === "chorus" ? vocal * 0.08 : vocal * 0.25;
  const energyBonus = role === "drop" || role === "chorus" ? energy * 0.1 : (1 - energy) * 0.08;
  return {
    time: grid.time,
    beatIndex: grid.beatIndex,
    barIndex: grid.barIndex,
    phraseIndex: grid.phraseIndex,
    sectionType: section.type,
    role,
    confidence: Math.max(0.2, Math.min(0.98, section.confidence + phraseBonus + energyBonus - vocalPenalty)),
  };
}

function nearestPhraseGrid(track: TrackAnalysis, time: number) {
  const preferred = track.beatGrid.filter((beat) => beat.beatIndex % 4 === 0 && (beat.barIndex % 8 === 0 || beat.barIndex % 16 === 0));
  const candidates = preferred.length ? preferred : track.beatGrid.filter((beat) => beat.beatIndex % 4 === 0);
  return (candidates.length ? candidates : track.beatGrid).reduce(
    (best, beat) => (Math.abs(beat.time - time) < Math.abs(best.time - time) ? beat : best),
    track.beatGrid[0] ?? { time, beatIndex: 0, barIndex: 0, phraseIndex: 0 },
  );
}

function entryRole(type: SongSectionType): CuePoint["role"] {
  if (type === "drop") return "drop";
  if (type === "chorus") return "chorus";
  if (type === "breakdown") return "breakdown";
  return "entry";
}

function exitRole(type: SongSectionType): CuePoint["role"] {
  if (type === "outro") return "outro";
  if (type === "breakdown") return "breakdown";
  if (type === "chorus") return "chorus";
  return "exit";
}

function fallbackCue(track: TrackAnalysis, role: CuePoint["role"]): CuePoint {
  const beat = track.beatGrid[0] ?? { time: 0, beatIndex: 0, barIndex: 0, phraseIndex: 0 };
  return { ...beat, sectionType: "intro", role, confidence: 0.25 };
}

function dedupeCues(cues: CuePoint[]): CuePoint[] {
  const map = new Map<string, CuePoint>();
  for (const cue of cues) {
    const key = `${cue.role}-${Math.round(cue.time * 10)}`;
    const current = map.get(key);
    if (!current || cue.confidence > current.confidence) map.set(key, cue);
  }
  return [...map.values()];
}
