import type { BeatGridPoint, SongSection, TrackAnalysis } from "../analysis/types";

export function makeTrack(overrides: Partial<TrackAnalysis> = {}): TrackAnalysis {
  const duration = overrides.duration ?? 180;
  const bpm = overrides.bpm ?? 124;
  const beatGrid = overrides.beatGrid ?? makeBeatGrid(duration, bpm);
  const sections = overrides.sections ?? defaultSections(duration, bpm);
  return {
    id: overrides.id ?? `track-${Math.random()}`,
    title: overrides.title ?? "Mock Track",
    artist: overrides.artist,
    duration,
    bpm,
    key: overrides.key ?? "A minor",
    camelotKey: overrides.camelotKey ?? "8A",
    energyCurve: overrides.energyCurve ?? [
      { time: 0, energy: 0.35 },
      { time: 32, energy: 0.48 },
      { time: 64, energy: 0.78 },
      { time: 96, energy: 0.82 },
      { time: 132, energy: 0.52 },
      { time: 160, energy: 0.38 },
    ],
    sections,
    beatGrid,
    vocalDensityCurve: overrides.vocalDensityCurve ?? [
      { time: 0, density: 0.05 },
      { time: 32, density: 0.65 },
      { time: 64, density: 0.75 },
      { time: 112, density: 0.28 },
      { time: 152, density: 0.08 },
    ],
    hasVocalAtTime: overrides.hasVocalAtTime,
  };
}

export function makeBeatGrid(duration: number, bpm: number): BeatGridPoint[] {
  const beatSeconds = 60 / bpm;
  const beats = Math.floor(duration / beatSeconds);
  return Array.from({ length: beats }, (_, beatIndex) => ({
    time: beatIndex * beatSeconds,
    beatIndex,
    barIndex: Math.floor(beatIndex / 4),
    phraseIndex: Math.floor(beatIndex / 32),
  }));
}

function defaultSections(duration: number, bpm: number): SongSection[] {
  const beat = 60 / bpm;
  return [
    section("intro", 0, 32 * beat, bpm, 0.9),
    section("verse", 32 * beat, 96 * beat, bpm, 0.72),
    section("chorus", 96 * beat, 160 * beat, bpm, 0.86),
    section("breakdown", 160 * beat, 192 * beat, bpm, 0.74),
    section("drop", 192 * beat, 256 * beat, bpm, 0.82),
    section("outro", Math.max(0, duration - 32 * beat), duration, bpm, 0.84),
  ];
}

function section(type: SongSection["type"], startTime: number, endTime: number, bpm: number, confidence: number): SongSection {
  const beat = 60 / bpm;
  return {
    type,
    startTime,
    endTime,
    startBeat: Math.round(startTime / beat),
    endBeat: Math.round(endTime / beat),
    confidence,
  };
}
