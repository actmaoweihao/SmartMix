import type { SongSection, SongSectionType, TrackAnalysis } from "./types";

export function valueAtCurve(curve: Array<{ time: number; energy?: number; density?: number }>, time: number, key: "energy" | "density"): number {
  if (!curve.length) return 0;
  const closest = curve.reduce((best, item) => (Math.abs(item.time - time) < Math.abs(best.time - time) ? item : best), curve[0]);
  return Number(closest[key] ?? 0);
}

export function energyAt(track: TrackAnalysis, time: number): number {
  return clamp01(valueAtCurve(track.energyCurve, time, "energy"));
}

export function vocalDensityAt(track: TrackAnalysis, time: number): number {
  if (track.hasVocalAtTime?.(time)) return 1;
  return clamp01(valueAtCurve(track.vocalDensityCurve ?? [], time, "density"));
}

export function averageRange(track: TrackAnalysis, start: number, duration: number, kind: "energy" | "vocal"): number {
  const samples = Math.max(2, Math.ceil(duration / 2));
  let total = 0;
  for (let index = 0; index < samples; index += 1) {
    const t = start + (duration * index) / Math.max(1, samples - 1);
    total += kind === "energy" ? energyAt(track, t) : vocalDensityAt(track, t);
  }
  return clamp01(total / samples);
}

export function sectionAtTime(track: TrackAnalysis, time: number): SongSection | null {
  return track.sections.find((section) => time >= section.startTime && time < section.endTime) ?? null;
}

export function findSection(track: TrackAnalysis, types: SongSectionType[]): SongSection | null {
  return track.sections.find((section) => types.includes(section.type)) ?? null;
}

export function trackAverageEnergy(track: TrackAnalysis): number {
  if (!track.energyCurve.length) return 0.5;
  return clamp01(track.energyCurve.reduce((sum, point) => sum + point.energy, 0) / track.energyCurve.length);
}

export function energyFlowScore(outgoing: TrackAnalysis, incoming: TrackAnalysis, target: "up" | "down" | "keep" = "keep"): number {
  const diff = trackAverageEnergy(incoming) - trackAverageEnergy(outgoing);
  if (target === "up") return clamp01(0.55 + diff);
  if (target === "down") return clamp01(0.55 - diff);
  return clamp01(1 - Math.abs(diff));
}

export function clamp01(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}
