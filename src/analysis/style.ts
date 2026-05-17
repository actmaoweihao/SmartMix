import type { StyleScore, TrackAnalysis, TrackStyle } from "./types";

const STYLE_FAMILIES: Record<string, TrackStyle[]> = {
  dance: ["house", "techno", "drum_bass", "electronic"],
  urban: ["hiphop", "rnb"],
  band: ["rock", "pop"],
  chill: ["ambient", "rnb", "electronic"],
};

const STYLE_BRIDGES = new Set([
  stylePair("house", "techno"),
  stylePair("house", "electronic"),
  stylePair("techno", "electronic"),
  stylePair("drum_bass", "electronic"),
  stylePair("hiphop", "rnb"),
  stylePair("pop", "rnb"),
  stylePair("pop", "rock"),
  stylePair("pop", "house"),
  stylePair("pop", "electronic"),
  stylePair("ambient", "electronic"),
]);

export function scoreStyleCompatibility(outgoing: TrackAnalysis, incoming: TrackAnalysis): StyleScore {
  const from = normalizeStyle(outgoing.style ?? outgoing.styleProfile?.primary);
  const to = normalizeStyle(incoming.style ?? incoming.styleProfile?.primary);
  const confidence = Math.min(styleConfidence(outgoing), styleConfidence(incoming));

  if (!from || !to || from === "unknown" || to === "unknown" || confidence < 0.28) {
    return {
      score: 0.62,
      relation: "unknown",
      from,
      to,
      confidence,
      explanation: "风格识别不够确定，按中性风格兼容处理",
    };
  }
  if (from === to) {
    return {
      score: 1,
      relation: "same",
      from,
      to,
      confidence,
      explanation: `同为 ${styleLabel(from)}，适合更长的乐句叠加`,
    };
  }
  if (STYLE_BRIDGES.has(stylePair(from, to))) {
    return {
      score: 0.84,
      relation: "bridge",
      from,
      to,
      confidence,
      explanation: `${styleLabel(from)} -> ${styleLabel(to)} 是可桥接风格，重点看 BPM、调性和段落`,
    };
  }
  const fromFamily = styleFamily(from);
  const toFamily = styleFamily(to);
  if (fromFamily && fromFamily === toFamily) {
    return {
      score: 0.76,
      relation: "same_family",
      from,
      to,
      confidence,
      explanation: `${styleLabel(from)} 和 ${styleLabel(to)} 属于同族风格，可以用 EQ 或短乐句衔接`,
    };
  }
  return {
    score: 0.46,
    relation: "contrast",
    from,
    to,
    confidence,
    explanation: `${styleLabel(from)} -> ${styleLabel(to)} 风格跨度大，优先短切、效果器或 breakdown 切换`,
  };
}

export function styleDistance(outgoing: Pick<TrackAnalysis, "style" | "styleConfidence" | "styleProfile">, incoming: Pick<TrackAnalysis, "style" | "styleConfidence" | "styleProfile">): number {
  return 1 - scoreStyleCompatibility(outgoing as TrackAnalysis, incoming as TrackAnalysis).score;
}

export function normalizeStyle(value: unknown): TrackStyle | undefined {
  if (!value) return undefined;
  const normalized = String(value).trim().toLowerCase().replace(/[-\s]+/g, "_");
  const aliases: Record<string, TrackStyle> = {
    hip_hop: "hiphop",
    hiphop_rap: "hiphop",
    rap: "hiphop",
    dnb: "drum_bass",
    drum_and_bass: "drum_bass",
    edm: "electronic",
    electronica: "electronic",
    "r&b": "rnb",
    r_b: "rnb",
  };
  const style = aliases[normalized] ?? normalized;
  return isTrackStyle(style) ? style : "unknown";
}

export function styleLabel(style: TrackStyle | undefined): string {
  return {
    house: "House",
    techno: "Techno",
    drum_bass: "Drum & Bass",
    hiphop: "Hip-Hop",
    rnb: "R&B",
    rock: "Rock",
    pop: "Pop",
    ambient: "Ambient",
    electronic: "Electronic",
    unknown: "Unknown",
  }[style ?? "unknown"];
}

export function styleFamily(style: TrackStyle | undefined): string | undefined {
  if (!style) return undefined;
  return Object.entries(STYLE_FAMILIES).find(([, styles]) => styles.includes(style))?.[0];
}

function styleConfidence(track: Pick<TrackAnalysis, "style" | "styleConfidence" | "styleProfile">): number {
  const value = track.styleConfidence ?? track.styleProfile?.confidence;
  if (Number.isFinite(value)) return clamp01(Number(value));
  return normalizeStyle(track.style ?? track.styleProfile?.primary) ? 0.5 : 0;
}

function stylePair(a: TrackStyle, b: TrackStyle): string {
  return [a, b].sort().join(":");
}

function isTrackStyle(value: string): value is TrackStyle {
  return ["house", "techno", "drum_bass", "hiphop", "rnb", "rock", "pop", "ambient", "electronic", "unknown"].includes(value);
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}
