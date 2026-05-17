import type { KeyCompatibilityDebug, KeyScore, TrackAnalysis, TransitionContext } from "./types";

const CAMELOT_MAP: Record<string, string> = {
  "A MINOR": "8A",
  "C MAJOR": "8B",
  "E MINOR": "9A",
  "G MAJOR": "9B",
  "B MINOR": "10A",
  "D MAJOR": "10B",
  "F# MINOR": "11A",
  "GB MINOR": "11A",
  "A MAJOR": "11B",
  "C# MINOR": "12A",
  "DB MINOR": "12A",
  "E MAJOR": "12B",
  "G# MINOR": "1A",
  "AB MINOR": "1A",
  "B MAJOR": "1B",
  "D# MINOR": "2A",
  "EB MINOR": "2A",
  "F# MAJOR": "2B",
  "GB MAJOR": "2B",
  "A# MINOR": "3A",
  "BB MINOR": "3A",
  "C# MAJOR": "3B",
  "DB MAJOR": "3B",
  "F MINOR": "4A",
  "G# MAJOR": "4B",
  "AB MAJOR": "4B",
  "C MINOR": "5A",
  "D# MAJOR": "5B",
  "EB MAJOR": "5B",
  "G MINOR": "6A",
  "A# MAJOR": "6B",
  "BB MAJOR": "6B",
  "D MINOR": "7A",
  "F MAJOR": "7B",
};

type ParsedCamelotKey = { number: number; letter: "A" | "B" };
type ResolvedKey = {
  input: string;
  normalized?: string;
  parsed?: ParsedCamelotKey;
  warnings: string[];
};

export function scoreKeyCompatibility(
  a: TrackAnalysis,
  b: TrackAnalysis,
  context: Pick<TransitionContext, "targetEnergy"> = {},
): KeyScore {
  const keyA = resolveTrackKey(a, "A");
  const keyB = resolveTrackKey(b, "B");
  const warnings = [...keyA.warnings, ...keyB.warnings];

  if (!keyA.normalized || !keyB.normalized || !keyA.parsed || !keyB.parsed) {
    return buildScore({
      keyA,
      keyB,
      relation: "unknown",
      score: 0.4,
      explanation: "未能识别至少一首歌的调性，因此更建议使用短切、淡出或效果器过渡。",
      warnings,
    });
  }

  const aParsed = keyA.parsed;
  const bParsed = keyB.parsed;
  const aCode = keyA.normalized;
  const bCode = keyB.normalized;
  const clockwise = clockwiseDelta(aParsed.number, bParsed.number);

  if (aCode === bCode) {
    return buildScore({
      keyA,
      keyB,
      relation: "same",
      score: 1,
      explanation: `${aCode} -> ${bCode} 是 Perfect Mix：完全相同的 Camelot 调性，适合稳定 harmonic mixing 和较长叠加。`,
      warnings,
    });
  }

  if (aParsed.number === bParsed.number && aParsed.letter !== bParsed.letter) {
    return buildScore({
      keyA,
      keyB,
      relation: "relative_major_minor",
      score: 0.93,
      explanation: `${aCode} -> ${bCode} 是 Scale Change：同数字 A/B 互换，属于关系大小调，通常高度兼容。`,
      warnings,
    });
  }

  if (aParsed.letter === bParsed.letter && Math.abs(clockwise) === 1) {
    const directionalRelation = energyRelation(clockwise, context.targetEnergy);
    return buildScore({
      keyA,
      keyB,
      relation: directionalRelation,
      score: directionalRelation === "adjacent" ? 0.88 : 0.9,
      explanation: `${aCode} -> ${bCode} 是 ${clockwise === 1 ? "+1 Mix" : "-1 Mix"}：同字母相邻 Camelot 调性，适合顺滑混音。`,
      warnings,
    });
  }

  if (aParsed.letter === bParsed.letter && clockwise === 2) {
    return buildScore({
      keyA,
      keyB,
      relation: "energy_boost",
      score: context.targetEnergy === "up" ? 0.84 : 0.8,
      explanation: `${aCode} -> ${bCode} 是 Energy Boost：同字母顺时针 +2，用于明显提能量，建议控制叠加长度。`,
      warnings,
    });
  }

  if (isDiagonalMix(aParsed, bParsed, clockwise)) {
    return buildScore({
      keyA,
      keyB,
      relation: "diagonal_mix",
      score: 0.74,
      explanation: `${aCode} -> ${bCode} 是 Diagonal Mix：跨 A/B 且逆时针 -1，属于特殊效果型兼容，优先短叠加或效果器过渡。`,
      warnings,
    });
  }

  if (aParsed.letter === bParsed.letter && clockwise === 7) {
    return buildScore({
      keyA,
      keyB,
      relation: "jaws_mix",
      score: 0.62,
      explanation: `${aCode} -> ${bCode} 是 Jaw's Mix：同字母 +7 的戏剧化跳转，适合短切或特殊段落，不建议长时间叠加。`,
      warnings,
    });
  }

  if (isMoodShifter(aParsed, bParsed, clockwise)) {
    return buildScore({
      keyA,
      keyB,
      relation: "mood_shifter",
      score: 0.66,
      explanation: `${aCode} -> ${bCode} 是 Mood Shifter：跨 A/B 且顺时针 +4，适合改变情绪色彩，建议用短混或段落切换。`,
      warnings,
    });
  }

  return buildScore({
    keyA,
    keyB,
    relation: "clash",
    score: 0.24,
    explanation: `${aCode} -> ${bCode} 不属于 Perfect、±1、Energy Boost、Scale Change、Diagonal、Jaw's 或 Mood Shifter，调性不兼容，不建议长时间叠加。`,
    warnings,
  });
}

export function scoreCamelotKeys(
  inputA: unknown,
  inputB: unknown,
  context: Pick<TransitionContext, "targetEnergy"> = {},
): KeyScore {
  return scoreKeyCompatibility(keyInputToTrack(inputA, "A"), keyInputToTrack(inputB, "B"), context);
}

export function toCamelot(track: TrackAnalysis): string | null {
  return resolveTrackKey(track, "track").normalized ?? null;
}

export function normalizeKeyToCamelot(value: unknown): string | null {
  const normalized = normalizeInput(value);
  if (!normalized) return null;
  const direct = parseCamelotKey(normalized);
  if (direct) return `${direct.number}${direct.letter}`;
  const traditional = normalizeTraditionalKeyName(normalized);
  return traditional ? CAMELOT_MAP[traditional] ?? null : null;
}

export function parseCamelotKey(value: unknown): ParsedCamelotKey | null {
  if (typeof value !== "string") return null;
  const match = value.trim().toUpperCase().match(/^(\d{1,2})([AB])$/);
  if (!match) return null;
  const number = Number(match[1]);
  if (!Number.isInteger(number) || number < 1 || number > 12) return null;
  return { number, letter: match[2] as "A" | "B" };
}

function resolveTrackKey(track: TrackAnalysis, label: string): ResolvedKey {
  const warnings: string[] = [];
  const camelotInput = normalizeInput(track.camelotKey);
  if (camelotInput) {
    const parsed = parseCamelotKey(camelotInput);
    if (!parsed) {
      warnings.push(`${label}: invalid Camelot key "${camelotInput}"`);
      return { input: camelotInput, warnings };
    }
    const normalized = `${parsed.number}${parsed.letter}`;
    return { input: camelotInput, normalized, parsed, warnings };
  }
  const keyInput = normalizeInput(track.key);
  if (!keyInput) {
    warnings.push(`${label}: missing musical key`);
    return { input: "", warnings };
  }
  const normalized = normalizeKeyToCamelot(keyInput) ?? undefined;
  const parsed = normalized ? parseCamelotKey(normalized) ?? undefined : undefined;
  if (!normalized || !parsed) warnings.push(`${label}: unsupported musical key "${keyInput}"`);
  return { input: keyInput, normalized, parsed, warnings };
}

function buildScore(input: {
  keyA: ResolvedKey;
  keyB: ResolvedKey;
  relation: KeyScore["relation"];
  score: number;
  explanation: string;
  warnings: string[];
}): KeyScore {
  const debug: KeyCompatibilityDebug = {
    inputA: input.keyA.input,
    inputB: input.keyB.input,
    normalizedA: input.keyA.normalized,
    normalizedB: input.keyB.normalized,
    parsedA: input.keyA.parsed,
    parsedB: input.keyB.parsed,
    relation: input.relation,
    score: round(input.score),
    explanation: input.explanation,
    warnings: input.warnings,
  };
  return { score: debug.score, relation: input.relation, explanation: input.explanation, debug };
}

function normalizeInput(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeTraditionalKeyName(value: string): string | null {
  const cleaned = value
    .replace(/♯/g, "#")
    .replace(/♭/g, "B")
    .replace(/-/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toUpperCase();
  if (!cleaned) return null;

  const compact = cleaned.replace(/\s+/g, "");
  const short = compact.match(/^([A-G](?:#|B)?)(M|MIN|MINOR|MAJ|MAJOR)?$/);
  if (short) {
    const root = short[1];
    const mode = short[2];
    if (mode === "M" || mode === "MIN" || mode === "MINOR") return `${root} MINOR`;
    if (mode === "MAJ" || mode === "MAJOR") return `${root} MAJOR`;
  }

  const parts = cleaned.split(" ");
  if (parts.length >= 2) {
    const root = parts[0];
    const mode = parts[1];
    if (mode === "MIN" || mode === "MINOR" || mode === "M") return `${root} MINOR`;
    if (mode === "MAJ" || mode === "MAJOR") return `${root} MAJOR`;
  }
  return null;
}

function clockwiseDelta(from: number, to: number): number {
  const forward = (to - from + 12) % 12;
  if (forward === 0) return 0;
  return forward <= 7 ? forward : forward - 12;
}

function energyRelation(clockwise: number, targetEnergy: TransitionContext["targetEnergy"]): KeyScore["relation"] {
  if (targetEnergy === "up" && clockwise === 1) return "energy_boost";
  if (targetEnergy === "down" && clockwise === -1) return "energy_drop";
  return "adjacent";
}

function isDiagonalMix(a: ParsedCamelotKey, b: ParsedCamelotKey, clockwise: number): boolean {
  if (a.letter === b.letter) return false;
  return (a.letter === "A" && b.letter === "B" && clockwise === -1) || (a.letter === "B" && b.letter === "A" && clockwise === 1);
}

function isMoodShifter(a: ParsedCamelotKey, b: ParsedCamelotKey, clockwise: number): boolean {
  if (a.letter === b.letter) return false;
  return (a.letter === "A" && b.letter === "B" && clockwise === 3) || (a.letter === "B" && b.letter === "A" && clockwise === -3);
}

function keyInputToTrack(input: unknown, label: string): TrackAnalysis {
  const key = typeof input === "string" ? input : "";
  return {
    id: `key-${label}`,
    title: `Key ${label}`,
    duration: 0,
    bpm: 120,
    key,
    camelotKey: parseCamelotKey(key) ? key : undefined,
    energyCurve: [],
    sections: [],
    beatGrid: [],
  };
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
