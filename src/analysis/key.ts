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

  if (aCode === bCode) {
    return buildScore({
      keyA,
      keyB,
      relation: "same",
      score: 1,
      explanation: `${aCode} -> ${bCode} 是完全相同的 Camelot 调性，适合稳定的 harmonic mixing 和较长叠加。`,
      warnings,
    });
  }

  if (aParsed.number === bParsed.number && aParsed.letter !== bParsed.letter) {
    return buildScore({
      keyA,
      keyB,
      relation: "relative_major_minor",
      score: 0.93,
      explanation: `${aCode} -> ${bCode} 是同数字 A/B 的关系大小调，旋律与和声通常高度兼容。`,
      warnings,
    });
  }

  const clockwise = clockwiseDelta(aParsed.number, bParsed.number);
  if (aParsed.letter === bParsed.letter && Math.abs(clockwise) === 1) {
    const directionalRelation = energyRelation(clockwise, context.targetEnergy);
    const score = directionalRelation === "adjacent" ? 0.88 : targetEnergyScore(directionalRelation, context.targetEnergy);
    return buildScore({
      keyA,
      keyB,
      relation: directionalRelation,
      score,
      explanation: adjacentExplanation(aCode, bCode, clockwise, directionalRelation, context.targetEnergy),
      warnings,
    });
  }

  return buildScore({
    keyA,
    keyB,
    relation: "clash",
    score: 0.24,
    explanation: `${aCode} -> ${bCode} 不属于同调、相邻调或关系大小调，调性不兼容，不建议长时间叠加；可考虑 quick cut、echo out、fade 或 end-to-end。`,
    warnings,
  });
}

export function scoreCamelotKeys(
  inputA: unknown,
  inputB: unknown,
  context: Pick<TransitionContext, "targetEnergy"> = {},
): KeyScore {
  return scoreKeyCompatibility(
    keyInputToTrack(inputA, "A"),
    keyInputToTrack(inputB, "B"),
    context,
  );
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
  return {
    score: debug.score,
    relation: input.relation,
    explanation: input.explanation,
    debug,
  };
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
  return forward <= 6 ? forward : forward - 12;
}

function energyRelation(
  clockwise: number,
  targetEnergy: TransitionContext["targetEnergy"],
): KeyScore["relation"] {
  if (targetEnergy === "up" && clockwise === 1) return "energy_boost";
  if (targetEnergy === "down" && clockwise === -1) return "energy_drop";
  return "adjacent";
}

function targetEnergyScore(relation: KeyScore["relation"], targetEnergy: TransitionContext["targetEnergy"]): number {
  if (relation === "energy_boost") return targetEnergy === "up" ? 0.9 : 0.86;
  if (relation === "energy_drop") return targetEnergy === "down" ? 0.9 : 0.86;
  return 0.88;
}

function adjacentExplanation(
  aCode: string,
  bCode: string,
  clockwise: number,
  relation: KeyScore["relation"],
  targetEnergy: TransitionContext["targetEnergy"],
): string {
  const direction = clockwise === 1 ? "顺时针相邻" : "逆时针相邻";
  if (relation === "energy_boost") return `${aCode} -> ${bCode} 是同字母的${direction} Camelot 调性，符合目标能量提升。`;
  if (relation === "energy_drop") return `${aCode} -> ${bCode} 是同字母的${direction} Camelot 调性，符合目标能量回落。`;
  const targetNote = targetEnergy ? "，但不把它强行解释为目标能量方向" : "";
  return `${aCode} -> ${bCode} 是同字母的${direction} Camelot 相邻调，适合 harmonic mixing${targetNote}。`;
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
