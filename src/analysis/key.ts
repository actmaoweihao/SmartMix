import type { KeyScore, TrackAnalysis } from "./types";

const CAMELOT_BY_NAME: Record<string, string> = {
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

export function scoreKeyCompatibility(a: TrackAnalysis, b: TrackAnalysis): KeyScore {
  const keyA = toCamelot(a);
  const keyB = toCamelot(b);
  if (!keyA || !keyB) return { score: 0.45, relation: "unknown", explanation: "调性信息不足，建议用短切或 FX 降低风险。" };
  const parsedA = parseCamelot(keyA);
  const parsedB = parseCamelot(keyB);
  if (!parsedA || !parsedB) return { score: 0.45, relation: "unknown", explanation: "无法解析 Camelot 调性，建议保守过渡。" };
  if (keyA === keyB) return { score: 1, relation: "same", explanation: `${keyA} 到 ${keyB} 是同调，适合长一点混合。` };
  if (parsedA.num === parsedB.num && parsedA.letter !== parsedB.letter) {
    return { score: 0.92, relation: "relative_major_minor", explanation: `${keyA} 到 ${keyB} 是关系大小调，旋律通常比较自然。` };
  }
  const clockwise = wrapDistance(parsedB.num - parsedA.num);
  if (parsedA.letter === parsedB.letter && Math.abs(clockwise) === 1) {
    return {
      score: clockwise === 1 ? 0.9 : 0.84,
      relation: clockwise === 1 ? "energy_boost" : "adjacent",
      explanation: `${keyA} 到 ${keyB} 是相邻 Camelot，${clockwise === 1 ? "适合轻微提能量" : "适合平稳衔接"}。`,
    };
  }
  if (parsedA.letter === parsedB.letter && clockwise === -1) {
    return { score: 0.82, relation: "energy_drop", explanation: `${keyA} 到 ${keyB} 可做能量回落型衔接。` };
  }
  return { score: 0.28, relation: "clash", explanation: `${keyA} 到 ${keyB} 调性距离较远，优先 quick cut、echo out 或短重叠。` };
}

export function toCamelot(track: TrackAnalysis): string | null {
  if (track.camelotKey && parseCamelot(track.camelotKey)) return track.camelotKey.toUpperCase();
  return CAMELOT_BY_NAME[track.key.trim().toUpperCase()] ?? null;
}

function parseCamelot(key: string): { num: number; letter: "A" | "B" } | null {
  const match = key.toUpperCase().match(/^(\d{1,2})([AB])$/);
  if (!match) return null;
  const num = Number(match[1]);
  if (num < 1 || num > 12) return null;
  return { num, letter: match[2] as "A" | "B" };
}

function wrapDistance(diff: number): number {
  if (diff > 6) return diff - 12;
  if (diff < -6) return diff + 12;
  return diff;
}
