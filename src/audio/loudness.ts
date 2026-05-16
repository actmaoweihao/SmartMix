import type { AudioBufferLike } from "./types";

export function integratedRmsDb(buffer: AudioBufferLike): number {
  let sum = 0;
  let count = 0;
  for (const channel of buffer.channels) {
    for (const sample of channel) {
      sum += sample * sample;
      count += 1;
    }
  }
  const rms = Math.sqrt(sum / Math.max(1, count));
  return 20 * Math.log10(Math.max(rms, 1e-9));
}

export function matchTransitionLoudnessBuffers(
  outgoing: AudioBufferLike,
  incoming: AudioBufferLike,
): { outgoingGainDb: number; incomingGainDb: number; loudnessDifferenceDb: number; warning?: string } {
  const outgoingDb = integratedRmsDb(outgoing);
  const incomingDb = integratedRmsDb(incoming);
  const diff = incomingDb - outgoingDb;
  const incomingGainDb = -diff;
  return {
    outgoingGainDb: 0,
    incomingGainDb: round(incomingGainDb),
    loudnessDifferenceDb: round(diff),
    warning: Math.abs(diff) > 8 ? "两首歌过渡段响度差较大，已建议降低较响的一方。" : undefined,
  };
}

export async function matchTransitionLoudness(
  outgoingSegmentPath: string,
  incomingSegmentPath: string,
): Promise<{ outgoingGainDb: number; incomingGainDb: number; loudnessDifferenceDb: number; warning?: string }> {
  void outgoingSegmentPath;
  void incomingSegmentPath;
  return { outgoingGainDb: 0, incomingGainDb: 0, loudnessDifferenceDb: 0, warning: "当前 TS fallback 未读取真实音频，后端集成时应使用 LUFS/RMS 实测。" };
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
