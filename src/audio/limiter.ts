import type { AudioBufferLike } from "./types";

export function peakLimit(buffer: AudioBufferLike, ceiling = 0.98): AudioBufferLike {
  const peak = Math.max(...buffer.channels.map((channel) => channel.reduce((max, sample) => Math.max(max, Math.abs(sample)), 0)), 1e-9);
  const gain = peak > ceiling ? ceiling / peak : 1;
  return {
    sampleRate: buffer.sampleRate,
    channels: buffer.channels.map((channel) => channel.map((sample) => sample * gain) as Float32Array),
  };
}
