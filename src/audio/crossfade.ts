import type { AudioBufferLike, CrossfadeCurve, GainAutomation } from "./types";

export function renderCrossfade(
  outgoingSegment: AudioBufferLike,
  incomingSegment: AudioBufferLike,
  options: {
    curve: CrossfadeCurve;
    overlapDuration: number;
    outgoingAutomation?: GainAutomation[];
    incomingAutomation?: GainAutomation[];
  },
): AudioBufferLike {
  const sampleRate = outgoingSegment.sampleRate;
  const samples = Math.min(outgoingSegment.channels[0]?.length ?? 0, incomingSegment.channels[0]?.length ?? 0, Math.max(1, Math.round(options.overlapDuration * sampleRate)));
  const channels = Math.max(outgoingSegment.channels.length, incomingSegment.channels.length);
  const output = Array.from({ length: channels }, (_, channel) => {
    const out = outgoingSegment.channels[channel] ?? outgoingSegment.channels[0] ?? new Float32Array(samples);
    const inc = incomingSegment.channels[channel] ?? incomingSegment.channels[0] ?? new Float32Array(samples);
    const mixed = new Float32Array(samples);
    for (let i = 0; i < samples; i += 1) {
      const x = samples <= 1 ? 1 : i / (samples - 1);
      const [outGain, inGain] = curveGains(options.curve, x);
      const clickSafe = clickGuard(x);
      mixed[i] = (out[i] * outGain + inc[i] * inGain) * clickSafe;
    }
    return mixed;
  });
  return { sampleRate, channels: output };
}

export function curveGains(curve: CrossfadeCurve, x: number): [number, number] {
  const t = clamp01(x);
  if (curve === "equal_power") return [Math.cos(t * Math.PI / 2), Math.sin(t * Math.PI / 2)];
  if (curve === "smoothstep") {
    const s = t * t * (3 - 2 * t);
    return [1 - s, s];
  }
  if (curve === "logarithmic") {
    return [Math.pow(1 - t, 1.8), Math.pow(t, 0.65)];
  }
  return [1 - t, t];
}

function clickGuard(x: number): number {
  const edge = 0.006;
  if (x < edge) return x / edge;
  if (x > 1 - edge) return (1 - x) / edge;
  return 1;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}
