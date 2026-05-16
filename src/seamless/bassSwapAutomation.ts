import type { CuePoint } from "../analysis/types";
import type { GainAutomation } from "../audio/types";

export function createBassSwapAutomation(
  outgoingCue: CuePoint,
  incomingCue: CuePoint,
  overlapDuration: number,
): { outgoingLowGain: GainAutomation[]; incomingLowGain: GainAutomation[] } {
  void outgoingCue;
  void incomingCue;
  const swapStart = Math.max(0, overlapDuration * 0.62);
  const swapEnd = Math.min(overlapDuration, swapStart + Math.max(0.25, overlapDuration * 0.12));
  return {
    outgoingLowGain: [
      { stem: "bass", startTime: 0, endTime: swapStart, startGain: 1, endGain: 1, curve: "linear" },
      { stem: "bass", startTime: swapStart, endTime: swapEnd, startGain: 1, endGain: 0.12, curve: "smoothstep" },
    ],
    incomingLowGain: [
      { stem: "bass", startTime: 0, endTime: swapStart, startGain: 0.08, endGain: 0.08, curve: "linear" },
      { stem: "bass", startTime: swapStart, endTime: swapEnd, startGain: 0.08, endGain: 1, curve: "smoothstep" },
    ],
  };
}
