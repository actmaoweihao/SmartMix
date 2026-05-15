import { describe, expect, it } from "vitest";
import { scoreBpmCompatibility } from "../analysis/bpm";
import { makeTrack } from "./fixtures";

describe("scoreBpmCompatibility", () => {
  it("recommends beatmix for close BPM", () => {
    const score = scoreBpmCompatibility(makeTrack({ bpm: 124 }), makeTrack({ bpm: 126 }));
    expect(score.category).toBe("same");
    expect(score.suggestedMethod).toBe("beatmix");
    expect(score.score).toBeGreaterThan(0.9);
  });

  it("treats half-time and double-time as compatible", () => {
    const score = scoreBpmCompatibility(makeTrack({ bpm: 70 }), makeTrack({ bpm: 140 }));
    expect(score.bpmDiff).toBe(0);
    expect(score.category).toBe("same");
  });

  it("recommends wide BPM loop for very large BPM gaps", () => {
    const score = scoreBpmCompatibility(makeTrack({ bpm: 92 }), makeTrack({ bpm: 128 }));
    expect(score.category).toBe("wide");
    expect(score.suggestedMethod).toBe("wide_bpm_loop");
  });
});
