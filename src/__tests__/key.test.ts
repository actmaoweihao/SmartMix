import { describe, expect, it } from "vitest";
import { scoreKeyCompatibility } from "../analysis/key";
import { makeTrack } from "./fixtures";

describe("scoreKeyCompatibility", () => {
  it("scores same Camelot key highly", () => {
    const score = scoreKeyCompatibility(makeTrack({ camelotKey: "8A" }), makeTrack({ camelotKey: "8A" }));
    expect(score.relation).toBe("same");
    expect(score.score).toBe(1);
  });

  it("scores adjacent and relative keys as compatible", () => {
    expect(scoreKeyCompatibility(makeTrack({ camelotKey: "8A" }), makeTrack({ camelotKey: "9A" })).score).toBeGreaterThan(0.8);
    expect(scoreKeyCompatibility(makeTrack({ camelotKey: "8A" }), makeTrack({ camelotKey: "8B" })).relation).toBe("relative_major_minor");
  });

  it("penalizes incompatible keys", () => {
    const score = scoreKeyCompatibility(makeTrack({ camelotKey: "8A" }), makeTrack({ camelotKey: "2B" }));
    expect(score.relation).toBe("clash");
    expect(score.score).toBeLessThan(0.4);
  });
});
