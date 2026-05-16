import { describe, expect, it } from "vitest";
import { normalizeKeyToCamelot, parseCamelotKey, scoreKeyCompatibility, toCamelot } from "../analysis/key";
import { makeTrack } from "./fixtures";

describe("Camelot key compatibility", () => {
  it("case 1: scores the same Camelot key as the highest compatibility", () => {
    const score = scoreKeyCompatibility(makeTrack({ camelotKey: "8A" }), makeTrack({ camelotKey: "8A" }));

    expect(score.relation).toBe("same");
    expect(score.score).toBeCloseTo(1);
    expect(score.debug?.parsedA).toEqual({ number: 8, letter: "A" });
    expect(score.debug?.parsedB).toEqual({ number: 8, letter: "A" });
  });

  it("case 2: scores same-letter adjacent keys as highly compatible", () => {
    for (const [from, to] of [
      ["8A", "7A"],
      ["8A", "9A"],
      ["8B", "7B"],
      ["8B", "9B"],
    ]) {
      const score = scoreKeyCompatibility(makeTrack({ camelotKey: from }), makeTrack({ camelotKey: to }));
      expect(score.relation).toBe("adjacent");
      expect(score.score).toBeGreaterThanOrEqual(0.85);
    }
  });

  it("case 3: supports 1 and 12 wrap-around adjacency", () => {
    for (const [from, to] of [
      ["1A", "12A"],
      ["12A", "1A"],
      ["1B", "12B"],
      ["12B", "1B"],
    ]) {
      const score = scoreKeyCompatibility(makeTrack({ camelotKey: from }), makeTrack({ camelotKey: to }));
      expect(score.relation).toBe("adjacent");
      expect(score.score).toBeGreaterThanOrEqual(0.85);
    }
  });

  it("case 4: scores same-number A/B as relative major/minor", () => {
    const minorToMajor = scoreKeyCompatibility(makeTrack({ camelotKey: "8A" }), makeTrack({ camelotKey: "8B" }));
    const majorToMinor = scoreKeyCompatibility(makeTrack({ camelotKey: "8B" }), makeTrack({ camelotKey: "8A" }));

    expect(minorToMajor.relation).toBe("relative_major_minor");
    expect(majorToMinor.relation).toBe("relative_major_minor");
    expect(minorToMajor.score).toBeGreaterThanOrEqual(0.9);
    expect(majorToMinor.score).toBeGreaterThanOrEqual(0.9);
  });

  it("case 5: scores non-Camelot-wheel relationships as clash", () => {
    for (const [from, to] of [
      ["8A", "2B"],
      ["3A", "10B"],
    ]) {
      const score = scoreKeyCompatibility(makeTrack({ camelotKey: from }), makeTrack({ camelotKey: to }));
      expect(score.relation).toBe("clash");
      expect(score.score).toBeLessThan(0.35);
      expect(score.explanation).toContain("不建议长时间叠加");
    }
  });

  it("case 6: treats invalid Camelot input as unknown without throwing", () => {
    const invalidInputs = ["0A", "13A", "8C", "A8", ""];
    for (const input of invalidInputs) {
      const track = makeTrack({ camelotKey: input, key: "" });
      const score = scoreKeyCompatibility(track, makeTrack({ camelotKey: "8A" }));
      expect(score.relation).toBe("unknown");
      expect(score.score).toBeLessThan(0.85);
    }

    const undefinedKeyTrack = makeTrack({ key: "" });
    delete undefinedKeyTrack.camelotKey;
    expect(() => scoreKeyCompatibility(undefinedKeyTrack, makeTrack({ camelotKey: "8A" }))).not.toThrow();
    expect(scoreKeyCompatibility(undefinedKeyTrack, makeTrack({ camelotKey: "8A" })).relation).toBe("unknown");
  });

  it("case 7: maps traditional musical keys and common enharmonic spellings to Camelot", () => {
    expect(normalizeKeyToCamelot("A minor")).toBe("8A");
    expect(normalizeKeyToCamelot("C major")).toBe("8B");
    expect(normalizeKeyToCamelot("E minor")).toBe("9A");
    expect(normalizeKeyToCamelot("G major")).toBe("9B");
    expect(normalizeKeyToCamelot("Bb major")).toBe("6B");
    expect(normalizeKeyToCamelot("Eb major")).toBe("5B");
    expect(normalizeKeyToCamelot("Ab minor")).toBe("1A");
    expect(toCamelot(makeTrack({ camelotKey: "", key: "C major" }))).toBe("8B");
  });

  it("distinguishes energy-direction relations only when targetEnergy asks for them", () => {
    const neutral = scoreKeyCompatibility(makeTrack({ camelotKey: "8A" }), makeTrack({ camelotKey: "9A" }));
    const up = scoreKeyCompatibility(makeTrack({ camelotKey: "8A" }), makeTrack({ camelotKey: "9A" }), { targetEnergy: "up" });
    const down = scoreKeyCompatibility(makeTrack({ camelotKey: "8A" }), makeTrack({ camelotKey: "7A" }), { targetEnergy: "down" });
    const boost = scoreKeyCompatibility(makeTrack({ camelotKey: "8A" }), makeTrack({ camelotKey: "10A" }));

    expect(neutral.relation).toBe("adjacent");
    expect(up.relation).toBe("energy_boost");
    expect(down.relation).toBe("energy_drop");
    expect(boost.relation).toBe("energy_boost");
    expect(boost.score).toBeGreaterThan(0.75);
  });

  it("supports special-effect Camelot relationships from the DJ Studio table", () => {
    expect(scoreKeyCompatibility(makeTrack({ camelotKey: "8A" }), makeTrack({ camelotKey: "7B" })).relation).toBe("diagonal_mix");
    expect(scoreKeyCompatibility(makeTrack({ camelotKey: "8A" }), makeTrack({ camelotKey: "3A" })).relation).toBe("jaws_mix");
    expect(scoreKeyCompatibility(makeTrack({ camelotKey: "8A" }), makeTrack({ camelotKey: "11B" })).relation).toBe("mood_shifter");
  });

  it("exports strict Camelot parsing for debugging and tests", () => {
    expect(parseCamelotKey("12b")).toEqual({ number: 12, letter: "B" });
    expect(parseCamelotKey("13A")).toBeNull();
  });
});
