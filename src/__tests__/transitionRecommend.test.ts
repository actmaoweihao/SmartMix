import { describe, expect, it } from "vitest";
import { explainTransition } from "../explain/explainTransition";
import { generatePracticePlan } from "../practice/practicePlan";
import { recommendNextTracks, recommendTransition } from "../transitions/recommend";
import { makeBeatGrid, makeTrack } from "./fixtures";
import type { SongSection } from "../analysis/types";

describe("recommendTransition", () => {
  it("recommends beatmix for close BPM, compatible key, low vocal intro/outro", () => {
    const outgoing = makeTrack({ bpm: 124, camelotKey: "8A", vocalDensityCurve: [{ time: 152, density: 0.05 }] });
    const incoming = makeTrack({ bpm: 126, camelotKey: "9A", vocalDensityCurve: [{ time: 0, density: 0.05 }] });
    const best = recommendTransition(outgoing, incoming, { maxComplexity: 5 })[0];

    expect(["beatmix", "bass_swap"]).toContain(best.method);
    expect(best.stepByStep.length).toBeGreaterThan(2);
  });

  it("recommends echo out or wide BPM loop for large BPM differences", () => {
    const outgoing = makeTrack({ bpm: 92, camelotKey: "8A" });
    const incoming = makeTrack({ bpm: 128, camelotKey: "3B" });
    const recs = recommendTransition(outgoing, incoming, { maxComplexity: 5 });
    const methods = recs.slice(0, 3).map((rec) => rec.method);

    expect(methods).not.toContain("beatmix");
    expect(methods.some((method) => method === "echo_out" || method === "breakdown_switch")).toBe(true);
    expect(recs.find((rec) => rec.method === "wide_bpm_loop")!.score).toBeGreaterThan(0);
  });

  it("lets vocal conflict push the recommendation toward quick cut", () => {
    const denseVocals = [
      { time: 0, density: 0.9 },
      { time: 30, density: 0.9 },
      { time: 60, density: 0.9 },
      { time: 120, density: 0.9 },
      { time: 152, density: 0.9 },
    ];
    const outgoing = makeTrack({ bpm: 118, camelotKey: "8A", vocalDensityCurve: denseVocals });
    const incoming = makeTrack({ bpm: 130, camelotKey: "2B", vocalDensityCurve: denseVocals });
    const best = recommendTransition(outgoing, incoming, { maxComplexity: 3 })[0];

    expect(["quick_cut", "echo_out"]).toContain(best.method);
  });

  it("downranks complex techniques in beginner mode", () => {
    const outgoing = makeTrack({ bpm: 90 });
    const incoming = makeTrack({ bpm: 132 });
    const beginner = recommendTransition(outgoing, incoming, { beginnerMode: true, maxComplexity: 5 });
    const advanced = recommendTransition(outgoing, incoming, { beginnerMode: false, maxComplexity: 5 });

    expect(beginner.find((rec) => rec.method === "wide_bpm_loop")!.score).toBeLessThan(advanced.find((rec) => rec.method === "wide_bpm_loop")!.score);
  });

  it("targetEnergy up prefers drop or chorus entry", () => {
    const outgoing = makeTrack({ bpm: 124, energyCurve: [{ time: 0, energy: 0.4 }, { time: 120, energy: 0.45 }] });
    const incoming = makeTrack({ bpm: 126, energyCurve: [{ time: 0, energy: 0.55 }, { time: 90, energy: 0.92 }] });
    const best = recommendTransition(outgoing, incoming, { targetEnergy: "up", maxComplexity: 5 })[0];

    expect(["drop", "chorus", "entry"]).toContain(best.incomingCue.role);
    expect(best.score).toBeGreaterThan(0.5);
  });

  it("returns explainable next track and practice outputs", () => {
    const current = makeTrack({ id: "a", title: "A" });
    const candidate = makeTrack({ id: "b", title: "B", bpm: 126 });
    const next = recommendNextTracks(current, [candidate], { maxResults: 1 })[0];
    const explanation = explainTransition(next.bestTransition);
    const practice = generatePracticePlan("beginner", [current, candidate]);

    expect(next.reasons.length).toBeGreaterThan(2);
    expect(next.bestTransition.debug?.reasons.length).toBeGreaterThan(0);
    expect(explanation).toContain("为什么");
    expect(practice.exercises).toHaveLength(4);
  });

  it("passes maxComplexity through next-track recommendations", () => {
    const current = makeTrack({ id: "a", title: "A", bpm: 92, camelotKey: "8A" });
    const candidate = makeTrack({ id: "b", title: "B", bpm: 128, camelotKey: "3B" });
    const next = recommendNextTracks(current, [candidate], { beginnerMode: true, maxComplexity: 2, maxResults: 1 })[0];

    expect(next.bestTransition.difficulty).toBeLessThanOrEqual(2);
    expect(next.bestTransition.method).not.toBe("wide_bpm_loop");
  });

  it("case 1 avoids long beatmix when both tracks are in dense vocal sections", () => {
    const outgoing = makeTrack({
      bpm: 100,
      camelotKey: "8A",
      sections: [section("chorus", 64, 96, 100, 0.95)],
      beatGrid: makeBeatGrid(130, 100),
      vocalDensityCurve: denseVocalCurve(130),
    });
    const incoming = makeTrack({
      bpm: 102,
      camelotKey: "8A",
      sections: [section("verse", 0, 40, 102, 0.95)],
      beatGrid: makeBeatGrid(130, 102),
      vocalDensityCurve: denseVocalCurve(130),
    });

    const recs = recommendTransition(outgoing, incoming, { beginnerMode: true, maxComplexity: 5 });
    const best = recs[0];
    const beatmix = recs.find((rec) => rec.method === "beatmix")!;

    expect(best.method === "quick_cut" || best.method === "echo_out").toBe(true);
    expect(beatmix.score).toBeLessThan(best.score);
    expect(explainTransition(best)).toContain("避免两段人声重叠");
    expect(best.debug?.vocalConflictScore).toBeGreaterThan(0.6);
  });

  it("case 2 prefers breakdown switch or echo out for wide BPM gap with a clean breakdown", () => {
    const outgoing = makeTrack({
      duration: 210,
      bpm: 128,
      camelotKey: "8A",
      sections: [section("breakdown", 160, 176, 128, 0.95), section("outro", 184, 210, 128, 0.8)],
      beatGrid: makeBeatGrid(210, 128),
      energyCurve: [{ time: 0, energy: 0.82 }, { time: 160, energy: 0.22 }, { time: 176, energy: 0.26 }],
      vocalDensityCurve: [{ time: 160, density: 0.05 }, { time: 176, density: 0.05 }],
    });
    const incoming = makeTrack({
      bpm: 95,
      camelotKey: "8A",
      sections: [section("intro", 0, 24, 95, 0.9), section("drop", 24, 64, 95, 0.82)],
      beatGrid: makeBeatGrid(160, 95),
      energyCurve: [{ time: 0, energy: 0.25 }, { time: 16, energy: 0.48 }, { time: 24, energy: 0.78 }],
      vocalDensityCurve: [{ time: 0, density: 0.05 }, { time: 16, density: 0.08 }],
    });

    const best = recommendTransition(outgoing, incoming, { beginnerMode: true, maxComplexity: 3 })[0];

    expect(["breakdown_switch", "echo_out"]).toContain(best.method);
    expect(best.method).not.toBe("beatmix");
    expect(best.stepByStep.some((step) => step.action === "enable_echo" || step.action === "filter_sweep")).toBe(true);
  });

  it("case 3 avoids long beatmix for incompatible keys in beginner mode", () => {
    const outgoing = makeTrack({ bpm: 120, camelotKey: "2A", vocalDensityCurve: lowVocalCurve(180) });
    const incoming = makeTrack({ bpm: 122, camelotKey: "9B", vocalDensityCurve: lowVocalCurve(180) });
    const best = recommendTransition(outgoing, incoming, { beginnerMode: true, maxComplexity: 3 })[0];

    expect(["fade", "quick_cut", "echo_out"]).toContain(best.method);
    expect(best.method).not.toBe("beatmix");
    expect(best.reason).toContain("调性不兼容，不建议长时间叠加");
  });

  it("case 4 pushes complex tricks behind simple compatible beginner transitions", () => {
    const outgoing = makeTrack({ bpm: 124, camelotKey: "8A", vocalDensityCurve: lowVocalCurve(180) });
    const incoming = makeTrack({ bpm: 126, camelotKey: "8A", vocalDensityCurve: lowVocalCurve(180) });
    const recs = recommendTransition(outgoing, incoming, { beginnerMode: true, maxComplexity: 5 });
    const best = recs[0];
    const wide = recs.find((rec) => rec.method === "wide_bpm_loop")!;
    const simpleBlend = recs.find((rec) => rec.method === "beatmix" || rec.method === "bass_swap")!;

    expect(["beatmix", "bass_swap"]).toContain(best.method);
    expect(wide.score).toBeLessThan(simpleBlend.score);
    expect(explainTransition(best)).toMatch(/先关 B 歌低频|第一拍|具体操作/);
  });

  it("case 5 uses a clear chorus or drop entry when target energy should rise", () => {
    const outgoing = makeTrack({
      bpm: 120,
      camelotKey: "8A",
      sections: [section("verse", 32, 96, 120, 0.8), section("outro", 130, 170, 120, 0.75)],
      beatGrid: makeBeatGrid(180, 120),
      energyCurve: [{ time: 32, energy: 0.35 }, { time: 96, energy: 0.42 }],
      vocalDensityCurve: [{ time: 32, density: 0.35 }, { time: 130, density: 0.12 }],
    });
    const incoming = makeTrack({
      bpm: 122,
      camelotKey: "8A",
      sections: [section("drop", 0, 48, 122, 0.95), section("chorus", 48, 96, 122, 0.88)],
      beatGrid: makeBeatGrid(150, 122),
      energyCurve: [{ time: 0, energy: 0.9 }, { time: 48, energy: 0.92 }],
      vocalDensityCurve: [{ time: 0, density: 0.08 }, { time: 48, density: 0.2 }],
    });

    const best = recommendTransition(outgoing, incoming, { targetEnergy: "up", beginnerMode: true, maxComplexity: 4 })[0];

    expect(["drop", "chorus"]).toContain(best.incomingCue.role);
    expect(best.reason).toContain("适合提升现场能量");
  });
});

function section(type: SongSection["type"], startTime: number, endTime: number, bpm: number, confidence: number): SongSection {
  const beat = 60 / bpm;
  return {
    type,
    startTime,
    endTime,
    startBeat: Math.round(startTime / beat),
    endBeat: Math.round(endTime / beat),
    confidence,
  };
}

function denseVocalCurve(duration: number) {
  return [
    { time: 0, density: 0.9 },
    { time: duration * 0.35, density: 0.92 },
    { time: duration * 0.65, density: 0.9 },
    { time: duration, density: 0.88 },
  ];
}

function lowVocalCurve(duration: number) {
  return [
    { time: 0, density: 0.05 },
    { time: duration * 0.5, density: 0.08 },
    { time: duration, density: 0.05 },
  ];
}
