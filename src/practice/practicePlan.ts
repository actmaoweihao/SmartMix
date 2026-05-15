import type { DJActionStep, PracticePlan, TrackAnalysis } from "../analysis/types";
import { recommendTransition } from "../transitions/recommend";

export function generatePracticePlan(userLevel: "beginner" | "intermediate" | "advanced", tracks: TrackAnalysis[]): PracticePlan {
  const pairs = pairTracks(tracks);
  const methodsByLevel = {
    beginner: ["fade", "end_to_end", "quick_cut", "beatmix"],
    intermediate: ["beatmix", "bass_swap", "echo_out", "breakdown_switch"],
    advanced: ["wide_bpm_loop", "acapella_mashup", "loop_build", "breakdown_switch"],
  }[userLevel];

  return {
    level: userLevel,
    exercises: methodsByLevel.map((method, index) => {
      const [a, b] = pairs[index % Math.max(1, pairs.length)] ?? [tracks[0], tracks[1] ?? tracks[0]];
      const rec = a && b ? recommendTransition(a, b, { beginnerMode: userLevel === "beginner" }).find((item) => item.method === method) : null;
      return {
        title: exerciseTitle(method),
        goal: exerciseGoal(method),
        tracks: [a?.title, b?.title].filter(Boolean) as string[],
        steps: rec?.stepByStep ?? fallbackSteps(method),
        successCriteria: ["按乐句第一拍启动下一首", "人声不长时间重叠", "低频切换时没有明显轰鸣", "能说清楚为什么选择这个接法"],
      };
    }),
  };
}

function pairTracks(tracks: TrackAnalysis[]): Array<[TrackAnalysis, TrackAnalysis]> {
  const pairs: Array<[TrackAnalysis, TrackAnalysis]> = [];
  for (let index = 0; index < tracks.length - 1; index += 1) pairs.push([tracks[index], tracks[index + 1]]);
  return pairs;
}

function exerciseTitle(method: string): string {
  return {
    fade: "练习渐隐接歌",
    end_to_end: "练习首尾强拍切换",
    quick_cut: "练习人声冲突快切",
    beatmix: "练习 8 小节对拍混音",
    bass_swap: "练习低频交换",
    echo_out: "练习回声退出",
    breakdown_switch: "练习空拍切换",
    wide_bpm_loop: "练习大 BPM 差 Loop",
    acapella_mashup: "练习清唱叠加",
    loop_build: "练习现场 Loop Build",
  }[method] ?? `练习 ${method}`;
}

function exerciseGoal(method: string): string {
  return `掌握 ${method} 的适用条件、cue 点选择和推子/EQ 操作。`;
}

function fallbackSteps(method: string): DJActionStep[] {
  return [{ atBeatOffset: 0, atTimeOffset: 0, action: "set_cue", targetDeck: "B", explanation: `准备 ${method} 练习 cue 点。` }];
}
