export type SongSectionType = "intro" | "verse" | "chorus" | "bridge" | "breakdown" | "drop" | "outro";

export type SongSection = {
  type: SongSectionType;
  startTime: number;
  endTime: number;
  startBeat: number;
  endBeat: number;
  confidence: number;
};

export type BeatGridPoint = {
  time: number;
  beatIndex: number;
  barIndex: number;
  phraseIndex: number;
};

export type TrackAnalysis = {
  id: string;
  title: string;
  artist?: string;
  duration: number;
  bpm: number;
  key: string;
  camelotKey?: string;
  energyCurve: Array<{ time: number; energy: number }>;
  sections: SongSection[];
  beatGrid: BeatGridPoint[];
  hasVocalAtTime?: (time: number) => boolean;
  vocalDensityCurve?: Array<{ time: number; density: number }>;
};

export type CuePoint = {
  time: number;
  beatIndex: number;
  barIndex: number;
  phraseIndex: number;
  sectionType: SongSectionType;
  role: "entry" | "exit" | "drop" | "breakdown" | "outro" | "chorus";
  confidence: number;
};

export type BpmScore = {
  score: number;
  bpmDiff: number;
  category: "same" | "close" | "medium" | "wide";
  suggestedMethod: "beatmix" | "quick_cut" | "echo_out" | "breakdown_switch" | "wide_bpm_loop";
};

export type KeyScore = {
  score: number;
  relation: "same" | "adjacent" | "relative_major_minor" | "energy_boost" | "energy_drop" | "clash" | "unknown";
  explanation: string;
};

export type VocalConflictScore = {
  score: number;
  conflictRegions: Array<{ start: number; end: number; severity: number }>;
  recommendation:
    | "safe_to_blend"
    | "shorten_overlap"
    | "use_eq_cut"
    | "use_echo_out"
    | "use_quick_cut"
    | "use_instrumental_bridge";
};

export type DJActionStep = {
  atBeatOffset: number;
  atTimeOffset: number;
  action:
    | "press_play"
    | "set_cue"
    | "start_loop"
    | "halve_loop"
    | "increase_filter"
    | "decrease_filter"
    | "enable_echo"
    | "disable_echo"
    | "fade_out"
    | "fade_in"
      | "eq_low_cut"
      | "eq_low_swap"
      | "filter_sweep"
      | "crossfader_move"
      | "stop_track";
  targetDeck: "A" | "B";
  value?: number | string;
  explanation: string;
};

export type TransitionMethod =
  | "fade"
  | "end_to_end"
  | "quick_cut"
  | "beatmix"
  | "bass_swap"
  | "echo_out"
  | "filter_sweep"
  | "breakdown_switch"
  | "wide_bpm_loop"
  | "loop_build"
  | "instrumental_bridge"
  | "acapella_mashup";

export type TransitionRecommendation = {
  method: TransitionMethod;
  score: number;
  difficulty: 1 | 2 | 3 | 4 | 5;
  outgoingCue: CuePoint;
  incomingCue: CuePoint;
  overlapDuration: number;
  reason: string;
  stepByStep: DJActionStep[];
  debug?: TransitionDebugBreakdown;
};

export type TransitionDebugBreakdown = {
  method: TransitionMethod;
  finalScore: number;
  bpmScore: number;
  keyScore: number;
  vocalConflictScore: number;
  phraseScore: number;
  energyScore: number;
  beginnerScore: number;
  beginnerPenalty: number;
  sectionSuitability: number;
  methodFit: number;
  reasons: string[];
};

export type TransitionContext = {
  targetEnergy?: "up" | "down" | "keep";
  beginnerMode?: boolean;
  maxComplexity?: 1 | 2 | 3 | 4 | 5;
};

export type PracticePlan = {
  level: string;
  exercises: Array<{
    title: string;
    goal: string;
    tracks: string[];
    steps: DJActionStep[];
    successCriteria: string[];
  }>;
};
