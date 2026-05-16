import type { CuePoint, KeyScore, TransitionMethod, TransitionRecommendation } from "../analysis/types";

export type CrossfadeCurve = "linear" | "equal_power" | "logarithmic" | "smoothstep";

export type AudioBufferLike = {
  sampleRate: number;
  channels: Float32Array[];
};

export type GainAutomation = {
  stem?: keyof StemAutomationPlan;
  startTime: number;
  endTime: number;
  startGain: number;
  endGain: number;
  curve?: CrossfadeCurve;
};

export type FilterAutomation = {
  target: "outgoing" | "incoming";
  type: "lowpass" | "highpass" | "lowshelf";
  startTime: number;
  endTime: number;
  startValue: number;
  endValue: number;
};

export type EffectAutomation = {
  type: "echo" | "reverb" | "filter";
  target: "outgoing" | "incoming";
  startTime: number;
  endTime: number;
  value?: number | string;
};

export type StemPaths = {
  vocals?: string;
  drums?: string;
  bass?: string;
  other?: string;
  accompaniment?: string;
  fullMix?: string;
};

export interface StemSeparator {
  separate(audioPath: string, outputDir: string): Promise<StemPaths>;
}

export type StemAutomationPlan = {
  vocals: GainAutomation[];
  drums: GainAutomation[];
  bass: GainAutomation[];
  other: GainAutomation[];
  filters: FilterAutomation[];
  effects: EffectAutomation[];
};

export type SeamlessTransitionOptions = {
  targetMode?: "quality" | "fast";
  useStemSeparation?: boolean;
  stemEngine?: "demucs" | "spleeter" | "none";
  timeStretchEngine?: "rubberband" | "pyrubberband" | "librosa_fallback";
  preserveFormants?: boolean;
  maxTempoChangePercent?: number;
  maxPitchShiftSemitones?: number;
  previewDurationBeforeTransition?: number;
  previewDurationAfterTransition?: number;
  exportFormat?: "wav" | "mp3";
  beginnerSafeMode?: boolean;
};

export type GeneratedTransition = {
  audioPath: string;
  previewStartTime: number;
  previewEndTime: number;
  outgoingCue: CuePoint;
  incomingCue: CuePoint;
  method: TransitionMethod;
  processingReport: TransitionProcessingReport;
  warnings: string[];
};

export type TransitionProcessingReport = {
  bpmBefore: { outgoing: number; incoming: number };
  bpmAfter: { outgoing: number; incoming: number };
  keyBefore: { outgoing: string | null; incoming: string | null };
  keyAfter: { outgoing: string | null; incoming: string | null };
  tempoChangePercent: number;
  pitchShiftSemitones: number;
  usedStemSeparation: boolean;
  usedFormantPreservation: boolean;
  crossfadeCurve: CrossfadeCurve;
  overlapDuration: number;
  vocalConflictBefore: number;
  vocalConflictAfter: number;
  loudnessMatchDb: number;
  riskScore: number;
  explanation: string;
};

export type TransitionAlignment = {
  outgoingExitTime: number;
  incomingEntryTime: number;
  outgoingDownbeatTime: number;
  incomingDownbeatTime: number;
  overlapDuration: number;
  phraseAligned: boolean;
  alignmentConfidence: number;
};

export type TempoAdjustmentPlan = {
  shouldStretch: boolean;
  targetBpm: number;
  stretchRatio: number;
  tempoChangePercent: number;
  risk: "low" | "medium" | "high";
};

export type PitchShiftPlan = {
  shouldPitchShift: boolean;
  targetKey: string | null;
  semitones: number;
  expectedCamelotRelation: KeyScore["relation"];
  risk: "low" | "medium" | "high";
};

export type RenderTransitionInput = {
  outgoingAudioPath: string;
  incomingAudioPath: string;
  recommendation: TransitionRecommendation;
  alignment: TransitionAlignment;
  automationPlan: StemAutomationPlan;
  options: SeamlessTransitionOptions;
};
