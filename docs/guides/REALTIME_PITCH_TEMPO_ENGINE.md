# Realtime Pitch/Tempo Engine

SmartMix exposes a reusable browser-side engine for realtime speed and pitch control.

It uses Superpowered WebAssembly + AudioWorklet for the high-quality path, with the app free to provide its own gain, EQ, fades, metering, or destination chain.

## Files

- `src/audio/realtimePitchTempo.js`
  - Public wrapper used by app code.
- `public/superpowered/smartmix-superpowered-player-processor.js`
  - AudioWorklet processor loaded by the browser as a static file.
- `public/superpowered/Superpowered.js`
- `public/superpowered/superpowered-npm.wasm`

The files under `public/superpowered/` must stay public/static. Vite cannot import public JS from `src/` modules.

## Main API

```js
import {
  DEFAULT_REALTIME_PITCH_TEMPO_TRANSFORM,
  createRealtimePitchTempoEngine,
  isRealtimePitchTempoActive,
  realtimeSpeed,
  realtimePitchRatio,
  realtimeTimelineSeconds,
} from "./audio/realtimePitchTempo";
```

`transform` shape:

```js
const transform = {
  ...DEFAULT_REALTIME_PITCH_TEMPO_TRANSFORM,
  enabled: true,
  speed: 1.2,
  pitchSemitones: -2,
  preserveFormants: true,
};
```

## Minimal Usage

```js
const engine = createRealtimePitchTempoEngine({
  setStatus: console.log,
  onFallback: (error) => console.warn("Realtime engine fallback", error),
  resolveTrackUrl: (track) => `/api/tracks/${track.id}/audio`,
  createProcessingChain({ context, item, localStart }) {
    const gain = context.createGain();
    gain.gain.setValueAtTime(1, localStart);
    gain.connect(context.destination);
    return {
      input: gain,
      activeNode: { trackId: item.track.id, mixerGain: gain },
    };
  },
});

const context = await engine.getAudioContext();
const result = await engine.scheduleTimeline({
  timeline,
  offset: 0,
  transform,
});

// Keep these so you can stop playback and update live controls.
activeSources.push(...result.controllers);
activeNodes.push(...result.nodes);

engine.updateControllers(activeSources, {
  ...transform,
  speed: 0.9,
  pitchSemitones: 3,
});
```

## Timeline Contract

`scheduleTimeline()` expects:

```js
{
  items: [{
    track: { id, duration, localId },
    start,
    end,
    sourceStart,
    fadeIn,
    fadeOut,
    fadeOutStart,
  }],
  total,
}
```

Use `realtimeTimelineSeconds(seconds, transform)` when building timelines under a speed transform. For example, a 120-second song at `1.2x` should occupy `100` timeline seconds.

## Error Handling

If Superpowered fails to initialize, load, or process audio, the engine sets `engine.failed = true` and calls `onFallback(error)`.

SmartMix currently responds by restarting playback through the basic realtime granular path. Other apps can choose to stop playback, display a message, or fall back to plain Web Audio.

## Licensing Note

Superpowered's JS/WASM SDK requires proper licensing for public distribution. The bundled trial key is for local evaluation only.
