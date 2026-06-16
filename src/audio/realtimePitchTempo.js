import { SuperpoweredGlue, SuperpoweredWebAudio } from "@superpoweredsdk/web";

export const DEFAULT_REALTIME_PITCH_TEMPO_TRANSFORM = Object.freeze({
  enabled: false,
  speed: 1,
  pitchSemitones: 0,
  preserveFormants: true,
  engine: "superpowered",
});

export function isRealtimePitchTempoActive(transform) {
  return Boolean(
    transform?.enabled &&
      (Math.abs(Number(transform.speed || 1) - 1) >= 0.001 || Math.abs(Number(transform.pitchSemitones) || 0) >= 0.001),
  );
}

export function realtimeSpeed(transform) {
  return isRealtimePitchTempoActive(transform) ? Math.max(0.001, Number(transform.speed) || 1) : 1;
}

export function realtimePitchRatio(transform) {
  return isRealtimePitchTempoActive(transform) ? 2 ** ((Number(transform.pitchSemitones) || 0) / 12) : 1;
}

export function realtimeTimelineSeconds(seconds, transform) {
  return Number(seconds || 0) / realtimeSpeed(transform);
}

export function createRealtimePitchTempoEngine(options = {}) {
  return new RealtimePitchTempoEngine(options);
}

class RealtimePitchTempoEngine {
  constructor({
    licenseKey = "ExampleLicenseKey-WillExpire-OnNextUpdate",
    processorUrl = `${window.location.origin}/superpowered/smartmix-superpowered-player-processor.js`,
    wasmUrl = `${window.location.origin}/superpowered/superpowered-npm.wasm`,
    minimumSampleRate = 44100,
    setStatus = () => {},
    onFallback = () => {},
    resolveTrackUrl,
    createProcessingChain,
  } = {}) {
    this.licenseKey = licenseKey;
    this.processorUrl = processorUrl;
    this.wasmUrl = wasmUrl;
    this.minimumSampleRate = minimumSampleRate;
    this.setStatus = setStatus;
    this.onFallback = onFallback;
    this.resolveTrackUrl = resolveTrackUrl;
    this.createProcessingChain = createProcessingChain;
    this.glue = null;
    this.manager = null;
    this.initializing = null;
    this.failed = false;
    this.error = "";
  }

  async getAudioContext() {
    if (this.manager) {
      if (this.manager.audioContext.state === "suspended") await this.manager.audioContext.resume();
      return this.manager.audioContext;
    }
    if (!this.initializing) {
      this.initializing = (async () => {
        const glue = await SuperpoweredGlue.Instantiate(this.licenseKey, this.wasmUrl);
        const manager = new SuperpoweredWebAudio(this.minimumSampleRate, glue);
        this.glue = glue;
        this.manager = manager;
        return manager;
      })();
    }
    try {
      const manager = await this.initializing;
      if (manager.audioContext.state === "suspended") await manager.audioContext.resume();
      this.failed = false;
      this.error = "";
      return manager.audioContext;
    } catch (error) {
      this.failed = true;
      this.error = error.message || "Superpowered initialization failed";
      this.setStatus(`${this.error}; falling back to basic realtime preview`);
      this.onFallback(this.error);
      throw error;
    }
  }

  async scheduleTimeline({ timeline, offset, transform }) {
    if (!this.manager || this.failed) return { started: 0, controllers: [], nodes: [] };
    const context = this.manager.audioContext;
    const startAt = context.currentTime + 0.16;
    const controllers = [];
    const nodes = [];

    for (const item of timeline.items) {
      if (item.end <= offset) continue;
      if (!item.track?.id) continue;
      const sourceOffset = item.sourceStart + Math.max(0, offset - item.start);
      const originalOffset = sourceOffset * realtimeSpeed(transform);
      if (originalOffset >= item.track.duration) continue;
      const scheduled = await this.scheduleItem({
        context,
        item,
        startAt,
        offset,
        sourceOffset,
        originalOffset,
        transform,
      });
      if (!scheduled) continue;
      controllers.push(scheduled.controller);
      nodes.push(scheduled.activeNode);
    }

    return { started: controllers.length, controllers, nodes };
  }

  async scheduleItem({ context, item, startAt, offset, sourceOffset, originalOffset, transform }) {
    if (typeof this.resolveTrackUrl !== "function") throw new Error("resolveTrackUrl is required");
    if (typeof this.createProcessingChain !== "function") throw new Error("createProcessingChain is required");

    let loaded = false;
    let loadTimeout = null;
    const node = await this.manager.createAudioNodeAsync(
      this.processorUrl,
      "SmartMixSuperpoweredPlayer",
      (message) => {
        if (message?.loaded) {
          loaded = true;
          if (loadTimeout) window.clearTimeout(loadTimeout);
          loadTimeout = null;
        }
        if (message?.error) this.failAndFallback(message.error);
      },
      0,
      1,
    );

    const localStart = startAt + Math.max(0, item.start - offset);
    const endAt = startAt + Math.max(0, item.end - offset);
    const chain = this.createProcessingChain({
      context,
      item,
      localStart,
      offset,
      sourceOffset,
      endAt,
    });
    node.connect(chain.input);

    node.sendMessageToAudioScope({
      load: this.resolveTrackUrl(item.track),
      startAt: localStart,
      endAt,
      sourceOffset: originalOffset,
      speed: realtimeSpeed(transform),
      pitchSemitones: Number(transform.pitchSemitones) || 0,
      preserveFormants: transform.preserveFormants !== false,
    });

    loadTimeout = window.setTimeout(() => {
      if (loaded || this.failed) return;
      this.failAndFallback("Superpowered audio load timed out");
    }, 5000);

    const controller = {
      kind: "superpowered",
      node,
      stop() {
        if (loadTimeout) window.clearTimeout(loadTimeout);
        try {
          node.sendMessageToAudioScope({ stop: true });
        } catch {
          // Node may already be torn down.
        }
        try {
          node.disconnect();
        } catch {
          // Already disconnected.
        }
        try {
          if (typeof node.destruct === "function") node.destruct();
        } catch {
          // Best-effort cleanup.
        }
        if (typeof chain.dispose === "function") chain.dispose();
      },
      updateTransform(nextTransform) {
        node.sendMessageToAudioScope({
          transform: true,
          speed: realtimeSpeed(nextTransform),
          pitchSemitones: Number(nextTransform.pitchSemitones) || 0,
          preserveFormants: nextTransform.preserveFormants !== false,
        });
      },
    };

    return {
      controller,
      activeNode: {
        ...chain.activeNode,
        source: controller,
      },
    };
  }

  updateControllers(controllers, transform) {
    controllers.forEach((controller) => {
      if (controller?.kind === "superpowered" && typeof controller.updateTransform === "function") {
        controller.updateTransform(transform);
      }
    });
  }

  resetFailure() {
    this.failed = false;
    this.error = "";
  }

  failAndFallback(message) {
    this.failed = true;
    this.error = message || "Superpowered realtime engine failed";
    this.setStatus(`Superpowered failed; falling back to basic realtime preview: ${this.error}`);
    this.onFallback(this.error);
  }
}
