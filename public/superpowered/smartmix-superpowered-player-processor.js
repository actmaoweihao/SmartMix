import { SuperpoweredWebAudio } from "./Superpowered.js";

class SmartMixSuperpoweredPlayer extends SuperpoweredWebAudio.AudioWorkletProcessor {
  onReady() {
    this.player = new this.Superpowered.AdvancedAudioPlayer(this.samplerate, 4, 2, 0, 0.501, 2, false);
    this.player.timeStretching = true;
    this.player.timeStretchingSound = 1;
    this.config = {
      url: "",
      startAt: 0,
      endAt: Number.POSITIVE_INFINITY,
      sourceOffset: 0,
      speed: 1,
      pitchSemitones: 0,
      preserveFormants: true,
    };
    this.loaded = false;
    this.started = false;
    this.ended = false;
  }

  onDestruct() {
    if (this.player) this.player.destruct();
  }

  onMessageFromMainScope(message) {
    try {
      if (message.SuperpoweredLoaded) {
        this.player.openMemory(this.Superpowered.arrayBufferToWASM(message.SuperpoweredLoaded.buffer), false, false);
        this.applyTransform();
        this.player.setPosition(Math.max(0, this.config.sourceOffset * 1000), true, false, false, false);
        this.loaded = true;
        this.started = false;
        this.sendMessageToMainScope({ loaded: true, url: message.SuperpoweredLoaded.url });
        return;
      }

      if (message.load) {
        this.config = { ...this.config, ...message };
        this.loaded = false;
        this.started = false;
        this.ended = false;
        this.Superpowered.downloadAndDecode(message.load, this);
        return;
      }

      if (message.transform) {
        this.config.speed = Number(message.speed) || 1;
        this.config.pitchSemitones = Number(message.pitchSemitones) || 0;
        this.config.preserveFormants = message.preserveFormants !== false;
        this.applyTransform();
        return;
      }

      if (message.stop && this.player) {
        this.ended = true;
        this.player.pause(0, 0);
      }
    } catch (error) {
      this.sendMessageToMainScope({ error: error.message || String(error) });
    }
  }

  applyTransform() {
    if (!this.player) return;
    this.player.timeStretching = true;
    this.player.timeStretchingSound = 1;
    this.player.playbackRate = Math.max(0.501, Math.min(2, Number(this.config.speed) || 1));
    this.player.pitchShiftCents = Math.round((Number(this.config.pitchSemitones) || 0) * 100);
    this.player.formantCorrection = this.config.preserveFormants ? 0.7 : 0;
  }

  processAudio(inputBuffer, outputBuffer, buffersize) {
    try {
      const now = currentFrame / this.samplerate;
      if (!this.loaded || this.ended || now < this.config.startAt || now >= this.config.endAt) {
        this.Superpowered.memorySet(outputBuffer.pointer, 0, buffersize * 8);
        if (this.started && now >= this.config.endAt) {
          this.ended = true;
          this.player.pause(0, 0);
        }
        return;
      }

      if (!this.started) {
        const elapsedSinceScheduledStart = Math.max(0, now - this.config.startAt);
        const catchupOffset = this.config.sourceOffset + elapsedSinceScheduledStart * (Number(this.config.speed) || 1);
        this.player.setPosition(Math.max(0, catchupOffset * 1000), false, false, false, false);
        this.player.play();
        this.started = true;
      }

      if (!this.player.processStereo(outputBuffer.pointer, false, buffersize, 1)) {
        this.Superpowered.memorySet(outputBuffer.pointer, 0, buffersize * 8);
      }
    } catch (error) {
      this.sendMessageToMainScope({ error: error.message || String(error) });
      this.Superpowered.memorySet(outputBuffer.pointer, 0, buffersize * 8);
    }
  }
}

if (typeof AudioWorkletProcessor === "function") {
  registerProcessor("SmartMixSuperpoweredPlayer", SmartMixSuperpoweredPlayer);
}

export default SmartMixSuperpoweredPlayer;
