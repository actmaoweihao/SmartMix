import "./styles.css";

const API = "http://127.0.0.1:8000";

const state = {
  tracks: [],
  originalOrder: [],
  selectedId: null,
  audioContext: null,
  activeSources: [],
  activeNodes: [],
  playStartContextTime: 0,
  playStartOffset: 0,
  playbackOffset: 0,
  timer: null,
  isPlaying: false,
  isExporting: false,
  projects: [],
  settings: {
    sortMode: "recommended",
    crossfade: 8,
    autoTransition: true,
    beatSync: false,
    filterMode: "lowpassSweep",
    exportFormat: "mp3",
    eq: { low: 0, mid: 0, high: 0 },
  },
};

const app = document.querySelector("#app");
app.innerHTML = `
  <main class="app-shell">
    <header class="topbar">
      <section>
        <p class="eyebrow">FastAPI Auto DJ</p>
        <h1>SmartMix</h1>
      </section>
      <div class="top-actions">
        <label class="upload-button" for="fileInput">
          <span>选择音频</span>
          <input id="fileInput" type="file" accept="audio/*" multiple />
        </label>
        <button id="refreshProjects" class="ghost" type="button">刷新项目</button>
      </div>
    </header>

    <section class="studio-grid">
      <aside class="control-panel" aria-label="混音控制">
        <div class="meter-block"><span>Tracks</span><strong id="trackCount">0</strong></div>
        <div class="meter-block"><span>Mix Time</span><strong id="mixDuration">00:00</strong></div>
        <div class="meter-block wide"><span>Status</span><strong id="statusText">等待后端</strong></div>

        <label class="field">
          <span>排序策略</span>
          <select id="sortMode">
            <option value="recommended">综合推荐</option>
            <option value="harmonic">谐和优先</option>
            <option value="bpmAsc">BPM 升序</option>
            <option value="bpmDesc">BPM 降序</option>
            <option value="energyArc">能量弧线</option>
            <option value="original">原始顺序</option>
          </select>
        </label>

        <label class="field">
          <span>过渡时长 <b id="crossfadeValue">8s</b></span>
          <input id="crossfade" type="range" min="2" max="24" value="8" />
        </label>

        <label class="toggle"><input id="autoTransition" type="checkbox" checked /><span>首尾能量自动微调</span></label>
        <label class="toggle"><input id="beatSync" type="checkbox" /><span>导出时节拍同步</span></label>

        <label class="field">
          <span>过渡滤波</span>
          <select id="filterMode">
            <option value="lowpassSweep">低通扫尾</option>
            <option value="highpassLift">高通抬入</option>
            <option value="none">关闭滤波</option>
          </select>
        </label>

        <div class="eq-bank">
          <label><span>Low</span><input id="eqLow" type="range" min="-1" max="1" value="0" step="0.05" /></label>
          <label><span>Mid</span><input id="eqMid" type="range" min="-1" max="1" value="0" step="0.05" /></label>
          <label><span>High</span><input id="eqHigh" type="range" min="-1" max="1" value="0" step="0.05" /></label>
        </div>

        <div class="button-row">
          <button id="sortButton" type="button">应用排序</button>
          <button id="playButton" type="button">预览</button>
          <button id="stopButton" type="button" class="secondary">停止</button>
        </div>

        <div class="export-row">
          <select id="exportFormat" aria-label="导出格式">
            <option value="mp3">MP3</option>
            <option value="wav">WAV</option>
          </select>
          <button id="exportButton" type="button" class="export">导出</button>
        </div>
        <a id="downloadLink" class="download-link" hidden>下载混音文件</a>

        <div class="project-box">
          <input id="projectName" type="text" placeholder="项目名" />
          <button id="saveProject" type="button">保存项目</button>
          <select id="projectList"><option value="">选择已保存项目</option></select>
          <button id="loadProject" type="button" class="secondary">加载项目</button>
        </div>
      </aside>

      <section class="workspace" aria-label="曲目工作台">
        <div id="dropZone" class="drop-zone">
          <div><span class="pulse-dot"></span><p>拖入音频文件，后端会立刻分析 BPM、调性和能量</p></div>
        </div>

        <section class="transport">
          <button id="restartButton" type="button" class="iconish">从头播放</button>
          <input id="mixProgress" type="range" min="0" max="0" value="0" step="0.01" />
          <time id="playTime">00:00 / 00:00</time>
        </section>

        <section class="waveform-panel">
          <div class="wave-head">
            <div>
              <span class="tiny-label">Selected Waveform</span>
              <strong id="selectedTitle">未选择曲目</strong>
            </div>
            <span id="handleReadout">入点 -- / 出点 --</span>
          </div>
          <canvas id="waveCanvas" width="1200" height="260"></canvas>
        </section>

        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>曲目</th>
                <th>时长</th>
                <th>BPM</th>
                <th>调性</th>
                <th>能量</th>
                <th>过渡</th>
                <th>状态</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody id="trackTable"></tbody>
          </table>
        </div>
      </section>
    </section>
  </main>
`;

const els = {
  fileInput: document.querySelector("#fileInput"),
  dropZone: document.querySelector("#dropZone"),
  trackTable: document.querySelector("#trackTable"),
  trackCount: document.querySelector("#trackCount"),
  mixDuration: document.querySelector("#mixDuration"),
  statusText: document.querySelector("#statusText"),
  sortMode: document.querySelector("#sortMode"),
  crossfade: document.querySelector("#crossfade"),
  crossfadeValue: document.querySelector("#crossfadeValue"),
  autoTransition: document.querySelector("#autoTransition"),
  beatSync: document.querySelector("#beatSync"),
  filterMode: document.querySelector("#filterMode"),
  exportFormat: document.querySelector("#exportFormat"),
  eqLow: document.querySelector("#eqLow"),
  eqMid: document.querySelector("#eqMid"),
  eqHigh: document.querySelector("#eqHigh"),
  sortButton: document.querySelector("#sortButton"),
  playButton: document.querySelector("#playButton"),
  stopButton: document.querySelector("#stopButton"),
  restartButton: document.querySelector("#restartButton"),
  exportButton: document.querySelector("#exportButton"),
  downloadLink: document.querySelector("#downloadLink"),
  mixProgress: document.querySelector("#mixProgress"),
  playTime: document.querySelector("#playTime"),
  waveCanvas: document.querySelector("#waveCanvas"),
  selectedTitle: document.querySelector("#selectedTitle"),
  handleReadout: document.querySelector("#handleReadout"),
  projectName: document.querySelector("#projectName"),
  saveProject: document.querySelector("#saveProject"),
  projectList: document.querySelector("#projectList"),
  loadProject: document.querySelector("#loadProject"),
  refreshProjects: document.querySelector("#refreshProjects"),
};

const wave = {
  ctx: els.waveCanvas.getContext("2d"),
  dragging: null,
};

bindEvents();
pingBackend();
loadProjectList();
render();

function bindEvents() {
  els.fileInput.addEventListener("change", (event) => addFiles([...event.target.files]));
  els.sortButton.addEventListener("click", applySort);
  els.playButton.addEventListener("click", () => previewMix(state.playbackOffset));
  els.restartButton.addEventListener("click", () => previewMix(0));
  els.stopButton.addEventListener("click", stopPreview);
  els.exportButton.addEventListener("click", exportMix);
  els.saveProject.addEventListener("click", saveProject);
  els.loadProject.addEventListener("click", loadSelectedProject);
  els.refreshProjects.addEventListener("click", loadProjectList);

  els.sortMode.addEventListener("change", syncSettings);
  els.crossfade.addEventListener("input", syncSettings);
  els.autoTransition.addEventListener("change", syncSettings);
  els.beatSync.addEventListener("change", syncSettings);
  els.filterMode.addEventListener("change", syncSettings);
  els.exportFormat.addEventListener("change", syncSettings);
  [els.eqLow, els.eqMid, els.eqHigh].forEach((input) => input.addEventListener("input", syncSettings));

  els.mixProgress.addEventListener("input", () => {
    state.playbackOffset = Number(els.mixProgress.value);
    renderTransport();
  });
  els.mixProgress.addEventListener("change", () => {
    if (state.isPlaying) previewMix(Number(els.mixProgress.value));
  });

  ["dragenter", "dragover"].forEach((type) => {
    els.dropZone.addEventListener(type, (event) => {
      event.preventDefault();
      els.dropZone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((type) => {
    els.dropZone.addEventListener(type, (event) => {
      event.preventDefault();
      els.dropZone.classList.remove("dragging");
    });
  });
  els.dropZone.addEventListener("drop", (event) => {
    addFiles([...event.dataTransfer.files].filter((file) => file.type.startsWith("audio/")));
  });

  els.waveCanvas.addEventListener("pointerdown", wavePointerDown);
  els.waveCanvas.addEventListener("pointermove", wavePointerMove);
  window.addEventListener("pointerup", () => {
    wave.dragging = null;
  });
}

function syncSettings() {
  state.settings.sortMode = els.sortMode.value;
  state.settings.crossfade = Number(els.crossfade.value);
  state.settings.autoTransition = els.autoTransition.checked;
  state.settings.beatSync = els.beatSync.checked;
  state.settings.filterMode = els.filterMode.value;
  state.settings.exportFormat = els.exportFormat.value;
  state.settings.eq.low = Number(els.eqLow.value);
  state.settings.eq.mid = Number(els.eqMid.value);
  state.settings.eq.high = Number(els.eqHigh.value);
  render();
}

async function pingBackend() {
  try {
    await fetchJson(`${API}/api/health`);
    setStatus("后端已连接");
  } catch {
    setStatus("后端未启动，运行 pnpm dev");
  }
}

async function addFiles(files) {
  if (!files.length) return;
  for (const file of files) {
    const tempId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
    const track = {
      id: tempId,
      localId: tempId,
      file,
      name: file.name,
      status: "uploading",
      error: "",
      buffer: null,
      peaks: [],
      duration: 0,
      bpm: null,
      key: "未知",
      key_index: null,
      mode: null,
      energy: 0,
      intro_low: 0,
      outro_low: 0,
      introPoint: 0,
      outroPoint: 0,
    };
    state.tracks.push(track);
    state.originalOrder.push(track.localId);
    state.selectedId ||= track.localId;
    render();
    await decodeLocal(track);
    await uploadAndAnalyze(track);
  }
}

async function decodeLocal(track) {
  try {
    const context = await getAudioContext();
    const arrayBuffer = await track.file.arrayBuffer();
    track.buffer = await context.decodeAudioData(arrayBuffer.slice(0));
    track.duration = track.buffer.duration;
    track.peaks = peaksFromBuffer(track.buffer, 900);
    track.introPoint = Math.min(8, Math.max(1, track.duration * 0.12));
    track.outroPoint = Math.max(track.introPoint + 1, track.duration - Math.min(8, track.duration * 0.12));
    render();
  } catch (error) {
    track.status = "error";
    track.error = error.message || "浏览器无法解码";
  }
}

async function uploadAndAnalyze(track) {
  try {
    setStatus(`后端分析 ${track.name}`);
    const form = new FormData();
    form.append("file", track.file);
    const result = await fetchJson(`${API}/api/tracks`, { method: "POST", body: form });
    track.id = result.id;
    track.status = "ready";
    track.duration = result.duration || track.duration;
    track.bpm = result.bpm;
    track.key = result.key || "未知";
    track.key_index = result.key_index;
    track.mode = result.mode;
    track.energy = result.energy || 0;
    track.intro_low = result.intro_low || 0;
    track.outro_low = result.outro_low || 0;
    track.peaks = result.peaks?.length ? result.peaks : track.peaks;
    track.introPoint = clamp(track.intro_low || track.introPoint, 0.5, Math.max(0.5, track.duration * 0.35));
    track.outroPoint = clamp(track.duration - (track.outro_low || state.settings.crossfade), track.duration * 0.55, Math.max(track.duration - 0.5, 0.5));
    setStatus(`完成分析 ${track.name}`);
  } catch (error) {
    track.status = "error";
    track.error = error.message || "后端分析失败";
    setStatus(`分析失败：${track.name}`);
  } finally {
    render();
  }
}

async function getAudioContext() {
  if (!state.audioContext) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    state.audioContext = new AudioContextClass();
  }
  if (state.audioContext.state === "suspended") await state.audioContext.resume();
  return state.audioContext;
}

function peaksFromBuffer(buffer, bins) {
  const channels = [];
  for (let i = 0; i < buffer.numberOfChannels; i += 1) channels.push(buffer.getChannelData(i));
  const length = buffer.length;
  const chunk = Math.max(1, Math.floor(length / bins));
  const peaks = [];
  for (let offset = 0; offset < length; offset += chunk) {
    let max = 0;
    const end = Math.min(offset + chunk, length);
    for (const channel of channels) {
      for (let i = offset; i < end; i += 1) max = Math.max(max, Math.abs(channel[i]));
    }
    peaks.push(max);
  }
  const peak = Math.max(...peaks, 1);
  return peaks.map((value) => value / peak);
}

function applySort() {
  const ready = playableTracks();
  if (ready.length < 2) return;
  const sorted = sortTracks(ready, state.settings.sortMode);
  const unresolved = state.tracks.filter((track) => track.status !== "ready");
  state.tracks = [...sorted, ...unresolved];
  state.playbackOffset = 0;
  setStatus("已应用排序");
  render();
}

function sortTracks(tracks, mode) {
  const copy = [...tracks];
  if (mode === "original") return copy.sort((a, b) => state.originalOrder.indexOf(a.localId) - state.originalOrder.indexOf(b.localId));
  if (mode === "bpmAsc") return copy.sort((a, b) => safeBpm(a) - safeBpm(b));
  if (mode === "bpmDesc") return copy.sort((a, b) => safeBpm(b) - safeBpm(a));
  if (mode === "energyArc") return energyArc(copy);
  return greedySort(copy, mode === "harmonic" ? "harmonic" : "recommended");
}

function safeBpm(track) {
  return track.bpm || 120;
}

function energyArc(tracks) {
  const sorted = [...tracks].sort((a, b) => a.energy - b.energy);
  const high = sorted.splice(Math.ceil(sorted.length * 0.65));
  return [...sorted, ...high.reverse()];
}

function greedySort(tracks, mode) {
  const remaining = [...tracks].sort((a, b) => a.energy - b.energy);
  const result = [remaining.shift()];
  while (remaining.length) {
    const current = result[result.length - 1];
    let bestIndex = 0;
    let bestScore = Infinity;
    remaining.forEach((candidate, index) => {
      const score = transitionScore(current, candidate, mode);
      if (score < bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    });
    result.push(remaining.splice(bestIndex, 1)[0]);
  }
  return result;
}

function transitionScore(a, b, mode) {
  const bpmDelta = Math.min(1, Math.abs(safeBpm(a) - safeBpm(b)) / 60);
  const harmonic = keyDistance(a, b);
  const energyDelta = Math.abs(a.energy - b.energy);
  if (mode === "harmonic") return harmonic * 0.6 + bpmDelta * 0.3 + energyDelta * 0.1;
  return bpmDelta * 0.55 + harmonic * 0.3 + energyDelta * 0.15;
}

function keyDistance(a, b) {
  if (a.key_index === null || b.key_index === null) return 0.5;
  const raw = Math.abs(a.key_index - b.key_index);
  const circular = Math.min(raw, 12 - raw) / 6;
  const modePenalty = a.mode === b.mode ? 0 : 0.15;
  return Math.min(1, circular + modePenalty);
}

async function previewMix(offset = 0) {
  const tracks = playableTracks().filter((track) => track.buffer);
  if (!tracks.length) return;
  stopPreview({ keepStatus: true });
  const context = await getAudioContext();
  const timeline = buildTimeline(tracks);
  const started = scheduleMix(context, tracks, timeline, Math.min(offset, timeline.total));
  if (!started) return;
  state.isPlaying = true;
  state.playStartContextTime = context.currentTime;
  state.playStartOffset = offset;
  state.playbackOffset = offset;
  state.timer = window.setInterval(tickPlayback, 80);
  setStatus("正在预览混音");
  render();
}

function scheduleMix(context, tracks, timeline, offset) {
  const startAt = context.currentTime + 0.08;
  let started = 0;
  timeline.items.forEach((item, index) => {
    if (item.end <= offset) return;
    const track = tracks[index];
    const sourceOffset = Math.max(0, offset - item.start);
    if (sourceOffset >= track.buffer.duration) return;

    const source = context.createBufferSource();
    source.buffer = track.buffer;
    const gain = context.createGain();
    const low = context.createBiquadFilter();
    const mid = context.createBiquadFilter();
    const high = context.createBiquadFilter();
    const transitionFilter = context.createBiquadFilter();
    configureEq(low, mid, high, transitionFilter, context.currentTime);
    source.connect(low).connect(mid).connect(high).connect(transitionFilter).connect(gain).connect(context.destination);

    const localStart = startAt + Math.max(0, item.start - offset);
    applyPreviewEnvelope(gain.gain, transitionFilter, localStart, track.buffer.duration, item.fadeIn, item.fadeOut, sourceOffset);
    source.start(localStart, sourceOffset);
    source.onended = () => {
      state.activeSources = state.activeSources.filter((itemSource) => itemSource !== source);
    };
    state.activeSources.push(source);
    state.activeNodes.push({ gain, low, mid, high, transitionFilter });
    started += 1;
  });
  return started > 0;
}

function configureEq(low, mid, high, transitionFilter, now) {
  low.type = "lowshelf";
  low.frequency.value = 220;
  low.gain.value = state.settings.eq.low * 10;
  mid.type = "peaking";
  mid.frequency.value = 1200;
  mid.Q.value = 0.9;
  mid.gain.value = state.settings.eq.mid * 9;
  high.type = "highshelf";
  high.frequency.value = 3400;
  high.gain.value = state.settings.eq.high * 10;
  transitionFilter.type = state.settings.filterMode === "highpassLift" ? "highpass" : "lowpass";
  transitionFilter.frequency.setValueAtTime(state.settings.filterMode === "none" ? 20000 : 16000, now);
}

function applyPreviewEnvelope(param, filterNode, startsAt, duration, fadeIn, fadeOut, offset) {
  const endAt = startsAt + Math.max(0, duration - offset);
  const absolutePosition = offset;
  param.cancelScheduledValues(startsAt);
  const inProgressFadeIn = fadeIn > 0 && absolutePosition < fadeIn;
  param.setValueAtTime(inProgressFadeIn ? absolutePosition / fadeIn : 1, startsAt);
  if (inProgressFadeIn) param.linearRampToValueAtTime(1, startsAt + (fadeIn - absolutePosition));

  const fadeOutStartOriginal = Math.max(0, duration - fadeOut);
  const fadeOutStart = startsAt + Math.max(0, fadeOutStartOriginal - offset);
  if (fadeOut > 0 && endAt > fadeOutStart) {
    param.setValueAtTime(1, fadeOutStart);
    param.linearRampToValueAtTime(0, endAt);
    if (state.settings.filterMode === "lowpassSweep") {
      filterNode.type = "lowpass";
      filterNode.frequency.setValueAtTime(16000, fadeOutStart);
      filterNode.frequency.exponentialRampToValueAtTime(900, endAt);
    }
  }

  if (state.settings.filterMode === "highpassLift" && fadeIn > 0) {
    filterNode.type = "highpass";
    filterNode.frequency.setValueAtTime(700, startsAt);
    filterNode.frequency.exponentialRampToValueAtTime(35, startsAt + fadeIn);
  }
}

function stopPreview(options = {}) {
  state.activeSources.forEach((source) => {
    try {
      source.stop();
    } catch {
      // The source may have already ended.
    }
  });
  state.activeSources = [];
  state.activeNodes = [];
  state.isPlaying = false;
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  if (!options.keepStatus) setStatus(playableTracks().length ? "已停止" : "等待上传");
  render();
}

function tickPlayback() {
  if (!state.audioContext || !state.isPlaying) return;
  const total = getMixDurationSeconds();
  state.playbackOffset = Math.min(total, state.playStartOffset + (state.audioContext.currentTime - state.playStartContextTime));
  if (state.playbackOffset >= total) {
    stopPreview({ keepStatus: true });
    setStatus("预览结束");
  }
  renderTransport();
  drawWaveform();
}

async function exportMix() {
  const tracks = playableTracks();
  if (!tracks.length) return;
  stopPreview({ keepStatus: true });
  state.isExporting = true;
  setStatus(`后端正在导出 ${state.settings.exportFormat.toUpperCase()}`);
  render();

  try {
    const payload = {
      trackIds: tracks.map((track) => track.id),
      tracks: tracks.map(exportableTrack),
      settings: state.settings,
      format: state.settings.exportFormat,
    };
    const result = await fetchJson(`${API}/api/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    els.downloadLink.href = `${API}${result.url}`;
    els.downloadLink.download = result.filename;
    els.downloadLink.hidden = false;
    setStatus("导出完成");
  } catch (error) {
    setStatus(error.message || "导出失败");
  } finally {
    state.isExporting = false;
    render();
  }
}

async function saveProject() {
  const name = els.projectName.value.trim() || `SmartMix ${new Date().toLocaleString()}`;
  try {
    await fetchJson(`${API}/api/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        tracks: state.tracks.map(exportableTrack),
        settings: state.settings,
      }),
    });
    setStatus("项目已保存");
    await loadProjectList();
  } catch (error) {
    setStatus(error.message || "项目保存失败");
  }
}

async function loadProjectList() {
  try {
    const result = await fetchJson(`${API}/api/projects`);
    state.projects = result.projects || [];
    els.projectList.innerHTML = `<option value="">选择已保存项目</option>${state.projects.map((project) => `<option value="${project.id}">${escapeHtml(project.name)}</option>`).join("")}`;
  } catch {
    state.projects = [];
  }
}

async function loadSelectedProject() {
  if (!els.projectList.value) return;
  try {
    const project = await fetchJson(`${API}/api/projects/${els.projectList.value}`);
    state.settings = { ...state.settings, ...project.settings };
    state.tracks = await Promise.all(project.tracks.map(rehydrateTrack));
    state.originalOrder = state.tracks.map((track) => track.localId);
    state.selectedId = state.tracks[0]?.localId || null;
    applySettingsToControls();
    setStatus("项目已加载");
    render();
  } catch (error) {
    setStatus(error.message || "项目加载失败");
  }
}

async function rehydrateTrack(saved) {
  const context = await getAudioContext();
  const response = await fetch(`${API}/api/tracks/${saved.id}/audio`);
  if (!response.ok) throw new Error("加载项目音频失败");
  const arrayBuffer = await response.arrayBuffer();
  const buffer = await context.decodeAudioData(arrayBuffer.slice(0));
  return {
    ...saved,
    file: null,
    buffer,
    localId: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`,
    peaks: saved.peaks?.length ? saved.peaks : peaksFromBuffer(buffer, 900),
    status: "ready",
    error: "",
  };
}

function exportableTrack(track) {
  return {
    id: track.id,
    name: track.name,
    duration: track.duration,
    bpm: track.bpm,
    key: track.key,
    key_index: track.key_index,
    mode: track.mode,
    energy: track.energy,
    intro_low: track.intro_low,
    outro_low: track.outro_low,
    introPoint: track.introPoint,
    outroPoint: track.outroPoint,
    peaks: track.peaks,
    status: track.status,
  };
}

function applySettingsToControls() {
  els.sortMode.value = state.settings.sortMode;
  els.crossfade.value = state.settings.crossfade;
  els.autoTransition.checked = state.settings.autoTransition;
  els.beatSync.checked = state.settings.beatSync;
  els.filterMode.value = state.settings.filterMode;
  els.exportFormat.value = state.settings.exportFormat;
  els.eqLow.value = state.settings.eq.low;
  els.eqMid.value = state.settings.eq.mid;
  els.eqHigh.value = state.settings.eq.high;
}

function buildTimeline(tracks = playableTracks()) {
  const items = [];
  let cursor = 0;
  tracks.forEach((track, index) => {
    const fadeIn = index === 0 ? 0 : getTransitionDuration(tracks[index - 1], track);
    const fadeOut = index < tracks.length - 1 ? getTransitionDuration(track, tracks[index + 1]) : 0;
    const start = index === 0 ? 0 : cursor - fadeIn;
    const end = start + track.duration;
    items.push({ track, start, end, fadeIn, fadeOut });
    cursor = end;
  });
  return { items, total: items.at(-1)?.end || 0 };
}

function getTransitionDuration(prev, next) {
  const requested = state.settings.crossfade;
  const maxByLength = Math.max(0.5, Math.min(prev.duration, next.duration) * 0.35);
  let actual = requested;
  if (Number.isFinite(prev.outroPoint) && Number.isFinite(next.introPoint)) {
    actual = Math.min(actual, Math.max(0.5, prev.duration - prev.outroPoint), Math.max(0.5, next.introPoint));
  }
  if (state.settings.autoTransition) {
    actual = Math.max(2, Math.min(actual, (prev.outro_low || 0) + (next.intro_low || 0) + 2));
  }
  return Math.min(actual, maxByLength);
}

function getMixDurationSeconds() {
  return buildTimeline().total;
}

function playableTracks() {
  return state.tracks.filter((track) => track.status === "ready");
}

function selectedTrack() {
  return state.tracks.find((track) => track.localId === state.selectedId) || null;
}

function moveTrack(localId, direction) {
  const index = state.tracks.findIndex((track) => track.localId === localId);
  const target = index + direction;
  if (index < 0 || target < 0 || target >= state.tracks.length) return;
  [state.tracks[index], state.tracks[target]] = [state.tracks[target], state.tracks[index]];
  render();
}

function removeTrack(localId) {
  state.tracks = state.tracks.filter((track) => track.localId !== localId);
  state.originalOrder = state.originalOrder.filter((item) => item !== localId);
  if (state.selectedId === localId) state.selectedId = state.tracks[0]?.localId || null;
  render();
}

function selectTrack(localId) {
  state.selectedId = localId;
  render();
}

function wavePointerDown(event) {
  const track = selectedTrack();
  if (!track?.duration) return;
  const rect = els.waveCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const width = rect.width;
  const introX = (track.introPoint / track.duration) * width;
  const outroX = (track.outroPoint / track.duration) * width;
  if (Math.abs(x - introX) < 18) wave.dragging = "intro";
  else if (Math.abs(x - outroX) < 18) wave.dragging = "outro";
  else {
    const localTime = clamp((x / width) * track.duration, 0, track.duration);
    const timeline = buildTimeline();
    const item = timeline.items.find((entry) => entry.track.localId === track.localId);
    if (item) {
      state.playbackOffset = clamp(item.start + localTime, 0, timeline.total);
      if (state.isPlaying) previewMix(state.playbackOffset);
      render();
    }
  }
}

function wavePointerMove(event) {
  const track = selectedTrack();
  if (!track?.duration || !wave.dragging) return;
  const rect = els.waveCanvas.getBoundingClientRect();
  const localTime = clamp(((event.clientX - rect.left) / rect.width) * track.duration, 0, track.duration);
  if (wave.dragging === "intro") track.introPoint = clamp(localTime, 0.5, Math.max(0.5, track.outroPoint - 0.5));
  if (wave.dragging === "outro") track.outroPoint = clamp(localTime, Math.min(track.duration - 0.5, track.introPoint + 0.5), Math.max(0.5, track.duration - 0.25));
  render();
}

function render() {
  renderMetrics();
  renderTable();
  renderTransport();
  drawWaveform();
}

function renderMetrics() {
  const playable = playableTracks();
  els.trackCount.textContent = state.tracks.length;
  els.mixDuration.textContent = formatTime(getMixDurationSeconds());
  els.crossfadeValue.textContent = `${state.settings.crossfade}s`;
  els.sortButton.disabled = playable.length < 2;
  els.playButton.disabled = playable.length < 1;
  els.restartButton.disabled = playable.length < 1;
  els.stopButton.disabled = !state.isPlaying;
  els.exportButton.disabled = playable.length < 1 || state.isExporting;
}

function renderTable() {
  if (!state.tracks.length) {
    els.trackTable.innerHTML = `<tr class="empty-row"><td colspan="9">还没有曲目。上传几首歌，让后端开始听。</td></tr>`;
    return;
  }
  els.trackTable.innerHTML = state.tracks.map((track, index) => {
    const statusClass = track.status === "ready" ? "ready" : track.status === "error" ? "error" : "";
    const statusText = track.status === "ready" ? "已就绪" : track.status === "error" ? track.error : "分析中";
    return `
      <tr class="${track.localId === state.selectedId ? "selected" : ""}" data-id="${track.localId}">
        <td>${index + 1}</td>
        <td><button class="track-title" type="button" data-action="select" data-id="${track.localId}" title="${escapeHtml(track.name)}">${escapeHtml(track.name)}</button></td>
        <td>${track.duration ? formatTime(track.duration) : "--:--"}</td>
        <td>${track.bpm || "--"}</td>
        <td>${track.key || "未知"}</td>
        <td>${Math.round((track.energy || 0) * 100)}</td>
        <td>${formatTime(track.introPoint)} / ${formatTime(track.duration - track.outroPoint)}</td>
        <td><span class="status-pill ${statusClass}">${escapeHtml(statusText)}</span></td>
        <td>
          <div class="row-actions">
            <button type="button" data-action="up" data-id="${track.localId}" ${index === 0 ? "disabled" : ""}>↑</button>
            <button type="button" data-action="down" data-id="${track.localId}" ${index === state.tracks.length - 1 ? "disabled" : ""}>↓</button>
            <button type="button" data-action="remove" data-id="${track.localId}" class="remove">×</button>
          </div>
        </td>
      </tr>
    `;
  }).join("");
  els.trackTable.querySelectorAll("button[data-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.action;
      const id = button.dataset.id;
      if (action === "select") selectTrack(id);
      if (action === "up") moveTrack(id, -1);
      if (action === "down") moveTrack(id, 1);
      if (action === "remove") removeTrack(id);
    });
  });
}

function renderTransport() {
  const total = getMixDurationSeconds();
  els.mixProgress.max = String(total);
  els.mixProgress.value = String(Math.min(state.playbackOffset, total));
  els.playTime.textContent = `${formatTime(state.playbackOffset)} / ${formatTime(total)}`;
}

function drawWaveform() {
  const canvas = els.waveCanvas;
  const ctx = wave.ctx;
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (canvas.width !== Math.floor(rect.width * ratio) || canvas.height !== Math.floor(rect.height * ratio)) {
    canvas.width = Math.floor(rect.width * ratio);
    canvas.height = Math.floor(rect.height * ratio);
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111317";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  for (let x = 0; x < width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  const track = selectedTrack();
  if (!track) {
    els.selectedTitle.textContent = "未选择曲目";
    els.handleReadout.textContent = "入点 -- / 出点 --";
    drawEmptyWave(ctx, width, height);
    return;
  }
  els.selectedTitle.textContent = track.name;
  els.handleReadout.textContent = `入点 ${formatTime(track.introPoint)} / 出点 ${formatTime(track.outroPoint)}`;

  const peaks = track.peaks || [];
  const center = height / 2;
  const barWidth = Math.max(1, width / Math.max(1, peaks.length));
  ctx.fillStyle = "#30d5a0";
  peaks.forEach((value, index) => {
    const barHeight = Math.max(2, value * (height * 0.42));
    ctx.fillRect(index * barWidth, center - barHeight, Math.max(1, barWidth * 0.72), barHeight * 2);
  });

  const introX = (track.introPoint / track.duration) * width;
  const outroX = (track.outroPoint / track.duration) * width;
  drawHandle(ctx, introX, height, "#f0cf5a", "IN");
  drawHandle(ctx, outroX, height, "#ff6b6b", "OUT");

  const timeline = buildTimeline();
  const item = timeline.items.find((entry) => entry.track.localId === track.localId);
  if (item && state.playbackOffset >= item.start && state.playbackOffset <= item.end) {
    const local = state.playbackOffset - item.start;
    const x = (local / track.duration) * width;
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.fillRect(x - 1, 0, 2, height);
  }
}

function drawEmptyWave(ctx, width, height) {
  ctx.fillStyle = "rgba(255,255,255,0.36)";
  ctx.font = "14px Bahnschrift, Segoe UI, sans-serif";
  ctx.fillText("选择曲目后可拖动 IN / OUT 过渡点，也可点击波形跳转播放位置", 24, height / 2);
}

function drawHandle(ctx, x, height, color, label) {
  ctx.fillStyle = color;
  ctx.fillRect(x - 1.5, 0, 3, height);
  ctx.fillRect(x - 17, 12, 34, 22);
  ctx.fillStyle = "#101113";
  ctx.font = "12px Bahnschrift, Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, x, 27);
  ctx.textAlign = "left";
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = response.statusText;
    try {
      const error = await response.json();
      message = error.detail || message;
    } catch {
      // Keep the status text.
    }
    throw new Error(message);
  }
  return response.json();
}

function setStatus(text) {
  els.statusText.textContent = text;
}

function formatTime(seconds) {
  const total = Math.max(0, Math.round(seconds || 0));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

window.addEventListener("resize", drawWaveform);
