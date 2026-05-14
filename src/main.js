import "./styles.css";

const API_HOST = window.location.hostname || "127.0.0.1";
const API_PROTOCOL = window.location.protocol === "https:" ? "https:" : "http:";
const API = `${API_PROTOCOL}//${API_HOST}:8001`;

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
  match: {
    fileA: null,
    fileB: null,
    result: null,
    loading: false,
    error: "",
  },
  settings: {
    sortMode: "recommended",
    crossfade: 8,
    autoTransition: true,
    beatSync: false,
    aiPrecision: true,
    phraseBars: 8,
    loudnessNormalize: true,
    targetLufs: -16,
    equalPowerFade: true,
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
        <label class="toggle"><input id="aiPrecision" type="checkbox" checked /><span>AI 精准小节混音</span></label>
        <label class="toggle"><input id="loudnessNormalize" type="checkbox" checked /><span>响度归一化</span></label>

        <label class="field">
          <span>重叠小节 <b id="phraseBarsValue">8 bars</b></span>
          <input id="phraseBars" type="range" min="4" max="16" value="8" step="4" />
        </label>

        <label class="field">
          <span>目标响度 <b id="targetLufsValue">-16 LUFS</b></span>
          <input id="targetLufs" type="range" min="-20" max="-10" value="-16" step="1" />
        </label>

        <label class="field">
          <span>过渡滤波</span>
          <select id="filterMode">
            <option value="dynamicEq">AI 动态 EQ 避让</option>
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

        <section class="match-panel" aria-label="两首歌匹配评分">
          <div class="match-head">
            <div>
              <span class="tiny-label">Pair Match</span>
              <strong>两首歌衔接匹配评分</strong>
            </div>
            <button id="matchButton" type="button">计算匹配度</button>
          </div>
          <div class="match-inputs">
            <label>
              <span>Song A</span>
              <input id="matchFileA" type="file" accept="audio/*" />
            </label>
            <label>
              <span>Song B</span>
              <input id="matchFileB" type="file" accept="audio/*" />
            </label>
          </div>
          <div id="matchResult" class="match-result">上传任意两首歌，系统会用 Camelot、BPM、能量和结构可过渡性计算匹配分。</div>
        </section>

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
  aiPrecision: document.querySelector("#aiPrecision"),
  loudnessNormalize: document.querySelector("#loudnessNormalize"),
  phraseBars: document.querySelector("#phraseBars"),
  phraseBarsValue: document.querySelector("#phraseBarsValue"),
  targetLufs: document.querySelector("#targetLufs"),
  targetLufsValue: document.querySelector("#targetLufsValue"),
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
  matchFileA: document.querySelector("#matchFileA"),
  matchFileB: document.querySelector("#matchFileB"),
  matchButton: document.querySelector("#matchButton"),
  matchResult: document.querySelector("#matchResult"),
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
  els.matchFileA.addEventListener("change", () => {
    state.match.fileA = els.matchFileA.files?.[0] || null;
    renderMatchResult();
  });
  els.matchFileB.addEventListener("change", () => {
    state.match.fileB = els.matchFileB.files?.[0] || null;
    renderMatchResult();
  });
  els.matchButton.addEventListener("click", calculatePairMatch);

  els.sortMode.addEventListener("change", syncSettings);
  els.crossfade.addEventListener("input", syncSettings);
  els.autoTransition.addEventListener("change", syncSettings);
  els.beatSync.addEventListener("change", syncSettings);
  els.aiPrecision.addEventListener("change", syncSettings);
  els.loudnessNormalize.addEventListener("change", syncSettings);
  els.phraseBars.addEventListener("input", syncSettings);
  els.targetLufs.addEventListener("input", syncSettings);
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
  state.settings.aiPrecision = els.aiPrecision.checked;
  state.settings.loudnessNormalize = els.loudnessNormalize.checked;
  state.settings.phraseBars = Number(els.phraseBars.value);
  state.settings.targetLufs = Number(els.targetLufs.value);
  state.settings.equalPowerFade = els.aiPrecision.checked;
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
      camelot: null,
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

async function calculatePairMatch() {
  if (!state.match.fileA || !state.match.fileB) {
    state.match.error = "请先选择两首音频。";
    renderMatchResult();
    return;
  }
  state.match.loading = true;
  state.match.error = "";
  state.match.result = null;
  renderMatchResult();
  try {
    await assertBackendReachable();
    const form = new FormData();
    form.append("file_a", state.match.fileA);
    form.append("file_b", state.match.fileB);
    state.match.result = await fetchJson(`${API}/api/match`, { method: "POST", body: form });
    setStatus("两歌匹配评分完成");
  } catch (error) {
    state.match.error = error.message || "匹配评分失败";
  } finally {
    state.match.loading = false;
    renderMatchResult();
  }
}

function renderMatchResult() {
  if (!els.matchResult) return;
  els.matchButton.disabled = state.match.loading || !state.match.fileA || !state.match.fileB;
  if (state.match.loading) {
    els.matchResult.innerHTML = `<div class="match-loading">正在分析两首歌并计算匹配度...</div>`;
    return;
  }
  if (state.match.error) {
    els.matchResult.innerHTML = `<div class="match-error">${escapeHtml(state.match.error)}</div>`;
    return;
  }
  if (!state.match.result) {
    const a = state.match.fileA?.name || "未选择";
    const b = state.match.fileB?.name || "未选择";
    els.matchResult.innerHTML = `A: ${escapeHtml(a)}<br />B: ${escapeHtml(b)}<br />上传任意两首歌，系统会用 Camelot、BPM、能量和结构可过渡性计算匹配分。`;
    return;
  }

  const result = state.match.result;
  const forward = result.directions.a_to_b;
  const reverse = result.directions.b_to_a;
  els.matchResult.innerHTML = `
    <div class="match-score">
      <span>${escapeHtml(result.overall_level)}</span>
      <strong>${Math.round(result.overall_score)}</strong>
      <em>推荐方向：${escapeHtml(result.recommended_direction)}</em>
    </div>
    ${renderDirectionMatch("A → B", forward)}
    ${renderDirectionMatch("B → A", reverse)}
  `;
}

function renderDirectionMatch(title, direction) {
  const c = direction.components;
  return `
    <div class="direction-card">
      <div class="direction-head">
        <strong>${title}</strong>
        <span>${Math.round(direction.total_score)} / ${escapeHtml(direction.level)}</span>
      </div>
      <div class="component-grid">
        ${renderComponent("Camelot", c.camelot.score, `${c.camelot.from || "--"} → ${c.camelot.to || "--"} · ${c.camelot.rank}`)}
        ${renderComponent("BPM", c.bpm.score, `Δ ${c.bpm.delta ?? "--"} BPM`)}
        ${renderComponent("Energy", c.energy.score, `Δ ${c.energy.delta ?? "--"}`)}
        ${renderComponent("Structure", c.structure.score, `${c.structure.phrase_bars || 0} bars / ${c.structure.overlap_seconds}s`)}
      </div>
    </div>
  `;
}

function renderComponent(label, score, note) {
  return `
    <div class="component">
      <span>${escapeHtml(label)}</span>
      <strong>${Math.round(score)}</strong>
      <small>${escapeHtml(note)}</small>
    </div>
  `;
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
    await assertBackendReachable();
    const form = new FormData();
    form.append("file", track.file);
    const result = await fetchJson(`${API}/api/tracks`, { method: "POST", body: form });
    track.id = result.id;
    track.status = "ready";
    track.duration = result.duration || track.duration;
    track.bpm = result.bpm;
    track.beats = result.beats || [];
    track.bars = result.bars || [];
    track.phrases = result.phrases || [];
    track.key = result.key || "未知";
    track.camelot = result.camelot || keyLabelToCamelot(track.key, result.mode);
    track.key_index = result.key_index;
    track.mode = result.mode;
    track.energy = result.energy || 0;
    track.intro_low = result.intro_low || 0;
    track.outro_low = result.outro_low || 0;
    track.loudness_lufs = result.loudness_lufs;
    track.true_peak_db = result.true_peak_db;
    track.transition_candidates = result.transition_candidates || null;
    track.peaks = result.peaks?.length ? result.peaks : track.peaks;
    track.introPoint = clamp(track.transition_candidates?.intro ?? track.intro_low ?? track.introPoint, 0.5, Math.max(0.5, track.duration * 0.35));
    track.outroPoint = clamp(track.transition_candidates?.outro ?? (track.duration - (track.outro_low || state.settings.crossfade)), track.duration * 0.55, Math.max(track.duration - 0.5, 0.5));
    setStatus(`完成分析 ${track.name}`);
  } catch (error) {
    applyLocalFallbackAnalysis(track, error);
  } finally {
    render();
  }
}

async function assertBackendReachable() {
  try {
    await fetchJson(`${API}/api/health`);
  } catch (error) {
    throw new Error(`后端连接失败，请确认 ${API}/api/health 可访问。${error.message || ""}`.trim());
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

function applyLocalFallbackAnalysis(track, error) {
  if (!track.buffer) {
    track.status = "error";
    track.error = error.message || "后端分析失败";
    setStatus(`分析失败：${track.name}`);
    return;
  }

  const mono = makeMono(track.buffer);
  const envelope = buildEnvelope(mono, track.buffer.sampleRate);
  const key = estimateLocalKey(mono, track.buffer.sampleRate);
  track.duration = track.buffer.duration;
  track.bpm = estimateLocalBpm(envelope.values, envelope.frameRate);
  track.beats = [];
  track.bars = [];
  track.phrases = [];
  track.key = key.label;
  track.camelot = keyLabelToCamelot(key.label, key.mode);
  track.key_index = key.index;
  track.mode = key.mode;
  track.energy = envelope.energy;
  track.intro_low = envelope.introLow;
  track.outro_low = envelope.outroLow;
  track.loudness_lufs = localLoudness(mono);
  track.true_peak_db = localPeakDb(mono);
  track.transition_candidates = {
    intro: track.introPoint,
    outro: track.outroPoint,
    confidence: 0.25,
  };
  track.introPoint = clamp(track.intro_low || track.introPoint, 0.5, Math.max(0.5, track.duration * 0.35));
  track.outroPoint = clamp(track.duration - (track.outro_low || state.settings.crossfade), track.duration * 0.55, Math.max(track.duration - 0.5, 0.5));
  track.status = "ready";
  track.error = "";
  setStatus(`已用浏览器本地分析：${track.name}`);
  console.warn("SmartMix backend analysis failed; local fallback was used.", error);
}

function makeMono(buffer) {
  const length = buffer.length;
  const mono = new Float32Array(length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < length; i += 1) mono[i] += data[i] / buffer.numberOfChannels;
  }
  return mono;
}

function buildEnvelope(samples, sampleRate) {
  const windowSize = Math.max(1024, Math.floor(sampleRate * 0.05));
  const values = [];
  let sumEnergy = 0;
  let peak = 0;
  for (let offset = 0; offset < samples.length; offset += windowSize) {
    let sum = 0;
    const end = Math.min(offset + windowSize, samples.length);
    for (let i = offset; i < end; i += 1) sum += samples[i] * samples[i];
    const rms = Math.sqrt(sum / Math.max(1, end - offset));
    values.push(rms);
    sumEnergy += rms;
    peak = Math.max(peak, rms);
  }
  const average = sumEnergy / Math.max(1, values.length);
  const threshold = Math.max(average * 0.55, peak * 0.08);
  const frameRate = sampleRate / windowSize;
  return {
    values,
    frameRate,
    energy: Math.min(1, average * 7 + peak * 1.5),
    introLow: countLowFrames(values, threshold, true) / frameRate,
    outroLow: countLowFrames(values, threshold, false) / frameRate,
  };
}

function countLowFrames(values, threshold, fromStart) {
  let count = 0;
  const limit = Math.min(values.length, Math.floor(values.length * 0.25));
  for (let i = 0; i < limit; i += 1) {
    const index = fromStart ? i : values.length - 1 - i;
    if (values[index] > threshold) break;
    count += 1;
  }
  return count;
}

function estimateLocalBpm(envelope, frameRate) {
  if (envelope.length < frameRate * 8) return null;
  const flux = envelope.map((value, index) => Math.max(0, value - (envelope[index - 1] || 0)));
  const mean = flux.reduce((sum, value) => sum + value, 0) / flux.length;
  const centered = flux.map((value) => Math.max(0, value - mean));
  let bestBpm = 120;
  let bestScore = -Infinity;
  for (let bpm = 70; bpm <= 180; bpm += 1) {
    const lag = Math.round((60 / bpm) * frameRate);
    if (lag < 1 || lag >= centered.length) continue;
    let score = 0;
    for (let i = lag; i < centered.length; i += 1) score += centered[i] * centered[i - lag];
    if (score > bestScore) {
      bestScore = score;
      bestBpm = bpm;
    }
  }
  return bestScore > 0 ? bestBpm : null;
}

function estimateLocalKey(samples, sampleRate) {
  const chroma = new Array(12).fill(0);
  const chunkSize = 4096;
  const step = Math.max(chunkSize, Math.floor(sampleRate * 1.2));
  let used = 0;
  for (let offset = 0; offset + chunkSize < samples.length && used < 80; offset += step) {
    const pitch = estimatePitch(samples.subarray(offset, offset + chunkSize), sampleRate);
    if (!pitch) continue;
    const midi = Math.round(69 + 12 * Math.log2(pitch / 440));
    chroma[((midi % 12) + 12) % 12] += 1;
    used += 1;
  }
  if (!used) return { label: "Unknown", index: null, mode: null };
  normalize(chroma);
  const majorProfile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const minorProfile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
  const keyNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  let best = { score: -Infinity, index: 0, mode: "major" };
  for (let root = 0; root < 12; root += 1) {
    const majorScore = correlateProfile(chroma, majorProfile, root);
    const minorScore = correlateProfile(chroma, minorProfile, root);
    if (majorScore > best.score) best = { score: majorScore, index: root, mode: "major" };
    if (minorScore > best.score) best = { score: minorScore, index: root, mode: "minor" };
  }
  return {
    label: `${keyNames[best.index]} ${best.mode === "major" ? "Maj" : "Min"}`,
    index: best.index,
    mode: best.mode,
  };
}

function localLoudness(samples) {
  const rms = Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / Math.max(1, samples.length) + 1e-12);
  return Math.round(20 * Math.log10(rms) * 100) / 100;
}

function localPeakDb(samples) {
  let peak = 1e-12;
  for (let i = 0; i < samples.length; i += 1) peak = Math.max(peak, Math.abs(samples[i]));
  return Math.round(20 * Math.log10(peak) * 100) / 100;
}

function estimatePitch(chunk, sampleRate) {
  let rms = 0;
  for (let i = 0; i < chunk.length; i += 1) rms += chunk[i] * chunk[i];
  rms = Math.sqrt(rms / chunk.length);
  if (rms < 0.015) return null;
  const minLag = Math.floor(sampleRate / 900);
  const maxLag = Math.floor(sampleRate / 80);
  let bestLag = -1;
  let bestCorrelation = 0;
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let correlation = 0;
    for (let i = 0; i < chunk.length - lag; i += 1) correlation += chunk[i] * chunk[i + lag];
    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestLag = lag;
    }
  }
  return bestLag > 0 ? sampleRate / bestLag : null;
}

function normalize(values) {
  const sum = values.reduce((total, value) => total + value, 0) || 1;
  for (let i = 0; i < values.length; i += 1) values[i] /= sum;
}

function correlateProfile(chroma, profile, root) {
  let score = 0;
  for (let i = 0; i < 12; i += 1) score += chroma[(i + root) % 12] * profile[i];
  return score;
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
  const codeA = a.camelot || keyLabelToCamelot(a.key, a.mode);
  const codeB = b.camelot || keyLabelToCamelot(b.key, b.mode);
  if (!codeA || !codeB) return 0.5;
  return Math.min(1, camelotDistance(codeA, codeB) / 6);
}

const KEY_TO_CAMELOT = {
  C: "8B", "C#": "9B", Db: "9B", D: "10B", "D#": "11B", Eb: "11B",
  E: "12B", F: "1B", "F#": "2B", Gb: "2B", G: "3B", "G#": "4B",
  Ab: "4B", A: "5B", "A#": "6B", Bb: "6B", B: "7B",
  Am: "8A", "A#m": "9A", Bbm: "9A", Bm: "10A", Cm: "11A",
  "C#m": "12A", Dbm: "12A", Dm: "1A", "D#m": "2A", Ebm: "2A",
  Em: "3A", Fm: "4A", "F#m": "5A", Gbm: "5A", Gm: "6A",
  "G#m": "7A", Abm: "7A",
};

function keyLabelToCamelot(label, mode) {
  if (!label || label === "Unknown" || label === "未知") return null;
  const parts = String(label).replace("Major", "Maj").replace("Minor", "Min").split(/\s+/);
  const root = parts[0];
  const inferred = String(mode || parts[1] || "").toLowerCase();
  const key = inferred.startsWith("min") || inferred === "minor" ? `${root}m` : root;
  return KEY_TO_CAMELOT[key] || null;
}

function parseCamelot(code) {
  return { num: Number(code.slice(0, -1)), mode: code.slice(-1) };
}

function camelotNumDistance(a, b) {
  const diff = Math.abs(a - b);
  return Math.min(diff, 12 - diff);
}

function camelotDistance(codeA, codeB) {
  const a = parseCamelot(codeA);
  const b = parseCamelot(codeB);
  const dNum = camelotNumDistance(a.num, b.num);
  let score;
  if (a.num === b.num) score = 0;
  else if (a.mode === b.mode) score = dNum;
  else score = dNum + 2;
  if (dNum === 5) score -= 1;
  if (dNum >= 6) score += 1;
  return Math.max(0, score);
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
    camelot: track.camelot,
    key_index: track.key_index,
    mode: track.mode,
    beats: track.beats || [],
    bars: track.bars || [],
    phrases: track.phrases || [],
    energy: track.energy,
    intro_low: track.intro_low,
    outro_low: track.outro_low,
    loudness_lufs: track.loudness_lufs,
    true_peak_db: track.true_peak_db,
    transition_candidates: track.transition_candidates,
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
  els.aiPrecision.checked = state.settings.aiPrecision;
  els.loudnessNormalize.checked = state.settings.loudnessNormalize;
  els.phraseBars.value = state.settings.phraseBars;
  els.targetLufs.value = state.settings.targetLufs;
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
  if (state.settings.aiPrecision) {
    const phrase = phraseTransitionSeconds(prev, next);
    if (phrase) actual = phrase;
    const prevOut = prev.transition_candidates?.outro;
    const nextIn = next.transition_candidates?.intro;
    if (Number.isFinite(prevOut) && Number.isFinite(nextIn)) {
      actual = Math.min(actual, Math.max(0.5, prev.duration - prevOut), Math.max(0.5, nextIn));
    }
  }
  if (Number.isFinite(prev.outroPoint) && Number.isFinite(next.introPoint)) {
    actual = Math.min(actual, Math.max(0.5, prev.duration - prev.outroPoint), Math.max(0.5, next.introPoint));
  }
  if (state.settings.autoTransition) {
    actual = Math.max(2, Math.min(actual, (prev.outro_low || 0) + (next.intro_low || 0) + 2));
  }
  return Math.min(actual, maxByLength);
}

function phraseTransitionSeconds(prev, next) {
  const bpms = [prev.bpm, next.bpm].filter((bpm) => Number.isFinite(bpm) && bpm > 0);
  if (!bpms.length) return null;
  const avg = bpms.reduce((sum, bpm) => sum + bpm, 0) / bpms.length;
  return Math.max(2, state.settings.phraseBars * 4 * (60 / avg));
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
  renderMatchResult();
  drawWaveform();
}

function renderMetrics() {
  const playable = playableTracks();
  els.trackCount.textContent = state.tracks.length;
  els.mixDuration.textContent = formatTime(getMixDurationSeconds());
  els.crossfadeValue.textContent = `${state.settings.crossfade}s`;
  els.phraseBarsValue.textContent = `${state.settings.phraseBars} bars`;
  els.targetLufsValue.textContent = `${state.settings.targetLufs} LUFS`;
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
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new Error(`无法连接后端 ${API}。请确认已运行 pnpm backend，或重新运行 pnpm dev。原始错误：${error.message}`);
  }
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
