import "./styles.css";
import { API_BASE_URL as API, apiUrl, fetchJson } from "./api/client";
import { explainTransition } from "./explain/explainTransition";
import { normalizeStyle, scoreStyleCompatibility, styleDistance, styleFamily, styleLabel } from "./analysis/style";
import { recommendNextTracks, recommendTransition } from "./transitions/recommend";

const state = {
  tracks: [],
  originalOrder: [],
  selectedId: null,
  audioContext: null,
  activeSources: [],
  activeNodes: [],
  liveRefreshTimer: null,
  playStartContextTime: 0,
  playStartOffset: 0,
  playbackOffset: 0,
  timer: null,
  isPlaying: false,
  isExporting: false,
  view: "studio",
  projects: [],
  stemDebugger: {
    trackId: null,
    isPlaying: false,
    activeSources: [],
    activeNodes: [],
    playStartContextTime: 0,
    playStartOffset: 0,
    playbackOffset: 0,
    timer: null,
    controls: {},
    isSeparating: false,
    scanFrame: null,
    referenceTrackId: null,
    isAutoMixing: false,
  },
  teaching: {
    open: false,
    targetEnergy: "keep",
    beginnerMode: true,
    maxComplexity: 3,
    previews: {},
    loadingPreviewId: null,
  },
  autoHandoff: {
    loading: false,
    rendering: false,
    renderedCount: 0,
    plan: null,
    error: "",
  },
  match: {
    fileA: null,
    fileB: null,
    result: null,
    repairResult: null,
    loading: false,
    repairLoading: false,
    error: "",
  },
  mashup: {
    trackAId: null,
    trackBId: null,
    mode: "groove_vocal_handoff",
    barsPerSegment: 16,
    useStems: true,
    vocalPriority: "auto",
    bedPreference: "auto",
    allowHybridBed: true,
    allowVocalPitchShift: false,
    maxVocalStretch: 1.06,
    analysis: null,
    plan: null,
    renderResult: null,
    analyzing: false,
    planning: false,
    rendering: false,
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
    mixStrategy: "auto",
    filterMode: "dynamicEq",
    exportFormat: "mp3",
    eq: { low: 0, mid: 0, high: 0 },
    mixStyleTransfer: {
      enabled: false,
      source: "reference-guided-auto-mix",
      trackId: null,
      referenceTrackId: null,
      referenceTrackName: "",
      params: null,
      result: null,
    },
  },
};

const DEFAULT_TRACK_MIXER = Object.freeze({
  gain: 1,
  eq: { low: 0, mid: 0, high: 0 },
});

const STEMS = Object.freeze([
  { id: "vocals", label: "\u4eba\u58f0", color: "#65d7f2", filter: "vocal" },
  { id: "drums", label: "\u9f13", color: "#65d7f2", filter: "drums" },
  { id: "bass", label: "\u8d1d\u65af", color: "#72e5ed", filter: "bass" },
  { id: "other", label: "\u5176\u4ed6", color: "#87eef0", filter: "other" },
]);

let stemSeparationChain = Promise.resolve();

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
        <button id="stemDebuggerToggle" class="ghost" type="button">\u5206\u8f68\u8c03\u8bd5</button>
        <button id="refreshProjects" class="ghost" type="button">刷新项目</button>
        <button id="teachingToggle" class="ghost" type="button">教学入口</button>
      </div>
    </header>

    <section id="studioView" class="studio-grid">
      <aside class="control-panel" aria-label="混音控制">
        <div class="meter-block"><span>Tracks</span><strong id="trackCount">0</strong></div>
        <div class="meter-block"><span>Mix Time</span><strong id="mixDuration">00:00</strong></div>
        <div class="meter-block wide"><span>Status</span><strong id="statusText">等待后端</strong></div>

        <label class="field">
          <span>排序策略</span>
          <select id="sortMode">
            <option value="recommended">综合推荐</option>
            <option value="harmonic">谐和优先</option>
            <option value="styleFlow">风格连续</option>
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
          <span>AI 混音策略</span>
          <select id="mixStrategy">
            <option value="auto">AI 自动判断</option>
            <option value="vocalHandoff">人声接鼓点</option>
            <option value="vocalSafe">保留人声清晰</option>
            <option value="bassSwap">低频交换切入</option>
            <option value="smooth">平滑氛围过渡</option>
            <option value="quickCut">快速切歌</option>
          </select>
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
          <button id="autoHandoffButton" type="button" class="secondary">Smart Beat Handoff</button>
          <button id="playButton" type="button">预览</button>
          <button id="stopButton" type="button" class="secondary">停止</button>
        </div>

        <div id="autoHandoffPanel" class="auto-handoff-panel"></div>

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
          <div><span class="pulse-dot"></span><p>拖入音频文件，后端会立刻分析 BPM、调性、风格和能量</p></div>
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
          <div id="matchResult" class="match-result">上传任意两首歌，系统会用 Camelot、BPM、风格、能量和结构可过渡性计算匹配分。</div>
        </section>

        <section id="teachingPanel" class="teaching-panel" aria-label="DJ 接歌教学" hidden>
          <div class="teaching-head">
            <div>
              <span class="tiny-label">DJ Lesson</span>
              <strong>下一首怎么接</strong>
            </div>
            <div class="teaching-controls">
              <label>
                <span>目标能量</span>
                <select id="teachingEnergy">
                  <option value="keep">保持</option>
                  <option value="up">上扬</option>
                  <option value="down">回落</option>
                </select>
              </label>
              <label class="mini-toggle"><input id="teachingBeginner" type="checkbox" checked /><span>新手模式</span></label>
              <label>
                <span>难度</span>
                <select id="teachingComplexity">
                  <option value="2">1-2</option>
                  <option value="3" selected>1-3</option>
                  <option value="4">1-4</option>
                  <option value="5">全部</option>
                </select>
              </label>
            </div>
          </div>
          <div id="teachingContent" class="teaching-content"></div>
        </section>

        <section class="mashup-panel" aria-label="双曲重组 / Mashup Builder">
          <div class="mashup-head">
            <div>
              <span class="tiny-label">Mashup Builder</span>
              <strong>双曲重组</strong>
            </div>
            <div class="mashup-actions">
              <button id="mashupAnalyzeButton" type="button">分析段落</button>
              <button id="mashupPlanButton" type="button" class="secondary">生成拼接方案</button>
              <button id="mashupRenderButton" type="button" class="export">渲染试听/导出</button>
            </div>
          </div>
          <div id="mashupFlow" class="mashup-flow"></div>
          <div class="mashup-controls">
            <label>
              <span>Song A</span>
              <select id="mashupTrackA"></select>
            </label>
            <label>
              <span>Song B</span>
              <select id="mashupTrackB"></select>
            </label>
            <label>
              <span>Mode</span>
              <select id="mashupMode">
                <option value="groove_vocal_handoff">Groove 人声接力</option>
                <option value="a_vocal_on_b_groove">A 人声 + B Groove</option>
                <option value="b_vocal_on_a_groove">B 人声 + A Groove</option>
                <option value="call_response_groove">Call/Response Groove</option>
                <option value="hook_exchange_groove">Hook 交换 Groove</option>
                <option value="auto">auto (Groove first)</option>
                <option value="smooth_join">smooth_join</option>
                <option value="hook_swap">hook_swap</option>
                <option value="a_vocal_b_instrumental">A vocals + B instrumental</option>
                <option value="b_vocal_a_instrumental">B vocals + A instrumental</option>
                <option value="energy_build">energy_build</option>
              </select>
            </label>
            <label>
              <span>Segment</span>
              <select id="mashupBars">
                <option value="8">8 bars</option>
                <option value="16" selected>16 bars</option>
              </select>
            </label>
            <label class="mashup-stems"><input id="mashupUseStems" type="checkbox" checked /><span>useStems</span></label>
            <label>
              <span>Vocal priority</span>
              <select id="mashupVocalPriority">
                <option value="auto" selected>auto</option>
                <option value="prefer_a">prefer A vocal</option>
                <option value="prefer_b">prefer B vocal</option>
              </select>
            </label>
            <label>
              <span>Groove bed</span>
              <select id="mashupBedPreference">
                <option value="auto" selected>auto</option>
                <option value="A">prefer A bed</option>
                <option value="B">prefer B bed</option>
              </select>
            </label>
            <label class="mashup-stems"><input id="mashupAllowHybridBed" type="checkbox" checked /><span>allow hybrid bed</span></label>
            <label class="mashup-stems"><input id="mashupAllowVocalPitchShift" type="checkbox" /><span>allow vocal pitch</span></label>
            <label>
              <span>Max vocal stretch</span>
              <select id="mashupMaxVocalStretch">
                <option value="1.03">1.03 conservative</option>
                <option value="1.06" selected>1.06 balanced</option>
                <option value="1.10">1.10 creative</option>
              </select>
            </label>
          </div>
          <div id="mashupSegments" class="mashup-segments"></div>
          <div id="mashupTimeline" class="mashup-timeline"></div>
          <div id="mashupResult" class="mashup-result"></div>
        </section>

        <section class="transport">
          <button id="restartButton" type="button" class="iconish">从头播放</button>
          <input id="mixProgress" type="range" min="0" max="0" value="0" step="0.01" />
          <time id="playTime">00:00 / 00:00</time>
        </section>

        <section class="mix-map" aria-label="混音时间线">
          <div class="mix-map-head">
            <div>
              <span class="tiny-label">Mix Timeline</span>
              <strong>重叠过渡时间线</strong>
            </div>
            <span id="transitionReadout">上传两首以上歌曲后显示过渡</span>
          </div>
          <div id="mixTimeline" class="mix-timeline"></div>
        </section>

        <section class="deck-panel" aria-label="双 Deck 混音控制">
          <div class="deck-head">
            <div>
              <span class="tiny-label">Deck Mixer</span>
              <strong>当前过渡双 Deck</strong>
            </div>
            <button id="jumpToTransition" type="button" class="secondary">跳到过渡</button>
          </div>
          <div id="deckMixer" class="deck-grid"></div>
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
          <div id="cueEditor" class="cue-editor"></div>
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
                <th>风格</th>
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

    <section id="stemDebuggerView" class="stem-debugger-view" hidden>
      <header class="stem-topbar">
        <div class="stem-titlebar">
          <button id="stemBackButton" class="stem-back" type="button" aria-label="\u8fd4\u56de\u4e3b\u5de5\u4f5c\u53f0">←</button>
          <div>
            <span class="tiny-label">Stem Debugger</span>
            <strong>\u5206\u8f68\u8c03\u8bd5</strong>
          </div>
        </div>
        <div class="stem-tools">
          <label class="stem-picker">
            <span>\u97f3\u9891</span>
            <select id="stemTrackSelect"></select>
          </label>
          <label class="stem-picker stem-reference-picker">
            <span>\u53c2\u8003\u66f2</span>
            <select id="stemReferenceSelect"></select>
          </label>
          <button id="stemAutoMixButton" type="button" class="secondary">\u53c2\u8003\u66f2\u81ea\u52a8\u6df7\u97f3</button>
          <button id="stemRestartButton" type="button" class="secondary">\u4ece\u5934</button>
          <button id="stemSeparateButton" type="button" class="secondary">Demucs \u5206\u8f68</button>
          <button id="stemPlayButton" type="button">\u64ad\u653e</button>
          <button id="stemStopButton" type="button" class="secondary">\u505c\u6b62</button>
        </div>
      </header>

      <div class="stem-ruler" aria-hidden="true"></div>
      <div id="stemMixResult" class="stem-mix-result" hidden></div>
      <div id="stemDeck" class="stem-deck"></div>

      <footer class="stem-transport">
        <button id="stemTransportPlay" type="button" aria-label="\u64ad\u653e\u6216\u6682\u505c">▶</button>
        <input id="stemProgress" type="range" min="0" max="0" value="0" step="0.01" />
        <time id="stemTime">00:00 / 00:00</time>
      </footer>
    </section>
  </main>
`;

const els = {
  studioView: document.querySelector("#studioView"),
  stemDebuggerView: document.querySelector("#stemDebuggerView"),
  fileInput: document.querySelector("#fileInput"),
  stemDebuggerToggle: document.querySelector("#stemDebuggerToggle"),
  stemBackButton: document.querySelector("#stemBackButton"),
  stemTrackSelect: document.querySelector("#stemTrackSelect"),
  stemReferenceSelect: document.querySelector("#stemReferenceSelect"),
  stemAutoMixButton: document.querySelector("#stemAutoMixButton"),
  stemMixResult: document.querySelector("#stemMixResult"),
  stemRestartButton: document.querySelector("#stemRestartButton"),
  stemSeparateButton: document.querySelector("#stemSeparateButton"),
  stemPlayButton: document.querySelector("#stemPlayButton"),
  stemStopButton: document.querySelector("#stemStopButton"),
  stemTransportPlay: document.querySelector("#stemTransportPlay"),
  stemProgress: document.querySelector("#stemProgress"),
  stemTime: document.querySelector("#stemTime"),
  stemDeck: document.querySelector("#stemDeck"),
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
  mixStrategy: document.querySelector("#mixStrategy"),
  filterMode: document.querySelector("#filterMode"),
  exportFormat: document.querySelector("#exportFormat"),
  eqLow: document.querySelector("#eqLow"),
  eqMid: document.querySelector("#eqMid"),
  eqHigh: document.querySelector("#eqHigh"),
  sortButton: document.querySelector("#sortButton"),
  autoHandoffButton: document.querySelector("#autoHandoffButton"),
  autoHandoffPanel: document.querySelector("#autoHandoffPanel"),
  playButton: document.querySelector("#playButton"),
  stopButton: document.querySelector("#stopButton"),
  restartButton: document.querySelector("#restartButton"),
  exportButton: document.querySelector("#exportButton"),
  downloadLink: document.querySelector("#downloadLink"),
  mixProgress: document.querySelector("#mixProgress"),
  playTime: document.querySelector("#playTime"),
  mixTimeline: document.querySelector("#mixTimeline"),
  transitionReadout: document.querySelector("#transitionReadout"),
  deckMixer: document.querySelector("#deckMixer"),
  jumpToTransition: document.querySelector("#jumpToTransition"),
  waveCanvas: document.querySelector("#waveCanvas"),
  cueEditor: document.querySelector("#cueEditor"),
  selectedTitle: document.querySelector("#selectedTitle"),
  handleReadout: document.querySelector("#handleReadout"),
  matchFileA: document.querySelector("#matchFileA"),
  matchFileB: document.querySelector("#matchFileB"),
  matchButton: document.querySelector("#matchButton"),
  repairMatchButton: null,
  matchResult: document.querySelector("#matchResult"),
  projectName: document.querySelector("#projectName"),
  saveProject: document.querySelector("#saveProject"),
  projectList: document.querySelector("#projectList"),
  loadProject: document.querySelector("#loadProject"),
  refreshProjects: document.querySelector("#refreshProjects"),
  teachingToggle: document.querySelector("#teachingToggle"),
  teachingPanel: document.querySelector("#teachingPanel"),
  teachingContent: document.querySelector("#teachingContent"),
  teachingEnergy: document.querySelector("#teachingEnergy"),
  teachingBeginner: document.querySelector("#teachingBeginner"),
  teachingComplexity: document.querySelector("#teachingComplexity"),
  mashupTrackA: document.querySelector("#mashupTrackA"),
  mashupTrackB: document.querySelector("#mashupTrackB"),
  mashupMode: document.querySelector("#mashupMode"),
  mashupBars: document.querySelector("#mashupBars"),
  mashupUseStems: document.querySelector("#mashupUseStems"),
  mashupVocalPriority: document.querySelector("#mashupVocalPriority"),
  mashupBedPreference: document.querySelector("#mashupBedPreference"),
  mashupAllowHybridBed: document.querySelector("#mashupAllowHybridBed"),
  mashupAllowVocalPitchShift: document.querySelector("#mashupAllowVocalPitchShift"),
  mashupMaxVocalStretch: document.querySelector("#mashupMaxVocalStretch"),
  mashupAnalyzeButton: document.querySelector("#mashupAnalyzeButton"),
  mashupPlanButton: document.querySelector("#mashupPlanButton"),
  mashupRenderButton: document.querySelector("#mashupRenderButton"),
  mashupFlow: document.querySelector("#mashupFlow"),
  mashupSegments: document.querySelector("#mashupSegments"),
  mashupTimeline: document.querySelector("#mashupTimeline"),
  mashupResult: document.querySelector("#mashupResult"),
};

const wave = {
  ctx: els.waveCanvas.getContext("2d"),
  dragging: null,
};

bindEvents();
pingBackend();
loadProjectList();
render();

function ensureRepairMatchButton() {
  if (els.repairMatchButton || !els.matchButton) return;
  const button = document.createElement("button");
  button.id = "repairMatchButton";
  button.type = "button";
  button.className = "secondary";
  button.textContent = "自动修复匹配";
  els.matchButton.insertAdjacentElement("afterend", button);
  els.repairMatchButton = button;
}

function bindEvents() {
  ensureRepairMatchButton();
  els.fileInput.addEventListener("change", (event) => addFiles([...event.target.files]));
  els.stemDebuggerToggle.addEventListener("click", openStemDebugger);
  els.stemBackButton.addEventListener("click", closeStemDebugger);
  els.stemTrackSelect.addEventListener("change", selectStemDebugTrack);
  els.stemReferenceSelect.addEventListener("change", selectStemReferenceTrack);
  els.stemAutoMixButton.addEventListener("click", runStemReferenceMix);
  els.stemRestartButton.addEventListener("click", () => playStemDebugger(0));
  els.stemSeparateButton.addEventListener("click", separateActiveStemTrack);
  els.stemPlayButton.addEventListener("click", toggleStemDebuggerPlayback);
  els.stemStopButton.addEventListener("click", stopStemDebuggerAndReset);
  els.stemTransportPlay.addEventListener("click", toggleStemDebuggerPlayback);
  els.stemDeck.addEventListener("input", updateStemControl);
  els.stemDeck.addEventListener("click", stemDeckClick);
  els.stemProgress.addEventListener("input", () => {
    state.stemDebugger.playbackOffset = Number(els.stemProgress.value);
    renderStemTransport();
    drawStemWaveforms();
  });
  els.stemProgress.addEventListener("change", () => {
    if (state.stemDebugger.isPlaying) playStemDebugger(Number(els.stemProgress.value));
  });
  els.sortButton.addEventListener("click", applySort);
  els.autoHandoffButton.addEventListener("click", generateAutoHandoffPlan);
  els.autoHandoffPanel.addEventListener("click", autoHandoffPanelClick);
  els.playButton.addEventListener("click", () => previewMix(state.playbackOffset));
  els.restartButton.addEventListener("click", () => previewMix(0));
  els.stopButton.addEventListener("click", stopPreview);
  els.jumpToTransition.addEventListener("click", jumpToActiveTransition);
  els.exportButton.addEventListener("click", exportMix);
  els.saveProject.addEventListener("click", saveProject);
  els.loadProject.addEventListener("click", loadSelectedProject);
  els.refreshProjects.addEventListener("click", loadProjectList);
  els.teachingToggle.addEventListener("click", toggleTeachingPanel);
  els.teachingEnergy.addEventListener("change", syncTeachingSettings);
  els.teachingBeginner.addEventListener("change", syncTeachingSettings);
  els.teachingComplexity.addEventListener("change", syncTeachingSettings);
  els.teachingContent.addEventListener("click", teachingPanelClick);
  els.mashupTrackA.addEventListener("change", syncMashupSettings);
  els.mashupTrackB.addEventListener("change", syncMashupSettings);
  els.mashupMode.addEventListener("change", syncMashupSettings);
  els.mashupBars.addEventListener("change", syncMashupSettings);
  els.mashupUseStems.addEventListener("change", syncMashupSettings);
  els.mashupVocalPriority.addEventListener("change", syncMashupSettings);
  els.mashupBedPreference.addEventListener("change", syncMashupSettings);
  els.mashupAllowHybridBed.addEventListener("change", syncMashupSettings);
  els.mashupAllowVocalPitchShift.addEventListener("change", syncMashupSettings);
  els.mashupMaxVocalStretch.addEventListener("change", syncMashupSettings);
  els.mashupTimeline.addEventListener("click", mashupTimelineClick);
  els.mashupAnalyzeButton.addEventListener("click", analyzeMashupSegments);
  els.mashupPlanButton.addEventListener("click", generateMashupPlan);
  els.mashupRenderButton.addEventListener("click", renderMashupExport);
  els.matchFileA.addEventListener("change", () => {
    state.match.fileA = els.matchFileA.files?.[0] || null;
    state.match.repairResult = null;
    renderMatchResult();
  });
  els.matchFileB.addEventListener("change", () => {
    state.match.fileB = els.matchFileB.files?.[0] || null;
    state.match.repairResult = null;
    renderMatchResult();
  });
  els.matchButton.addEventListener("click", calculatePairMatch);
  els.repairMatchButton.addEventListener("click", repairPairMatch);

  els.sortMode.addEventListener("change", syncSettings);
  els.crossfade.addEventListener("input", syncSettings);
  els.autoTransition.addEventListener("change", syncSettings);
  els.beatSync.addEventListener("change", syncSettings);
  els.aiPrecision.addEventListener("change", syncSettings);
  els.loudnessNormalize.addEventListener("change", syncSettings);
  els.phraseBars.addEventListener("input", syncSettings);
  els.targetLufs.addEventListener("input", syncSettings);
  els.mixStrategy.addEventListener("change", syncSettings);
  els.filterMode.addEventListener("change", syncSettings);
  els.exportFormat.addEventListener("change", syncSettings);
  [els.eqLow, els.eqMid, els.eqHigh].forEach((input) => input.addEventListener("input", syncSettings));

  els.mixProgress.addEventListener("input", () => {
    state.playbackOffset = Number(els.mixProgress.value);
    renderTransport();
    syncMixTimelinePlayback();
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
  els.cueEditor.addEventListener("input", updateCueEditor);
  els.cueEditor.addEventListener("click", cueEditorClick);
  els.deckMixer.addEventListener("input", updateDeckMixer);
  els.deckMixer.addEventListener("click", deckMixerClick);
  els.mixTimeline.addEventListener("click", seekTimelineClick);
  window.addEventListener("pointerup", () => {
    wave.dragging = null;
  });
}

function syncSettings() {
  const previousScheduleSettings = {
    crossfade: state.settings.crossfade,
    autoTransition: state.settings.autoTransition,
    aiPrecision: state.settings.aiPrecision,
    phraseBars: state.settings.phraseBars,
    mixStrategy: state.settings.mixStrategy,
    filterMode: state.settings.filterMode,
  };
  state.settings.sortMode = els.sortMode.value;
  state.settings.crossfade = Number(els.crossfade.value);
  state.settings.autoTransition = els.autoTransition.checked;
  state.settings.beatSync = els.beatSync.checked;
  state.settings.aiPrecision = els.aiPrecision.checked;
  state.settings.loudnessNormalize = els.loudnessNormalize.checked;
  state.settings.phraseBars = Number(els.phraseBars.value);
  state.settings.targetLufs = Number(els.targetLufs.value);
  state.settings.mixStrategy = els.mixStrategy.value;
  state.settings.equalPowerFade = els.aiPrecision.checked;
  state.settings.filterMode = els.filterMode.value;
  state.settings.exportFormat = els.exportFormat.value;
  state.settings.eq.low = Number(els.eqLow.value);
  state.settings.eq.mid = Number(els.eqMid.value);
  state.settings.eq.high = Number(els.eqHigh.value);
  applyLiveMixerChanges();
  render();
  if (state.isPlaying && scheduleSettingsChanged(previousScheduleSettings)) {
    scheduleLivePreviewRefresh();
  }
}

async function pingBackend() {
  try {
    await fetchJson("/api/health");
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
      style: "unknown",
      style_label: "Unknown",
      style_confidence: 0,
      style_profile: null,
      energy: 0,
      intro_low: 0,
      outro_low: 0,
      introPoint: 0,
      outroPoint: 0,
      mixer: cloneDefaultMixer(),
    };
    state.tracks.push(track);
    state.originalOrder.push(track.localId);
    state.selectedId ||= track.localId;
    render();
    await decodeLocal(track);
    await uploadAndAnalyze(track);
  }
}

function cloneDefaultMixer() {
  return {
    gain: DEFAULT_TRACK_MIXER.gain,
    eq: { ...DEFAULT_TRACK_MIXER.eq },
  };
}

function ensureTrackMixer(track) {
  track.mixer = {
    gain: Number.isFinite(track.mixer?.gain) ? track.mixer.gain : DEFAULT_TRACK_MIXER.gain,
    eq: {
      low: Number.isFinite(track.mixer?.eq?.low) ? track.mixer.eq.low : DEFAULT_TRACK_MIXER.eq.low,
      mid: Number.isFinite(track.mixer?.eq?.mid) ? track.mixer.eq.mid : DEFAULT_TRACK_MIXER.eq.mid,
      high: Number.isFinite(track.mixer?.eq?.high) ? track.mixer.eq.high : DEFAULT_TRACK_MIXER.eq.high,
    },
  };
  return track.mixer;
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
    state.match.result = await fetchJson("/api/match", { method: "POST", body: form });
    setStatus("两歌匹配评分完成");
  } catch (error) {
    state.match.error = error.message || "匹配评分失败";
  } finally {
    state.match.loading = false;
    renderMatchResult();
  }
}

async function repairPairMatch() {
  if (!state.match.fileA || !state.match.fileB) {
    state.match.error = "请先选择两首音频。";
    renderMatchResult();
    return;
  }
  state.match.repairLoading = true;
  state.match.error = "";
  state.match.repairResult = null;
  renderMatchResult();
  try {
    await assertBackendReachable();
    const form = new FormData();
    form.append("file_a", state.match.fileA);
    form.append("file_b", state.match.fileB);
    form.append("process_target", "auto");
    form.append("include_key", "true");
    form.append("include_tempo", "true");
    form.append("include_energy", "true");
    form.append("max_tempo_change_percent", "10");
    form.append("max_pitch_shift_semitones", "4");
    form.append("format", "wav");
    state.match.repairResult = await fetchJson("/api/match/repair", { method: "POST", body: form });
    state.match.result = state.match.repairResult.original_match || state.match.result;
    setStatus("已生成匹配修复版本");
  } catch (error) {
    state.match.error = error.message || "自动修复匹配失败";
  } finally {
    state.match.repairLoading = false;
    renderMatchResult();
  }
}

function renderMatchResult() {
  if (!els.matchResult) return;
  els.matchButton.disabled = state.match.loading || !state.match.fileA || !state.match.fileB;
  if (els.repairMatchButton) els.repairMatchButton.disabled = state.match.loading || state.match.repairLoading || !state.match.fileA || !state.match.fileB;
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
      <em>Recommended direction: ${escapeHtml(result.recommended_direction)}</em>
    </div>
    ${renderTrackAnalysisSummary(result.track_a, result.track_b)}
    ${renderDirectionMatch("A → B", forward)}
    ${renderDirectionMatch("B → A", reverse)}
    ${renderTuneRecommendations(result.tuning_recommendations || [])}
    ${state.match.repairLoading ? `<div class="match-loading">正在生成匹配修复版本，这可能需要几十秒...</div>` : ""}
    ${renderRepairResult(state.match.repairResult)}
  `;
}

function renderTuneRecommendations(recommendations) {
  if (!recommendations.length) {
    return "";
  }
  return `
    <div class="tune-recommendations">
      <div class="direction-head">
        <strong>Harmonic tuning</strong>
        <span>${recommendations.length} option${recommendations.length === 1 ? "" : "s"}</span>
      </div>
      ${recommendations
        .slice(0, 3)
        .map(
          (item) => `
            <div class="tune-option">
              <strong>${escapeHtml(item.track_name || item.source)} -> ${escapeHtml(item.target_camelot)}</strong>
              <span>${item.semitones > 0 ? "+" : ""}${item.semitones} st · risk ${escapeHtml(item.quality_risk)}</span>
              <small>${escapeHtml(item.reason || "")}</small>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderRepairResult(result) {
  if (!result) return "";
  const plan = result.repair_plan || {};
  const repaired = result.repaired_track;
  const original = result.original_match;
  const repairedMatch = result.repaired_match;
  const operations = plan.operations || [];
  const before = original ? `${Math.round(original.overall_score)} / ${escapeHtml(original.overall_level)}` : "--";
  const after = repairedMatch ? `${Math.round(repairedMatch.overall_score)} / ${escapeHtml(repairedMatch.overall_level)}` : "--";
  return `
    <div class="repair-result">
      <div class="direction-head">
        <strong>自动修复匹配</strong>
        <span>${escapeHtml(before)} → ${escapeHtml(after)}</span>
      </div>
      <p>${escapeHtml(plan.explanation || "已生成处理计划。")}</p>
      <div class="preview-report">
        ${operations.length ? operations.map(renderRepairOperation).join("") : "<span>无需处理</span>"}
      </div>
      ${repaired ? `
        <div class="repair-audio">
          <audio controls src="${API}${escapeHtml(repaired.url)}"></audio>
          <a class="download-link" href="${API}${escapeHtml(repaired.url)}" download="${escapeHtml(repaired.name || "match-repaired.wav")}">下载处理后的歌曲</a>
        </div>
      ` : ""}
      ${(result.warnings || []).map((warning) => `<small>${escapeHtml(warning)}</small>`).join("")}
    </div>
  `;
}

function renderRepairOperation(operation) {
  if (operation.type === "pitch") {
    return `<span>Pitch ${escapeHtml(operation.from_camelot || "--")} → ${escapeHtml(operation.target_camelot || "--")} (${operation.semitones > 0 ? "+" : ""}${operation.semitones || 0} st)</span>`;
  }
  if (operation.type === "tempo") {
    return `<span>Tempo ${operation.from_bpm || "--"} → ${operation.target_bpm || "--"} BPM (${operation.change_percent || 0}%)</span>`;
  }
  if (operation.type === "energy") {
    return `<span>Energy ${operation.source_lufs || "--"} → ${operation.target_lufs || "--"} LUFS · low x${operation.low_gain || 1}</span>`;
  }
  return `<span>${escapeHtml(operation.type || "process")}</span>`;
}

function renderTrackAnalysisSummary(trackA, trackB) {
  return `
    <div class="component-grid match-track-summary">
      ${renderTrackSummaryCard("Song A analysis", trackA)}
      ${renderTrackSummaryCard("Song B analysis", trackB)}
    </div>
  `;
}

function renderTrackSummaryCard(label, track) {
  const energyIndex = track?.energy_profile?.energy_index;
  const lufs = track?.energy_profile?.lufs;
  const duration = Number.isFinite(track?.duration) ? `${Math.round(track.duration)}s` : "--";
  const energy = Number.isFinite(energyIndex) ? `${energyIndex}` : "--";
  const loudness = Number.isFinite(lufs) ? `${lufs} LUFS` : "--";
  return renderComponent(
    label,
    Number.isFinite(energyIndex) ? energyIndex : 0,
    `${track?.bpm ?? "--"} BPM · ${track?.key || "--"} · ${track?.camelot || "--"} · energy ${energy} · ${loudness} · ${duration}`,
  );
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
        ${renderComponent("Energy", c.energy.score, c.energy.summary || `diff ${c.energy.delta ?? "--"}`)}
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
    const result = await fetchJson("/api/tracks", { method: "POST", body: form });
    track.id = result.id;
    track.status = "ready";
    track.duration = result.duration || track.duration;
    track.bpm = result.bpm;
    track.beats = result.beats || [];
    track.bars = result.bars || [];
    track.phrases = result.phrases || [];
    track.downbeat_offset = result.downbeat_offset || 0;
    track.beat_confidence = result.beat_confidence || 0;
    track.key = result.key || "未知";
    track.camelot = result.camelot || keyLabelToCamelot(track.key, result.mode);
    track.key_index = result.key_index;
    track.mode = result.mode;
    track.energy = result.energy || 0;
    track.energy_profile = result.energy_profile || null;
    applyTrackStyle(track, result);
    track.energy_curve = result.energy_curve || result.transition_candidates?.energy_curve || null;
    track.vocal_density_curve = result.vocal_density_curve || result.transition_candidates?.vocal_density_curve || null;
    track.sections = result.sections || result.transition_candidates?.sections || null;
    track.intro_low = result.intro_low || 0;
    track.outro_low = result.outro_low || 0;
    track.loudness_lufs = result.loudness_lufs;
    track.true_peak_db = result.true_peak_db;
    track.transition_candidates = result.transition_candidates || null;
    track.peaks = result.peaks?.length ? result.peaks : track.peaks;
    track.introPoint = clamp(track.transition_candidates?.intro ?? track.intro_low ?? track.introPoint, 0.5, Math.max(0.5, track.duration * 0.35));
    track.outroPoint = clamp(track.transition_candidates?.outro ?? (track.duration - (track.outro_low || state.settings.crossfade)), track.duration * 0.55, Math.max(track.duration - 0.5, 0.5));
    ensureTrackMixer(track);
    queueTrackStemSeparation(track);
    setStatus(`完成分析 ${track.name}`);
  } catch (error) {
    applyLocalFallbackAnalysis(track, error);
  } finally {
    render();
  }
}

async function assertBackendReachable() {
  try {
    await fetchJson("/api/health");
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
  track.energy_profile = envelope.energyProfile;
  applyTrackStyle(track, estimateLocalStyle(track.bpm, envelope.energy, envelope.energyProfile));
  track.energy_curve = null;
  track.vocal_density_curve = null;
  track.sections = null;
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
  ensureTrackMixer(track);
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
    energyProfile: buildLocalEnergyProfile(values, samples, sampleRate, average, peak),
    introLow: countLowFrames(values, threshold, true) / frameRate,
    outroLow: countLowFrames(values, threshold, false) / frameRate,
  };
}

function buildLocalEnergyProfile(values, samples, sampleRate, average, peak) {
  const rmsDb = values.map((value) => 20 * Math.log10(Math.max(value, 1e-9))).sort((a, b) => a - b);
  const p10 = percentile(rmsDb, 10);
  const p50 = percentile(rmsDb, 50);
  const p85 = percentile(rmsDb, 85);
  const p95 = percentile(rmsDb, 95);
  const lufs = localLoudness(samples);
  const peakDb = localPeakDb(samples);
  const crest = peakDb - lufs;
  const dynamicRange = Math.max(0, p95 - p10);
  const introCount = Math.max(1, Math.min(values.length, Math.round(16 / (values.length ? samples.length / sampleRate / values.length : 1))));
  const introAvg = avg(values.slice(0, introCount));
  const outroAvg = avg(values.slice(-introCount));
  const fullAvg = Math.max(average, 1e-9);
  const introDelta = 20 * Math.log10(Math.max(introAvg, 1e-9) / fullAvg);
  const outroDelta = 20 * Math.log10(Math.max(outroAvg, 1e-9) / fullAvg);
  const lowFrequencyRatio = 0;
  const components = {
    loudness: clamp01((lufs + 30) / 18) * 100,
    rms_body: clamp01((p85 + 32) / 24) * 100,
    crest_density: clamp01((18 - crest) / 14) * 100,
    low_frequency: 0,
    dynamic_motion: clamp01((dynamicRange - 3) / 14) * 100,
    transition_contrast: Math.min(1, (Math.abs(introDelta) + Math.abs(outroDelta)) / 24) * 100,
  };
  const energyIndex =
    components.loudness * 0.30 +
    components.rms_body * 0.25 +
    components.crest_density * 0.15 +
    components.low_frequency * 0.15 +
    components.dynamic_motion * 0.10 +
    components.transition_contrast * 0.05;
  return {
    energy_index: round1(energyIndex),
    lufs: round2(lufs),
    true_peak_db: round2(peakDb),
    rms_p10_db: round2(p10),
    rms_p50_db: round2(p50),
    rms_p85_db: round2(p85),
    rms_p95_db: round2(p95),
    crest_factor_db: round2(crest),
    low_frequency_ratio: lowFrequencyRatio,
    dynamic_range_db: round2(dynamicRange),
    intro_relative_energy: round4(Math.min(1, introAvg / fullAvg / 2)),
    outro_relative_energy: round4(Math.min(1, outroAvg / fullAvg / 2)),
    intro_delta_db: round2(introDelta),
    outro_delta_db: round2(outroDelta),
    transition_contrast: round4(components.transition_contrast / 100),
    components,
  };
}

function applyTrackStyle(track, source) {
  const profile = source?.style_profile || source?.styleProfile || source || {};
  const style = normalizeStyle(source?.style || source?.genre || profile.primary);
  track.style = style || "unknown";
  track.style_label = source?.style_label || profile.label || styleLabel(track.style);
  track.style_confidence = clamp(Number(source?.style_confidence ?? source?.styleConfidence ?? profile.confidence ?? 0), 0, 1);
  track.style_profile = {
    ...(profile && typeof profile === "object" ? profile : {}),
    primary: track.style,
    label: track.style_label,
    confidence: track.style_confidence,
  };
}

function estimateLocalStyle(bpm, energy, profile = {}) {
  const tempo = Number(bpm) || 0;
  const halfTempo = tempo >= 130 ? tempo / 2 : tempo;
  const low = Number(profile.low_frequency_ratio) || 0;
  const dynamic = clamp((Number(profile.dynamic_range_db) || 0) / 18, 0, 1);
  const energyValue = clamp(Number(energy) || 0, 0, 1);
  const scores = {
    house: rangeAffinity(tempo, 116, 132, 12) * 0.45 + energyValue * 0.25 + low * 0.2 + (1 - dynamic) * 0.1,
    hiphop: rangeAffinity(halfTempo, 70, 104, 18) * 0.46 + low * 0.25 + (1 - energyValue * 0.45) * 0.14 + dynamic * 0.15,
    rock: rangeAffinity(tempo, 86, 172, 24) * 0.35 + dynamic * 0.32 + energyValue * 0.2 + (1 - low) * 0.13,
    pop: rangeAffinity(tempo, 88, 132, 20) * 0.34 + (1 - Math.abs(energyValue - 0.6) / 0.6) * 0.3 + dynamic * 0.14 + (1 - low) * 0.12,
    electronic: rangeAffinity(tempo, 100, 150, 26) * 0.34 + energyValue * 0.28 + low * 0.2 + (1 - dynamic) * 0.18,
    ambient: (1 - energyValue) * 0.45 + dynamic * 0.25 + (1 - low) * 0.15 + (tempo ? 0 : 0.15),
  };
  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [primary, topScore] = ranked[0];
  const secondScore = ranked[1]?.[1] || 0;
  const confidence = clamp(0.2 + topScore * 0.4 + Math.max(0, topScore - secondScore) * 1.2, 0, 0.72);
  return {
    style: primary,
    style_label: styleLabel(primary),
    style_confidence: confidence,
    style_profile: {
      primary,
      label: styleLabel(primary),
      confidence,
      scores,
      method: "browser-fallback-heuristic",
    },
  };
}

function rangeAffinity(value, low, high, grace) {
  if (!value) return 0.35;
  if (value >= low && value <= high) return 1;
  if (value < low) return clamp(1 - (low - value) / grace, 0, 1);
  return clamp(1 - (value - high) / grace, 0, 1);
}

function percentile(sortedValues, p) {
  if (!sortedValues.length) return 0;
  const index = (sortedValues.length - 1) * (p / 100);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sortedValues[lower];
  return sortedValues[lower] * (upper - index) + sortedValues[upper] * (index - lower);
}

function avg(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function round1(value) {
  return Math.round(value * 10) / 10;
}

function round2(value) {
  return Math.round(value * 100) / 100;
}

function round4(value) {
  return Math.round(value * 10000) / 10000;
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
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

async function generateAutoHandoffPlan() {
  const ready = playableTracks();
  if (ready.length < 2) return;
  state.autoHandoff.loading = true;
  state.autoHandoff.error = "";
  render();
  try {
    await assertBackendReachable();
    const plan = await fetchJson("/api/auto-handoff/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        trackIds: ready.map((track) => track.id),
        tracks: ready.map((track) => serializableTrackForBackend(track)),
        settings: {
          phraseBars: state.settings.phraseBars,
          maxTempoChangePercent: 10,
          preferStems: true,
          targetEnergy: "arc",
        },
      }),
    });
    state.autoHandoff.plan = plan;
    applyAutoHandoffPlan(plan);
    setStatus(`Smart Beat Handoff planned: ${Math.round(plan.score || 0)}/100`);
  } catch (error) {
    state.autoHandoff.error = error.message || "Smart Beat Handoff failed";
  } finally {
    state.autoHandoff.loading = false;
    render();
  }
}

function serializableTrackForBackend(track) {
  return {
    id: track.id,
    localId: track.localId,
    name: track.name,
    duration: track.duration,
    bpm: track.bpm,
    key: track.key,
    camelot: track.camelot,
    mode: track.mode,
    energy: track.energy,
    introPoint: track.introPoint,
    outroPoint: track.outroPoint,
    energy_profile: track.energy_profile,
    transition_candidates: track.transition_candidates,
    bars: track.bars,
    phrases: track.phrases,
    sections: track.sections,
    energy_curve: track.energy_curve,
    vocal_density_curve: track.vocal_density_curve,
  };
}

function applyAutoHandoffPlan(plan) {
  if (!plan?.orderedTrackIds?.length) return;
  const readyById = new Map(playableTracks().map((track) => [track.id, track]));
  const sorted = plan.orderedTrackIds.map((id) => readyById.get(id)).filter(Boolean);
  const sortedIds = new Set(sorted.map((track) => track.id));
  const remainingReady = playableTracks().filter((track) => !sortedIds.has(track.id));
  const unresolved = state.tracks.filter((track) => track.status !== "ready");
  state.tracks = [...sorted, ...remainingReady, ...unresolved];

  for (const transition of plan.transitions || []) {
    const prev = readyById.get(transition.fromTrackId);
    const next = readyById.get(transition.toTrackId);
    if (!prev || !next) continue;
    const outgoingTime = Number(transition.outgoingCue?.time);
    const incomingTime = Number(transition.incomingCue?.time);
    if (Number.isFinite(outgoingTime)) prev.outroPoint = clamp(outgoingTime, 0.5, Math.max(0.5, prev.duration - 0.25));
    if (Number.isFinite(incomingTime)) next.introPoint = clamp(incomingTime, 0, Math.max(0, next.duration - 0.25));
    next.autoHandoffTransition = transition;
  }

  const durations = (plan.transitions || []).map((item) => Number(item.durationSec)).filter((value) => Number.isFinite(value) && value > 0);
  if (durations.length) state.settings.crossfade = clamp(Math.round(durations.reduce((sum, value) => sum + value, 0) / durations.length), 2, 24);
  state.settings.aiPrecision = true;
  state.settings.autoTransition = true;
  state.settings.mixStrategy = "auto";
  state.settings.filterMode = "dynamicEq";
  state.playbackOffset = 0;
  applySettingsToControls();
}

function autoHandoffPanelClick(event) {
  const button = event.target.closest("[data-auto-handoff-render]");
  if (!button) return;
  renderAutoHandoffPreviews();
}

async function renderAutoHandoffPreviews() {
  const plan = state.autoHandoff.plan;
  if (!plan?.transitions?.length || state.autoHandoff.rendering) return;
  const readyById = new Map(playableTracks().map((track) => [track.id, track]));
  state.autoHandoff.rendering = true;
  state.autoHandoff.error = "";
  state.autoHandoff.renderedCount = 0;
  render();
  try {
    for (const transition of plan.transitions) {
      const current = readyById.get(transition.fromTrackId);
      const next = readyById.get(transition.toTrackId);
      if (!current?.id || !next?.id) continue;
      const recommendation = recommendationFromAutoHandoff(transition);
      const result = await fetchJson("/api/transition-preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          outgoingTrackId: current.id,
          incomingTrackId: next.id,
          recommendation,
          options: {
            targetMode: "quality",
            useStemSeparation: true,
            stemEngine: "demucs",
            timeStretchEngine: "rubberband",
            preserveFormants: true,
            previewDurationBeforeTransition: 8,
            previewDurationAfterTransition: 8,
            exportFormat: "wav",
            beginnerSafeMode: false,
          },
        }),
      });
      const preview = {
        ...result,
        outgoingLocalId: current.localId,
        incomingLocalId: next.localId,
        outgoingTrackId: current.id,
        incomingTrackId: next.id,
        timelineApplied: true,
      };
      state.teaching.previews[next.localId] = preview;
      await hydrateTeachingPreviewAudio(preview);
      const effective = effectiveTeachingCue(recommendation, preview);
      current.outroPoint = clamp(effective.outgoingTime, 0.5, Math.max(0.5, current.duration - 0.25));
      next.introPoint = clamp(effective.incomingTime, 0, Math.max(0, next.duration - 0.25));
      next.appliedTransitionPreview = serializableTransitionPreview(preview, current, next);
      transition.renderedPreview = {
        url: preview.url,
        method: preview.method,
        renderMethod: preview.processingReport?.renderMethod,
        riskScore: preview.processingReport?.riskScore,
      };
      state.autoHandoff.renderedCount += 1;
      render();
    }
    setStatus(`Rendered ${state.autoHandoff.renderedCount} Smart Beat Handoff previews`);
  } catch (error) {
    state.autoHandoff.error = error.message || "Failed to render Smart Beat Handoff previews";
  } finally {
    state.autoHandoff.rendering = false;
    render();
  }
}

function recommendationFromAutoHandoff(transition) {
  const method = {
    drum_bed_handoff: "beatmix",
    bass_swap_handoff: "bass_swap",
    percussive_loop_bridge: "loop_build",
    vocal_safe_bridge: "echo_out",
    effect_tail_handoff: "echo_out",
  }[transition.type] || "beatmix";
  return {
    method,
    score: clamp((Number(transition.score) || 60) / 100, 0, 1),
    difficulty: transition.risk === "low" ? 2 : transition.risk === "medium" ? 3 : 4,
    reason: transition.explanation || "Smart Beat Handoff",
    overlapDuration: Number(transition.durationSec) || state.settings.crossfade,
    outgoingCue: {
      time: Number(transition.outgoingCue?.time) || 0,
      role: transition.outgoingCue?.role || "exit",
      sectionType: "outro",
      confidence: clamp((Number(transition.outgoingCue?.score) || 50) / 100, 0, 1),
    },
    incomingCue: {
      time: Number(transition.incomingCue?.time) || 0,
      role: transition.incomingCue?.role || "entry",
      sectionType: "intro",
      confidence: clamp((Number(transition.incomingCue?.score) || 50) / 100, 0, 1),
    },
    stepByStep: [],
  };
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
  const relation = camelotRelation(codeA, codeB);
  return {
    same: 0,
    relative_major_minor: 0.08,
    adjacent: 0.12,
    energy_boost: 0.22,
    diagonal_mix: 0.34,
    mood_shifter: 0.44,
    jaws_mix: 0.5,
    clash: 0.85,
    unknown: 0.5,
  }[relation];
}

const KEY_TO_CAMELOT = {
  C: "8B", "C#": "3B", Db: "3B", D: "10B", "D#": "5B", Eb: "5B",
  E: "12B", F: "7B", "F#": "2B", Gb: "2B", G: "9B", "G#": "4B",
  Ab: "4B", A: "11B", "A#": "6B", Bb: "6B", B: "1B",
  Am: "8A", "A#m": "3A", Bbm: "3A", Bm: "10A", Cm: "5A",
  "C#m": "12A", Dbm: "12A", Dm: "7A", "D#m": "2A", Ebm: "2A",
  Em: "9A", Fm: "4A", "F#m": "11A", Gbm: "11A", Gm: "6A",
  "G#m": "1A", Abm: "1A",
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
  const match = String(code || "").trim().toUpperCase().match(/^(\d{1,2})([AB])$/);
  if (!match) return null;
  const num = Number(match[1]);
  if (!Number.isInteger(num) || num < 1 || num > 12) return null;
  return { num, mode: match[2] };
}

function camelotNumDistance(a, b) {
  const diff = Math.abs(a - b);
  return Math.min(diff, 12 - diff);
}

function camelotClockwiseDelta(a, b) {
  const forward = (b - a + 12) % 12;
  if (forward === 0) return 0;
  return forward <= 7 ? forward : forward - 12;
}

function camelotRelation(codeA, codeB) {
  const a = parseCamelot(codeA);
  const b = parseCamelot(codeB);
  if (!a || !b) return "unknown";
  const clockwise = camelotClockwiseDelta(a.num, b.num);
  if (a.num === b.num && a.mode === b.mode) return "same";
  if (a.num === b.num && a.mode !== b.mode) return "relative_major_minor";
  if (a.mode === b.mode && camelotNumDistance(a.num, b.num) === 1) return "adjacent";
  if (a.mode === b.mode && clockwise === 2) return "energy_boost";
  if (a.mode !== b.mode && ((a.mode === "A" && b.mode === "B" && clockwise === -1) || (a.mode === "B" && b.mode === "A" && clockwise === 1))) return "diagonal_mix";
  if (a.mode === b.mode && clockwise === 7) return "jaws_mix";
  if (a.mode !== b.mode && ((a.mode === "A" && b.mode === "B" && clockwise === 3) || (a.mode === "B" && b.mode === "A" && clockwise === -3))) return "mood_shifter";
  return "clash";
}

async function previewMix(offset = 0) {
  const tracks = playableTracks().filter((track) => track.buffer);
  if (!tracks.length) return;
  stopStemDebugger({ keepStatus: true });
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
  const renderedRegions = renderedTransitionRegions(timeline);
  let started = 0;
  timeline.items.forEach((item) => {
    if (item.end <= offset) return;
    const track = item.track;
    const sourceOffset = item.sourceStart + Math.max(0, offset - item.start);
    if (sourceOffset >= track.buffer.duration) return;

    const source = context.createBufferSource();
    source.buffer = track.buffer;
    const envelopeGain = context.createGain();
    const renderedDuckGain = context.createGain();
    const mixerGain = context.createGain();
    const low = context.createBiquadFilter();
    const mid = context.createBiquadFilter();
    const high = context.createBiquadFilter();
    const dynamicLow = context.createBiquadFilter();
    const dynamicMid = context.createBiquadFilter();
    const dynamicHigh = context.createBiquadFilter();
    const transitionFilter = context.createBiquadFilter();
    configureEq(track, low, mid, high, transitionFilter, context.currentTime);
    configureDynamicEq(dynamicLow, dynamicMid, dynamicHigh);
    source
      .connect(low)
      .connect(mid)
      .connect(high)
      .connect(dynamicLow)
      .connect(dynamicMid)
      .connect(dynamicHigh)
      .connect(transitionFilter)
      .connect(envelopeGain)
      .connect(renderedDuckGain)
      .connect(mixerGain)
      .connect(context.destination);

    const localStart = startAt + Math.max(0, item.start - offset);
    mixerGain.gain.setValueAtTime(ensureTrackMixer(track).gain, localStart);
    applyPreviewEnvelope(item, envelopeGain.gain, transitionFilter, dynamicLow, dynamicMid, dynamicHigh, localStart, offset, sourceOffset);
    applyRenderedPreviewDuck(item, renderedDuckGain.gain, renderedRegions, startAt, offset);
    source.start(localStart, sourceOffset);
    source.onended = () => {
      state.activeSources = state.activeSources.filter((itemSource) => itemSource !== source);
      state.activeNodes = state.activeNodes.filter((node) => node.source !== source);
    };
    state.activeSources.push(source);
    state.activeNodes.push({
      source,
      trackId: track.localId,
      mixerGain,
      envelopeGain,
      renderedDuckGain,
      low,
      mid,
      high,
      dynamicLow,
      dynamicMid,
      dynamicHigh,
      transitionFilter,
    });
    started += 1;
  });
  started += scheduleRenderedTransitionPreviews(context, timeline, renderedRegions, startAt, offset);
  return started > 0;
}

function configureEq(track, low, mid, high, transitionFilter, now) {
  const mixer = ensureTrackMixer(track);
  low.type = "lowshelf";
  low.frequency.value = 220;
  low.gain.value = mixerEqDb(mixer.eq.low, "low");
  mid.type = "peaking";
  mid.frequency.value = 1200;
  mid.Q.value = 0.9;
  mid.gain.value = mixerEqDb(mixer.eq.mid, "mid");
  high.type = "highshelf";
  high.frequency.value = 3400;
  high.gain.value = mixerEqDb(mixer.eq.high, "high");
  transitionFilter.type = state.settings.filterMode === "highpassLift" ? "highpass" : "lowpass";
  transitionFilter.frequency.setValueAtTime(state.settings.filterMode === "none" ? 20000 : 16000, now);
}

function renderedTransitionRegions(timeline) {
  return timeline.items
    .filter((item) => item.transitionIn?.renderedPreview?.buffer)
    .map((item) => {
      const preview = item.transitionIn.renderedPreview;
      const previousItem = timeline.items[item.index - 1];
      const rawStart = previousItem.start + (Number(preview.previewStartTime) - previousItem.sourceStart);
      const start = Math.max(previousItem.start, rawStart);
      const previewOffset = Math.max(0, start - rawStart);
      const duration = Math.max(0, Math.min(preview.buffer.duration - previewOffset, item.end - start));
      return {
        start,
        end: start + duration,
        preview,
        previewOffset,
        trackIds: new Set([previousItem.track.localId, item.track.localId]),
      };
    })
    .filter((region) => region.end > region.start + 0.05);
}

function applyRenderedPreviewDuck(item, param, regions, startAt, mixOffset) {
  param.setValueAtTime(1, startAt);
  regions
    .filter((region) => region.trackIds.has(item.track.localId) && region.end > item.start && region.start < item.end)
    .forEach((region) => {
      const duckStart = startAt + Math.max(0, region.start - mixOffset);
      const duckEnd = startAt + Math.max(0, region.end - mixOffset);
      if (duckEnd <= startAt) return;
      const fade = 0.025;
      param.setValueAtTime(1, Math.max(startAt, duckStart - fade));
      param.linearRampToValueAtTime(0.0001, Math.max(startAt + 0.001, duckStart + fade));
      param.setValueAtTime(0.0001, Math.max(startAt + 0.002, duckEnd - fade));
      param.linearRampToValueAtTime(1, Math.max(startAt + 0.003, duckEnd + fade));
    });
}

function scheduleRenderedTransitionPreviews(context, timeline, regions, startAt, offset) {
  let started = 0;
  regions.forEach((region) => {
    if (region.end <= offset) return;
    const sourceOffset = region.previewOffset + Math.max(0, offset - region.start);
    if (sourceOffset >= region.preview.buffer.duration) return;
    const source = context.createBufferSource();
    const gain = context.createGain();
    source.buffer = region.preview.buffer;
    source.connect(gain).connect(context.destination);
    const localStart = startAt + Math.max(0, region.start - offset);
    const playDuration = Math.min(region.end - Math.max(offset, region.start), region.preview.buffer.duration - sourceOffset);
    if (playDuration <= 0.05) return;
    gain.gain.setValueAtTime(1, localStart);
    source.start(localStart, sourceOffset, playDuration);
    source.onended = () => {
      state.activeSources = state.activeSources.filter((itemSource) => itemSource !== source);
    };
    state.activeSources.push(source);
    started += 1;
  });
  return started;
}

function configureDynamicEq(low, mid, high) {
  low.type = "lowshelf";
  low.frequency.value = 220;
  low.gain.value = 0;
  mid.type = "peaking";
  mid.frequency.value = 1200;
  mid.Q.value = 0.9;
  mid.gain.value = 0;
  high.type = "highshelf";
  high.frequency.value = 3400;
  high.gain.value = 0;
}

function mixerEqDb(value, band) {
  return (Number(value) || 0) * 12 + globalEqDb(band);
}

function globalEqDb(band) {
  const value = Number(state.settings.eq[band]) || 0;
  return value * (band === "mid" ? 5 : 6);
}

function applyLiveMixerChanges(trackId = null) {
  if (!state.audioContext || !state.activeNodes.length) return;
  const now = state.audioContext.currentTime;
  state.activeNodes.forEach((node) => {
    if (trackId && node.trackId !== trackId) return;
    const track = state.tracks.find((item) => item.localId === node.trackId);
    if (!track) return;
    const mixer = ensureTrackMixer(track);
    smoothSetAudioParam(node.mixerGain.gain, mixer.gain, now);
    smoothSetAudioParam(node.low.gain, mixerEqDb(mixer.eq.low, "low"), now);
    smoothSetAudioParam(node.mid.gain, mixerEqDb(mixer.eq.mid, "mid"), now);
    smoothSetAudioParam(node.high.gain, mixerEqDb(mixer.eq.high, "high"), now);
  });
}

function smoothSetAudioParam(param, value, now) {
  param.cancelScheduledValues(now);
  param.setTargetAtTime(value, now, 0.015);
}

function scheduleSettingsChanged(previous) {
  return (
    previous.crossfade !== state.settings.crossfade ||
    previous.autoTransition !== state.settings.autoTransition ||
    previous.aiPrecision !== state.settings.aiPrecision ||
    previous.phraseBars !== state.settings.phraseBars ||
    previous.mixStrategy !== state.settings.mixStrategy ||
    previous.filterMode !== state.settings.filterMode
  );
}

function scheduleLivePreviewRefresh() {
  if (state.liveRefreshTimer) window.clearTimeout(state.liveRefreshTimer);
  state.liveRefreshTimer = window.setTimeout(() => {
    state.liveRefreshTimer = null;
    if (state.isPlaying) previewMix(state.playbackOffset);
  }, 120);
}

function applyPreviewEnvelope(item, param, filterNode, low, mid, high, startsAt, mixOffset, sourceOffset) {
  const track = item.track;
  const endAt = startsAt + Math.max(0, track.buffer.duration - sourceOffset);
  const elapsed = Math.max(0, mixOffset - item.start);
  const strategy = transitionStrategyForItem(item);
  const vocalHandoffIn = strategy === "vocalHandoff" && item.transitionIn;
  const vocalHandoffOut = strategy === "vocalHandoff" && item.transitionOut;
  param.cancelScheduledValues(startsAt);
  const entrySeconds = vocalHandoffIn ? Math.min(0.45, item.fadeIn * 0.16) : item.fadeIn;
  const inProgressFadeIn = entrySeconds > 0 && elapsed < entrySeconds;
  param.setValueAtTime(inProgressFadeIn ? elapsed / entrySeconds : 1, startsAt);
  if (inProgressFadeIn) param.linearRampToValueAtTime(1, startsAt + (entrySeconds - elapsed));

  const fadeOutStart = item.fadeOutStart == null ? null : startsAt + Math.max(0, item.fadeOutStart - mixOffset);
  const fadeOutEnd = fadeOutStart == null ? null : fadeOutStart + item.fadeOut;
  if (item.fadeOut > 0 && fadeOutStart != null && fadeOutEnd != null && endAt > fadeOutStart) {
    if (vocalHandoffOut) {
      const releaseStart = fadeOutStart + item.fadeOut * 0.76;
      param.setValueAtTime(1, fadeOutStart);
      param.setValueAtTime(0.92, releaseStart);
      param.linearRampToValueAtTime(0, fadeOutEnd);
    } else {
      param.setValueAtTime(1, fadeOutStart);
      param.linearRampToValueAtTime(0, fadeOutEnd);
    }
    if (state.settings.filterMode === "lowpassSweep") {
      filterNode.type = "lowpass";
      filterNode.frequency.setValueAtTime(16000, fadeOutStart);
      filterNode.frequency.exponentialRampToValueAtTime(900, fadeOutEnd);
    }
  }

  if (state.settings.filterMode === "highpassLift" && item.fadeIn > 0) {
    filterNode.type = "highpass";
    filterNode.frequency.setValueAtTime(700, startsAt);
    filterNode.frequency.exponentialRampToValueAtTime(35, startsAt + item.fadeIn);
  }

  if (state.settings.filterMode === "dynamicEq") {
    automateDynamicEq(track, item, low, mid, high, startsAt, mixOffset, endAt);
  }
}

function automateDynamicEq(track, item, low, mid, high, startsAt, mixOffset, endAt) {
  const baseLow = 0;
  const baseMid = 0;
  const baseHigh = 0;
  const strategy = item.transitionIn
    ? resolveMixStrategy(item.transitionIn.prevTrack, item.track)
    : resolveMixStrategy(item.track, item.transitionOut?.nextTrack);
  const curves = strategyEqCurves(strategy);

  if (item.fadeIn > 0) {
    low.gain.setValueAtTime(baseLow + curves.inLow, startsAt);
    mid.gain.setValueAtTime(baseMid + curves.inMid, startsAt);
    high.gain.setValueAtTime(baseHigh + curves.inHigh, startsAt);
    if (strategy === "vocalHandoff") {
      low.gain.setValueAtTime(baseLow + curves.inLow, startsAt + item.fadeIn * 0.68);
      low.gain.linearRampToValueAtTime(baseLow, startsAt + item.fadeIn);
      mid.gain.linearRampToValueAtTime(baseMid, startsAt + Math.min(0.45, item.fadeIn * 0.18));
      high.gain.linearRampToValueAtTime(baseHigh, startsAt + Math.min(0.6, item.fadeIn * 0.22));
    } else {
      low.gain.linearRampToValueAtTime(baseLow, startsAt + item.fadeIn * 0.45);
      mid.gain.linearRampToValueAtTime(baseMid, startsAt + item.fadeIn);
      high.gain.linearRampToValueAtTime(baseHigh, startsAt + item.fadeIn * 0.8);
    }
  }

  const fadeOutStart = item.fadeOutStart == null ? null : startsAt + Math.max(0, item.fadeOutStart - mixOffset);
  const fadeOutEnd = fadeOutStart == null ? null : fadeOutStart + item.fadeOut;
  if (item.fadeOut > 0 && fadeOutStart != null && fadeOutEnd != null && endAt > fadeOutStart) {
    low.gain.setValueAtTime(baseLow, fadeOutStart);
    mid.gain.setValueAtTime(baseMid, fadeOutStart);
    high.gain.setValueAtTime(baseHigh, fadeOutStart);
    if (strategy === "vocalHandoff") {
      low.gain.linearRampToValueAtTime(baseLow + curves.outLow, fadeOutEnd);
      mid.gain.linearRampToValueAtTime(baseMid + curves.outMid, fadeOutStart + Math.min(0.5, item.fadeOut * 0.16));
      high.gain.linearRampToValueAtTime(baseHigh + curves.outHigh, fadeOutStart + Math.min(0.6, item.fadeOut * 0.18));
    } else {
      low.gain.linearRampToValueAtTime(baseLow + curves.outLow, fadeOutEnd);
      mid.gain.linearRampToValueAtTime(baseMid + curves.outMid, fadeOutEnd);
      high.gain.linearRampToValueAtTime(baseHigh + curves.outHigh, fadeOutEnd);
    }
  }
}

function transitionStrategyForItem(item) {
  if (item.transitionIn) return resolveMixStrategy(item.transitionIn.prevTrack, item.track);
  if (item.transitionOut) return resolveMixStrategy(item.track, item.transitionOut.nextTrack);
  return state.settings.mixStrategy || "auto";
}

function resolveMixStrategy(prevTrack, nextTrack) {
  const selected = state.settings.mixStrategy || "auto";
  if (selected !== "auto") return selected;
  const plannedType = nextTrack?.autoHandoffTransition?.type;
  if (plannedType === "bass_swap_handoff" || plannedType === "drum_bed_handoff") return "bassSwap";
  if (plannedType === "vocal_safe_bridge") return "vocalSafe";
  if (plannedType === "effect_tail_handoff" || plannedType === "percussive_loop_bridge") return "smooth";
  const prevVocal = prevTrack?.transition_candidates?.outro_vocal_density || 0;
  const nextVocal = nextTrack?.transition_candidates?.intro_vocal_density || 0;
  const bpmDelta = Math.abs((prevTrack?.bpm || 0) - (nextTrack?.bpm || 0));
  const prevEnergy = prevTrack?.transition_candidates?.outro_energy || prevTrack?.energy || 0;
  const nextEnergy = nextTrack?.transition_candidates?.intro_energy || nextTrack?.energy || 0;
  const energyLift = (nextTrack?.energy || 0) - (prevTrack?.energy || 0);
  if (bpmDelta > 20) return "smooth";
  if (nextVocal >= 0.32 && prevEnergy >= 0.25 && nextEnergy >= 0.22 && bpmDelta <= 18) return "vocalHandoff";
  if (prevVocal > 0.55 || nextVocal > 0.55) return "vocalSafe";
  if (bpmDelta <= 4 && energyLift > 0.08) return "bassSwap";
  if (bpmDelta > 12) return "smooth";
  return "bassSwap";
}

function strategyEqCurves(strategy) {
  return {
    vocalHandoff: { inLow: -18, inMid: -2, inHigh: -4, outLow: -3, outMid: -15, outHigh: -9 },
    vocalSafe: { inLow: -4, inMid: -12, inHigh: -6, outLow: -8, outMid: -3, outHigh: -1 },
    bassSwap: { inLow: -1, inMid: -8, inHigh: -4, outLow: -14, outMid: -5, outHigh: -2 },
    smooth: { inLow: -5, inMid: -7, inHigh: -7, outLow: -8, outMid: -7, outHigh: -4 },
    quickCut: { inLow: 0, inMid: -4, inHigh: -2, outLow: -16, outMid: -8, outHigh: -6 },
  }[strategy] || { inLow: -2, inMid: -9, inHigh: -5, outLow: -12, outMid: -5, outHigh: -2 };
}

function strategyLabel(strategy) {
  return {
    auto: "AI 自动判断",
    vocalHandoff: "人声接鼓点",
    vocalSafe: "保留人声清晰",
    bassSwap: "低频交换切入",
    smooth: "平滑氛围过渡",
    quickCut: "快速切歌",
  }[strategy] || "AI 自动判断";
}

function strategyActions(strategy) {
  return {
    vocalHandoff: ["上一首保留鼓组律动，快速压掉中频人声", "下一首先接入人声/旋律，鼓和低频在重叠后段再打开"],
    vocalSafe: ["压低新歌中频，等上一首人声离开", "低频轻推，避免主唱和旋律打架"],
    bassSwap: ["旧歌低频快速下潜", "新歌鼓和贝斯提前建立"],
    smooth: ["等功率慢淡化", "中高频缓慢进入，减少突兀切换"],
    quickCut: ["缩短重叠感", "更像 DJ 的快速换歌点"],
  }[strategy] || ["根据人声密度、能量和 BPM 自动选择 EQ 曲线"];
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
  if (state.liveRefreshTimer) window.clearTimeout(state.liveRefreshTimer);
  state.liveRefreshTimer = null;
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
  syncMixTimelinePlayback();
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
    const result = await fetchJson("/api/export", {
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
    await fetchJson("/api/projects", {
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
    const result = await fetchJson("/api/projects");
    state.projects = result.projects || [];
    els.projectList.innerHTML = `<option value="">选择已保存项目</option>${state.projects.map((project) => `<option value="${project.id}">${escapeHtml(project.name)}</option>`).join("")}`;
  } catch {
    state.projects = [];
  }
}

async function loadSelectedProject() {
  if (!els.projectList.value) return;
  try {
    const project = await fetchJson(`/api/projects/${els.projectList.value}`);
    state.settings = { ...state.settings, ...project.settings };
    state.tracks = await Promise.all(project.tracks.map(rehydrateTrack));
    state.originalOrder = state.tracks.map((track) => track.localId);
    state.selectedId = state.tracks[0]?.localId || null;
    applySettingsToControls();
    state.tracks.forEach((track) => queueTrackStemSeparation(track));
    setStatus("项目已加载");
    render();
  } catch (error) {
    setStatus(error.message || "项目加载失败");
  }
}

async function rehydrateTrack(saved) {
  const context = await getAudioContext();
  const response = await fetch(apiUrl(`/api/tracks/${saved.id}/audio`));
  if (!response.ok) throw new Error("加载项目音频失败");
  const arrayBuffer = await response.arrayBuffer();
  const buffer = await context.decodeAudioData(arrayBuffer.slice(0));
  return {
    ...saved,
    file: null,
    buffer,
    localId: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`,
    peaks: saved.peaks?.length ? saved.peaks : peaksFromBuffer(buffer, 900),
    mixer: {
      gain: Number.isFinite(saved.mixer?.gain) ? saved.mixer.gain : DEFAULT_TRACK_MIXER.gain,
      eq: {
        low: Number.isFinite(saved.mixer?.eq?.low) ? saved.mixer.eq.low : DEFAULT_TRACK_MIXER.eq.low,
        mid: Number.isFinite(saved.mixer?.eq?.mid) ? saved.mixer.eq.mid : DEFAULT_TRACK_MIXER.eq.mid,
        high: Number.isFinite(saved.mixer?.eq?.high) ? saved.mixer.eq.high : DEFAULT_TRACK_MIXER.eq.high,
      },
    },
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
    style: track.style,
    style_label: track.style_label,
    style_confidence: track.style_confidence,
    style_profile: track.style_profile,
    beats: track.beats || [],
    bars: track.bars || [],
    phrases: track.phrases || [],
    downbeat_offset: track.downbeat_offset || 0,
    beat_confidence: track.beat_confidence || 0,
    energy: track.energy,
    energy_profile: track.energy_profile,
    energy_curve: track.energy_curve,
    vocal_density_curve: track.vocal_density_curve,
    sections: track.sections,
    intro_low: track.intro_low,
    outro_low: track.outro_low,
    loudness_lufs: track.loudness_lufs,
    true_peak_db: track.true_peak_db,
    transition_candidates: {
      ...(track.transition_candidates || {}),
      intro: track.introPoint,
      outro: track.outroPoint,
      source: "user-adjusted",
    },
    introPoint: track.introPoint,
    outroPoint: track.outroPoint,
    mixer: ensureTrackMixer(track),
    stemMixer: serializeStemMixerSettings(track),
    appliedTransitionPreview: track.appliedTransitionPreview || null,
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
  els.mixStrategy.value = state.settings.mixStrategy || "auto";
  els.filterMode.value = state.settings.filterMode;
  els.teachingEnergy.value = state.teaching.targetEnergy;
  els.teachingBeginner.checked = state.teaching.beginnerMode;
  els.teachingComplexity.value = String(state.teaching.maxComplexity);
  els.exportFormat.value = state.settings.exportFormat;
  els.eqLow.value = state.settings.eq.low;
  els.eqMid.value = state.settings.eq.mid;
  els.eqHigh.value = state.settings.eq.high;
}

function buildTimeline(tracks = playableTracks()) {
  const items = [];
  tracks.forEach((track, index) => {
    ensureTrackMixer(track);
    if (index === 0) {
      items.push({
        track,
        index,
        lane: 0,
        start: 0,
        sourceStart: 0,
        end: track.duration,
        fadeIn: 0,
        fadeOut: 0,
        fadeOutStart: null,
        transitionIn: null,
        transitionOut: null,
      });
      return;
    }

    const previous = tracks[index - 1];
    const previousItem = items[index - 1];
    const plan = planClientTransition(previous, track);
    plan.prevTrack = previous;
    plan.nextTrack = track;
    previousItem.fadeOut = plan.seconds;
    previousItem.fadeOutStart = previousItem.start + Math.max(0, plan.prevOverlapStart - previousItem.sourceStart);
    previousItem.end = Math.min(previousItem.end, previousItem.fadeOutStart + plan.seconds);
    previousItem.transitionOut = plan;

    const start = previousItem.fadeOutStart;
    const sourceStart = plan.nextOverlapStart;
    items.push({
      track,
      index,
      lane: index % 2,
      start,
      sourceStart,
      end: start + Math.max(0, track.duration - sourceStart),
      fadeIn: plan.seconds,
      fadeOut: 0,
      fadeOutStart: null,
      transitionIn: plan,
      transitionOut: null,
    });
  });
  return { items, total: items.at(-1)?.end || 0 };
}

function getTransitionDuration(prev, next) {
  return planClientTransition(prev, next).seconds;
}

function planClientTransition(prev, next) {
  const requested = state.settings.crossfade;
  const maxByLength = Math.max(0.5, Math.min(prev.duration, next.duration) * 0.35);
  let actual = requested;
  let prevOut = prev.outroPoint;
  let nextIn = next.introPoint;
  if (state.settings.aiPrecision) {
    const phrase = phraseTransitionSeconds(prev, next);
    if (phrase) actual = phrase;
    prevOut = prev.transition_candidates?.outro ?? prevOut;
    nextIn = next.transition_candidates?.intro ?? nextIn;
  }
  if (Number.isFinite(prev.outroPoint)) prevOut = prev.outroPoint;
  if (Number.isFinite(next.introPoint)) nextIn = next.introPoint;
  prevOut = clamp(Number(prevOut) || Math.max(0, prev.duration - requested), 0, Math.max(0, prev.duration - 0.25));
  nextIn = clamp(Number(nextIn) || requested, 0, next.duration);
  actual = Math.min(actual, Math.max(0.5, prev.duration - prevOut), Math.max(0.5, nextIn));
  if (state.settings.autoTransition) {
    actual = Math.max(2, Math.min(actual, (prev.outro_low || 0) + (next.intro_low || 0) + 2));
  }
  actual = Math.min(actual, maxByLength);
  const renderedPreview = renderedPreviewForTransition(prev, next);
  return {
    seconds: actual,
    strategy: resolveMixStrategy(prev, next),
    prevOverlapStart: prevOut,
    nextIntro: nextIn,
    nextOverlapStart: resolveMixStrategy(prev, next) === "vocalHandoff" ? nextIn : Math.max(0, nextIn - actual),
    confidence: Math.min(0.95, (prev.transition_candidates?.confidence || 0.35) * 0.5 + (next.transition_candidates?.confidence || 0.35) * 0.5),
    renderedPreview,
  };
}

function renderedPreviewForTransition(prev, next) {
  const preview = teachingPreviewFor(next.localId, prev.localId);
  if (!preview?.buffer || !preview.timelineApplied) return null;
  const outgoingTime = Number(preview.outgoingCue?.time ?? preview.alignment?.outgoingExitTime);
  const incomingTime = Number(preview.incomingCue?.time ?? preview.alignment?.incomingEntryTime);
  if (!Number.isFinite(outgoingTime) || !Number.isFinite(incomingTime)) return null;
  if (Math.abs((prev.outroPoint || 0) - outgoingTime) > 0.35) return null;
  if (Math.abs((next.introPoint || 0) - incomingTime) > 0.35) return null;
  return preview;
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

function activeStemTrack() {
  const playable = playableTracks().filter((track) => track.buffer);
  if (!playable.length) return state.tracks.find((track) => track.buffer) || null;
  const chosen = playable.find((track) => track.localId === state.stemDebugger.trackId);
  return chosen || playable.find((track) => track.localId === state.selectedId) || playable[0] || null;
}

function stemReferenceTracks(active = activeStemTrack()) {
  return playableTracks().filter((track) => track.buffer && track.id && track.id !== active?.id);
}

function selectedStemReferenceTrack(active = activeStemTrack()) {
  const candidates = stemReferenceTracks(active);
  if (!candidates.length) return null;
  const selected = state.stemDebugger.referenceTrackId || state.settings.mixStyleTransfer.referenceTrackId;
  return candidates.find((track) => track.id === selected || track.localId === selected) || candidates[0];
}

function openStemDebugger() {
  const track = selectedTrack()?.buffer ? selectedTrack() : playableTracks().find((item) => item.buffer) || state.tracks.find((item) => item.buffer);
  if (track) state.stemDebugger.trackId = track.localId;
  const reference = selectedStemReferenceTrack(track);
  state.stemDebugger.referenceTrackId = reference?.id || null;
  if (track?.status === "ready") queueTrackStemSeparation(track);
  state.view = "stems";
  stopPreview({ keepStatus: true });
  render();
}

function closeStemDebugger() {
  state.view = "studio";
  stopStemDebugger({ keepStatus: true });
  render();
}

function selectStemDebugTrack() {
  state.stemDebugger.trackId = els.stemTrackSelect.value || null;
  state.stemDebugger.playbackOffset = 0;
  const track = activeStemTrack();
  const reference = selectedStemReferenceTrack(track);
  state.stemDebugger.referenceTrackId = reference?.id || null;
  if (track?.status === "ready") queueTrackStemSeparation(track);
  if (state.stemDebugger.isPlaying) playStemDebugger(0);
  renderStemDebugger();
}

function ensureStemControl(stemId) {
  if (!state.stemDebugger.controls[stemId]) {
    state.stemDebugger.controls[stemId] = { gain: 1, mute: false, solo: false };
  }
  return state.stemDebugger.controls[stemId];
}

function selectStemReferenceTrack() {
  state.stemDebugger.referenceTrackId = els.stemReferenceSelect.value || null;
  state.settings.mixStyleTransfer.referenceTrackId = state.stemDebugger.referenceTrackId;
  renderStemMixResult();
}

function neutralStemStyle() {
  return {
    gainDb: 0,
    pan: 0.5,
    eqDb: { low: 0, mid: 0, high: 0 },
    compressor: { thresholdDb: 0, ratio: 1, attackMs: 25, releaseMs: 100, makeupGainDb: 0 },
  };
}

function effectiveStemStyle(stemId) {
  if (!state.settings.mixStyleTransfer.enabled) return neutralStemStyle();
  const stem = state.settings.mixStyleTransfer.params?.stems?.[stemId];
  if (!stem) return neutralStemStyle();
  return {
    gainDb: Number(stem.gainDb) || 0,
    pan: Number.isFinite(stem.pan) ? stem.pan : 0.5,
    eqDb: {
      low: Number(stem.eqDb?.low) || 0,
      mid: Number(stem.eqDb?.mid) || 0,
      high: Number(stem.eqDb?.high) || 0,
    },
    compressor: {
      thresholdDb: Number(stem.compressor?.thresholdDb) || 0,
      ratio: Number(stem.compressor?.ratio) || 1,
      attackMs: Number(stem.compressor?.attackMs) || 25,
      releaseMs: Number(stem.compressor?.releaseMs) || 100,
      makeupGainDb: Number(stem.compressor?.makeupGainDb) || 0,
    },
    reverbSend: Number(stem.reverbSend) || 0,
  };
}

function effectiveMasterStyle() {
  if (!state.settings.mixStyleTransfer.enabled) {
    return { gainDb: 0, eqDb: { low: 0, mid: 0, high: 0 }, compressor: { thresholdDb: 0, ratio: 1, attackMs: 25, releaseMs: 100, makeupGainDb: 0 } };
  }
  const master = state.settings.mixStyleTransfer.params?.master || {};
  return {
    gainDb: Number(master.gainDb) || 0,
    eqDb: {
      low: Number(master.eqDb?.low) || 0,
      mid: Number(master.eqDb?.mid) || 0,
      high: Number(master.eqDb?.high) || 0,
    },
    compressor: {
      thresholdDb: Number(master.compressor?.thresholdDb) || 0,
      ratio: Number(master.compressor?.ratio) || 1,
      attackMs: Number(master.compressor?.attackMs) || 25,
      releaseMs: Number(master.compressor?.releaseMs) || 100,
      makeupGainDb: Number(master.compressor?.makeupGainDb) || 0,
    },
    targetLufs: Number(master.targetLufs) || state.settings.targetLufs,
    stereoWidth: Number(master.stereoWidth) || 1,
  };
}

function serializeStemMixerSettings(track = null) {
  const transferTrackId = state.settings.mixStyleTransfer.trackId;
  const trackMatchesTransfer = !track || !transferTrackId || track.id === transferTrackId;
  const enabled = Boolean(state.settings.mixStyleTransfer.enabled && trackMatchesTransfer);
  const soloActive = anyStemSolo();
  return {
    enabled: enabled || (trackMatchesTransfer && STEMS.some((stem) => {
      const control = ensureStemControl(stem.id);
      return Math.abs(control.gain - 1) > 0.001 || control.mute || control.solo;
    })),
    source: "reference-guided-auto-mix",
    trackId: transferTrackId,
    referenceTrackId: state.settings.mixStyleTransfer.referenceTrackId,
    referenceTrackName: state.settings.mixStyleTransfer.referenceTrackName,
    outputUrl: state.settings.mixStyleTransfer.result?.url || "",
    reportUrl: state.settings.mixStyleTransfer.result?.reportUrl || "",
    stems: Object.fromEntries(STEMS.map((stem) => {
      const control = ensureStemControl(stem.id);
      const style = enabled ? effectiveStemStyle(stem.id) : neutralStemStyle();
      const audible = !control.mute && (!soloActive || control.solo);
      return [stem.id, {
        gain: audible ? control.gain : 0,
        mute: control.mute,
        solo: control.solo,
        pan: style.pan,
        eqDb: style.eqDb,
        compressor: style.compressor,
      }];
    })),
    master: effectiveMasterStyle(),
  };
}

function anyStemSolo() {
  return STEMS.some((stem) => ensureStemControl(stem.id).solo);
}

function effectiveStemGain(stemId) {
  const control = ensureStemControl(stemId);
  if (control.mute) return 0;
  if (anyStemSolo() && !control.solo) return 0;
  return clamp(Number(control.gain) || 0, 0, 1.5);
}

function updateStemControl(event) {
  const input = event.target.closest("input[data-stem-volume]");
  if (!input) return;
  const control = ensureStemControl(input.dataset.stemVolume);
  control.gain = Number(input.value);
  updateStemControlReadout(input, control);
  applyLiveStemControls(input.dataset.stemVolume);
}

function stemDeckClick(event) {
  const button = event.target.closest("button[data-stem-action]");
  if (!button) return;
  const control = ensureStemControl(button.dataset.stem);
  if (button.dataset.stemAction === "mute") control.mute = !control.mute;
  if (button.dataset.stemAction === "solo") control.solo = !control.solo;
  applyLiveStemControls();
  renderStemDebugger();
}

function updateStemControlReadout(input, control) {
  const readout = input.closest(".stem-control-strip")?.querySelector("[data-stem-volume-readout]");
  if (readout) readout.textContent = `${Math.round(control.gain * 100)}%`;
}

function toggleStemDebuggerPlayback() {
  if (state.stemDebugger.isPlaying) {
    stopStemDebugger({ keepStatus: true });
    renderStemDebugger();
  } else {
    playStemDebugger(state.stemDebugger.playbackOffset);
  }
}

function stopStemDebuggerAndReset() {
  stopStemDebugger();
  state.stemDebugger.playbackOffset = 0;
  renderStemDebugger();
}

function trackHasRealStems(track) {
  return Boolean(track?.stems && STEMS.every((stem) => track.stems[stem.id]?.buffer));
}

function stemBufferFor(track, stemId) {
  return track?.stems?.[stemId]?.buffer || track?.buffer || null;
}

function isStemPending(track) {
  return track?.stemStatus === "queued" || track?.stemStatus === "loading";
}

function queueTrackStemSeparation(track, options = {}) {
  if (!track?.id || track.status !== "ready") return;
  if (!options.force && (trackHasRealStems(track) || isStemPending(track))) return;
  track.stemStatus = "queued";
  track.stemError = "";
  renderStemDebugger();
  stemSeparationChain = stemSeparationChain
    .catch(() => {})
    .then(() => runTrackStemSeparation(track, options));
}

async function separateActiveStemTrack() {
  const track = activeStemTrack();
  if (!track?.id || track.status !== "ready") {
    setStatus("\u8bf7\u5148\u7b49\u5f85\u66f2\u76ee\u4e0a\u4f20\u5e76\u5206\u6790\u5b8c\u6210");
    return;
  }
  queueTrackStemSeparation(track, { force: true });
}

async function runStemReferenceMix() {
  const track = activeStemTrack();
  const reference = selectedStemReferenceTrack(track);
  if (!track?.id || track.status !== "ready") {
    setStatus("\u8bf7\u5148\u9009\u62e9\u5df2\u5b8c\u6210\u5206\u6790\u7684\u97f3\u9891");
    return;
  }
  if (!trackHasRealStems(track)) {
    setStatus("\u53c2\u8003\u66f2\u81ea\u52a8\u6df7\u97f3\u9700\u8981\u5148\u5b8c\u6210 Demucs \u771f\u5206\u8f68");
    queueTrackStemSeparation(track);
    renderStemDebugger();
    return;
  }
  if (!reference?.id) {
    setStatus("\u8bf7\u5728\u5206\u8f68\u754c\u9762\u9009\u62e9\u4e00\u9996\u53c2\u8003\u66f2");
    return;
  }
  state.stemDebugger.isAutoMixing = true;
  state.settings.mixStyleTransfer.result = null;
  renderStemDebugger();
  try {
    setStatus(`\u6b63\u5728\u7528 ${reference.name} \u751f\u6210\u53c2\u8003\u66f2\u81ea\u52a8\u6df7\u97f3`);
    await assertBackendReachable();
    const result = await fetchJson(`/api/tracks/${track.id}/reference-mix`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        referenceTrackId: reference.id,
        style: "auto",
        optimize: true,
        optimizeSeconds: 30,
        optimizeTrials: 18,
      }),
    });
    applyReferenceMixResult(track, reference, result);
    if (state.stemDebugger.isPlaying) playStemDebugger(state.stemDebugger.playbackOffset);
    setStatus(`\u5df2\u751f\u6210\u53c2\u8003\u66f2\u81ea\u52a8\u6df7\u97f3: ${reference.name}`);
  } catch (error) {
    state.settings.mixStyleTransfer.result = { error: error.message || "\u53c2\u8003\u66f2\u81ea\u52a8\u6df7\u97f3\u5931\u8d25" };
    setStatus(state.settings.mixStyleTransfer.result.error);
  } finally {
    state.stemDebugger.isAutoMixing = false;
    renderStemDebugger();
  }
}

function applyReferenceMixResult(track, reference, result) {
  const mixer = result?.mixer;
  if (!mixer?.stems) throw new Error("\u540e\u7aef\u672a\u8fd4\u56de\u53ef\u7528\u7684\u6df7\u97f3\u53c2\u6570");
  state.settings.mixStyleTransfer.enabled = true;
  state.settings.mixStyleTransfer.trackId = track.id;
  state.settings.mixStyleTransfer.referenceTrackId = reference.id;
  state.settings.mixStyleTransfer.referenceTrackName = reference.name;
  state.settings.mixStyleTransfer.params = mixer;
  state.settings.mixStyleTransfer.result = result;
  state.stemDebugger.referenceTrackId = reference.id;
  STEMS.forEach((stem) => {
    const control = ensureStemControl(stem.id);
    const style = effectiveStemStyle(stem.id);
    control.gain = clamp(dbToGain(style.gainDb), 0, 1.5);
    control.mute = false;
    control.solo = false;
  });
  track.stemReferenceMix = result;
  applyLiveStemControls();
}

async function runTrackStemSeparation(track, options = {}) {
  if (!track?.id || track.status !== "ready") return;
  track.stemStatus = "loading";
  track.stemError = "";
  renderStemDebugger();
  try {
    setStatus(`Demucs \u6b63\u5728\u5206\u79bb ${track.name}`);
    await assertBackendReachable();
    const result = await fetchJson(`/api/tracks/${track.id}/stems`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device: "auto", force: Boolean(options.force) }),
    });
    await hydrateTrackStems(track, result);
    track.stemStatus = "ready";
    track.stemEngine = result.engine || "demucs";
    if (state.view === "stems" && state.stemDebugger.isPlaying && activeStemTrack()?.localId === track.localId) {
      playStemDebugger(state.stemDebugger.playbackOffset);
    }
    setStatus(result.cached ? `\u5df2\u8f7d\u5165\u7f13\u5b58\u5206\u8f68 ${track.name}` : `Demucs \u5206\u8f68\u5b8c\u6210 ${track.name}`);
  } catch (error) {
    track.stemStatus = "error";
    track.stemError = error.message || "Demucs \u5206\u8f68\u5931\u8d25";
    setStatus(track.stemError);
  } finally {
    renderStemDebugger();
  }
}

async function hydrateTrackStems(track, result) {
  const context = await getAudioContext();
  const stems = result?.stems || {};
  track.stems = track.stems || {};
  await Promise.all(
    STEMS.map(async (stem) => {
      const item = stems[stem.id];
      if (!item?.url) throw new Error(`Missing ${stem.id} stem`);
      const response = await fetch(apiUrl(item.url));
      if (!response.ok) throw new Error(`${stem.label} stem download failed`);
      const arrayBuffer = await response.arrayBuffer();
      const buffer = await context.decodeAudioData(arrayBuffer.slice(0));
      track.stems[stem.id] = {
        ...item,
        buffer,
        peaks: peaksFromBuffer(buffer, 900),
      };
    }),
  );
}

async function playStemDebugger(offset = 0) {
  const track = activeStemTrack();
  if (!track?.buffer) return;
  stopPreview({ keepStatus: true });
  stopStemDebugger({ keepStatus: true });
  const context = await getAudioContext();
  const safeOffset = clamp(Number(offset) || 0, 0, Math.max(0, track.buffer.duration - 0.05));
  const startAt = context.currentTime + 0.08;

  STEMS.forEach((stem) => {
    const stemBuffer = stemBufferFor(track, stem.id);
    if (!stemBuffer) return;
    const source = context.createBufferSource();
    const outputGain = context.createGain();
    source.buffer = stemBuffer;
    connectStemChain(context, source, outputGain, trackHasRealStems(track) ? "none" : stem.filter, stem.id);
    outputGain.gain.setValueAtTime(effectiveStemGain(stem.id), startAt);
    outputGain.connect(context.destination);
    source.start(startAt, safeOffset);
    source.onended = () => {
      state.stemDebugger.activeSources = state.stemDebugger.activeSources.filter((itemSource) => itemSource !== source);
      state.stemDebugger.activeNodes = state.stemDebugger.activeNodes.filter((node) => node.source !== source);
    };
    state.stemDebugger.activeSources.push(source);
    state.stemDebugger.activeNodes.push({ source, stemId: stem.id, outputGain });
  });

  state.stemDebugger.isPlaying = true;
  state.stemDebugger.playStartContextTime = context.currentTime;
  state.stemDebugger.playStartOffset = safeOffset;
  state.stemDebugger.playbackOffset = safeOffset;
  state.stemDebugger.timer = window.setInterval(tickStemPlayback, 80);
  setStatus("\u6b63\u5728\u5206\u8f68\u8c03\u8bd5");
  renderStemDebugger();
}

function connectStemChain(context, source, outputGain, filter, stemId) {
  const input = context.createGain();
  source.connect(input);
  let tail = input;
  if (filter === "bass") {
    const lowpass = context.createBiquadFilter();
    lowpass.type = "lowpass";
    lowpass.frequency.value = 185;
    lowpass.Q.value = 0.9;
    input.connect(lowpass);
    tail = lowpass;
  }
  if (filter === "vocal") {
    const highpass = context.createBiquadFilter();
    const lowpass = context.createBiquadFilter();
    const presence = context.createBiquadFilter();
    highpass.type = "highpass";
    highpass.frequency.value = 220;
    lowpass.type = "lowpass";
    lowpass.frequency.value = 3600;
    presence.type = "peaking";
    presence.frequency.value = 1300;
    presence.Q.value = 0.85;
    presence.gain.value = 3.5;
    input.connect(highpass).connect(lowpass).connect(presence);
    tail = presence;
  }
  if (filter === "drums") {
    const highpass = context.createBiquadFilter();
    const snap = context.createBiquadFilter();
    highpass.type = "highpass";
    highpass.frequency.value = 90;
    snap.type = "peaking";
    snap.frequency.value = 5200;
    snap.Q.value = 1.2;
    snap.gain.value = 4;
    input.connect(highpass).connect(snap);
    tail = snap;
  }
  if (filter === "other") {
    const highpass = context.createBiquadFilter();
    const notch = context.createBiquadFilter();
    highpass.type = "highpass";
    highpass.frequency.value = 650;
    notch.type = "notch";
    notch.frequency.value = 1400;
    notch.Q.value = 0.7;
    input.connect(highpass).connect(notch);
    tail = notch;
  }
  connectStemStyleChain(context, tail, outputGain, stemId);
}

function connectStemStyleChain(context, input, outputGain, stemId) {
  if (!state.settings.mixStyleTransfer.enabled) {
    input.connect(outputGain);
    return;
  }
  const style = effectiveStemStyle(stemId);
  const low = context.createBiquadFilter();
  const mid = context.createBiquadFilter();
  const high = context.createBiquadFilter();
  const compressor = context.createDynamicsCompressor();
  low.type = "lowshelf";
  low.frequency.value = 160;
  low.gain.value = style.eqDb.low;
  mid.type = "peaking";
  mid.frequency.value = 1800;
  mid.Q.value = 0.9;
  mid.gain.value = style.eqDb.mid;
  high.type = "highshelf";
  high.frequency.value = 7200;
  high.gain.value = style.eqDb.high;
  compressor.threshold.value = style.compressor.thresholdDb;
  compressor.knee.value = 6;
  compressor.ratio.value = style.compressor.ratio;
  compressor.attack.value = style.compressor.attackMs / 1000;
  compressor.release.value = style.compressor.releaseMs / 1000;
  const makeup = context.createGain();
  makeup.gain.value = dbToGain(style.compressor.makeupGainDb);
  if (context.createStereoPanner) {
    const panner = context.createStereoPanner();
    panner.pan.value = clamp((style.pan - 0.5) * 2, -1, 1);
    input.connect(low).connect(mid).connect(high).connect(compressor).connect(makeup).connect(panner).connect(outputGain);
    return;
  }
  input.connect(low).connect(mid).connect(high).connect(compressor).connect(makeup).connect(outputGain);
}

function applyLiveStemControls(stemId = null) {
  if (!state.audioContext) return;
  const now = state.audioContext.currentTime;
  state.stemDebugger.activeNodes.forEach((node) => {
    if (stemId && node.stemId !== stemId) return;
    smoothSetAudioParam(node.outputGain.gain, effectiveStemGain(node.stemId), now);
  });
}

function stopStemDebugger(options = {}) {
  state.stemDebugger.activeSources.forEach((source) => {
    try {
      source.stop();
    } catch {
      // Source may already have ended.
    }
  });
  state.stemDebugger.activeSources = [];
  state.stemDebugger.activeNodes = [];
  state.stemDebugger.isPlaying = false;
  if (state.stemDebugger.timer) window.clearInterval(state.stemDebugger.timer);
  state.stemDebugger.timer = null;
  if (!options.keepStatus && state.view === "stems") setStatus("\u5206\u8f68\u8c03\u8bd5\u5df2\u505c\u6b62");
}

function tickStemPlayback() {
  const track = activeStemTrack();
  if (!state.audioContext || !state.stemDebugger.isPlaying || !track?.duration) return;
  state.stemDebugger.playbackOffset = Math.min(
    track.duration,
    state.stemDebugger.playStartOffset + (state.audioContext.currentTime - state.stemDebugger.playStartContextTime),
  );
  if (state.stemDebugger.playbackOffset >= track.duration) {
    stopStemDebugger({ keepStatus: true });
    state.stemDebugger.playbackOffset = 0;
  }
  renderStemTransport();
  drawStemWaveforms();
}

function activeTransition(timeline = buildTimeline()) {
  if (timeline.items.length < 2) return null;
  const active = timeline.items.slice(1).find((item) => {
    const overlapStart = item.start;
    const overlapEnd = item.start + item.fadeIn;
    return state.playbackOffset >= overlapStart && state.playbackOffset <= overlapEnd;
  });
  if (active) {
    return { prev: timeline.items[active.index - 1], next: active, plan: active.transitionIn };
  }
  const selectedIndex = timeline.items.findIndex((item) => item.track.localId === state.selectedId);
  const nextIndex = selectedIndex >= 0 ? Math.min(selectedIndex + 1, timeline.items.length - 1) : 1;
  const index = Math.max(1, nextIndex);
  return { prev: timeline.items[index - 1], next: timeline.items[index], plan: timeline.items[index].transitionIn };
}

function updateDeckMixer(event) {
  const input = event.target.closest("input[data-mixer]");
  if (!input) return;
  const track = state.tracks.find((item) => item.localId === input.dataset.id);
  if (!track) return;
  const mixer = ensureTrackMixer(track);
  const value = Number(input.value);
  if (input.dataset.mixer === "gain") mixer.gain = value;
  if (input.dataset.mixer in mixer.eq) mixer.eq[input.dataset.mixer] = value;
  updateMixerReadout(input, mixer);
  applyLiveMixerChanges(track.localId);
}

function updateMixerReadout(input, mixer) {
  const readout = input.closest(".deck-slider")?.querySelector("[data-mixer-readout]");
  if (!readout) return;
  const param = input.dataset.mixer;
  if (param === "gain") {
    readout.textContent = `${Math.round(mixer.gain * 100)}%`;
  } else if (param in mixer.eq) {
    readout.textContent = `${Math.round(mixer.eq[param] * 12)} dB`;
  }
}

function deckMixerClick(event) {
  const button = event.target.closest("button[data-action='select']");
  if (!button) return;
  selectTrack(button.dataset.id);
}

function jumpToActiveTransition() {
  const transition = activeTransition();
  if (!transition) return;
  state.playbackOffset = transition.next.start;
  if (state.isPlaying) previewMix(state.playbackOffset);
  render();
}

function seekTimelineClick(event) {
  const clip = event.target.closest("[data-time]");
  if (!clip) return;
  state.playbackOffset = Number(clip.dataset.time) || 0;
  if (state.isPlaying) previewMix(state.playbackOffset);
  render();
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
  if (state.stemDebugger.trackId === localId) {
    stopStemDebugger({ keepStatus: true });
    state.stemDebugger.trackId = playableTracks()[0]?.localId || state.tracks.find((track) => track.buffer)?.localId || null;
  }
  render();
}

function selectTrack(localId) {
  state.selectedId = localId;
  render();
}

function toggleTeachingPanel() {
  state.teaching.open = !state.teaching.open;
  if (state.teaching.open && !selectedTrack()) state.selectedId = playableTracks()[0]?.localId || state.selectedId;
  render();
}

function syncTeachingSettings() {
  state.teaching.targetEnergy = els.teachingEnergy.value;
  state.teaching.beginnerMode = els.teachingBeginner.checked;
  state.teaching.maxComplexity = Number(els.teachingComplexity.value) || 3;
  renderTeachingPanel();
}

async function teachingPanelClick(event) {
  const previewButton = event.target.closest("[data-teaching-preview]");
  if (previewButton) {
    await generateTeachingPreview(previewButton.dataset.teachingPreview);
    return;
  }
  const button = event.target.closest("[data-teaching-apply]");
  if (!button) return;
  await applyTeachingRecommendation(button.dataset.teachingApply);
}

async function generateTeachingPreview(nextId) {
  const current = selectedTrack()?.status === "ready" ? selectedTrack() : playableTracks()[0];
  const next = state.tracks.find((track) => track.localId === nextId);
  if (!current?.id || !next?.id) {
    setStatus("需要先上传并完成后端分析，才能生成真实过渡试听");
    return;
  }
  const rec = recommendTransition(toTeachingAnalysis(current), toTeachingAnalysis(next), {
    targetEnergy: state.teaching.targetEnergy,
    beginnerMode: state.teaching.beginnerMode,
    maxComplexity: state.teaching.maxComplexity,
  })[0];
  state.teaching.loadingPreviewId = nextId;
  renderTeachingPanel();
  try {
    const result = await fetchJson("/api/transition-preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        outgoingTrackId: current.id,
        incomingTrackId: next.id,
        recommendation: rec,
        options: {
          targetMode: "quality",
          useStemSeparation: true,
          stemEngine: "demucs",
          timeStretchEngine: "rubberband",
          preserveFormants: true,
          previewDurationBeforeTransition: 8,
          previewDurationAfterTransition: 8,
          exportFormat: "wav",
          beginnerSafeMode: state.teaching.beginnerMode,
        },
      }),
    });
    const preview = {
      ...result,
      outgoingLocalId: current.localId,
      incomingLocalId: next.localId,
      outgoingTrackId: current.id,
      incomingTrackId: next.id,
    };
    state.teaching.previews[nextId] = preview;
    await hydrateTeachingPreviewAudio(preview);
    setStatus("已生成无缝过渡试听");
    return preview;
  } catch (error) {
    setStatus(error.message || "过渡试听生成失败");
    return null;
  } finally {
    state.teaching.loadingPreviewId = null;
    renderTeachingPanel();
  }
}

async function applyTeachingRecommendation(nextId) {
  const current = selectedTrack()?.status === "ready" ? selectedTrack() : playableTracks()[0];
  const next = state.tracks.find((track) => track.localId === nextId);
  if (!current || !next || current.localId === next.localId) return;
  const rec = recommendTransition(toTeachingAnalysis(current), toTeachingAnalysis(next), {
    targetEnergy: state.teaching.targetEnergy,
    beginnerMode: state.teaching.beginnerMode,
    maxComplexity: state.teaching.maxComplexity,
  })[0];
  let preview = teachingPreviewFor(nextId, current.localId);
  if (!preview) {
    setStatus("先生成无缝试听，确保使用和导出走同一份过渡音频");
    preview = await generateTeachingPreview(nextId);
    if (!preview) return;
  }
  const effective = effectiveTeachingCue(rec, preview);
  preview.timelineApplied = true;

  current.outroPoint = clamp(effective.outgoingTime, 0.5, Math.max(0.5, current.duration - 0.25));
  next.introPoint = clamp(effective.incomingTime, 0, Math.max(0, next.duration - 0.25));
  next.appliedTransitionPreview = serializableTransitionPreview(preview, current, next);
  applyRecommendedMixSettings({ ...rec, method: effective.method, overlapDuration: effective.overlapDuration });

  const currentIndex = state.tracks.findIndex((track) => track.localId === current.localId);
  const nextIndex = state.tracks.findIndex((track) => track.localId === next.localId);
  if (currentIndex >= 0 && nextIndex >= 0 && nextIndex !== currentIndex + 1) {
    const [picked] = state.tracks.splice(nextIndex, 1);
    const insertAt = state.tracks.findIndex((track) => track.localId === current.localId) + 1;
    state.tracks.splice(insertAt, 0, picked);
  }
  state.selectedId = current.localId;
  setStatus(`已应用教学接法：${methodLabel(effective.method)} · A OUT ${formatTime(effective.outgoingTime)} / B IN ${formatTime(effective.incomingTime)}`);
  applySettingsToControls();
  render();
}

function applyRecommendedMixSettings(rec) {
  state.settings.crossfade = clamp(Math.round(rec.overlapDuration || 2), 2, 24);
  if (rec.method === "beatmix" || rec.method === "bass_swap") {
    state.settings.mixStrategy = "bassSwap";
    state.settings.filterMode = "dynamicEq";
  } else if (rec.method === "quick_cut") {
    state.settings.mixStrategy = "quickCut";
    state.settings.filterMode = "dynamicEq";
    state.settings.crossfade = 2;
  } else if (rec.method === "echo_out" || rec.method === "breakdown_switch") {
    state.settings.mixStrategy = "smooth";
    state.settings.filterMode = "highpassLift";
  } else {
    state.settings.mixStrategy = "smooth";
    state.settings.filterMode = "dynamicEq";
  }
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
      state.playbackOffset = clamp(item.start + localTime - item.sourceStart, 0, timeline.total);
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
  renderShellView();
  renderMetrics();
  renderTable();
  renderTransport();
  renderMixTimeline();
  renderDeckMixer();
  renderAutoHandoffPanel();
  renderMatchResult();
  renderTeachingPanel();
  renderMashupPanel();
  if (state.view === "studio") drawWaveform();
  renderStemDebugger();
}

function renderShellView() {
  const stemsOpen = state.view === "stems";
  els.studioView.hidden = stemsOpen;
  els.stemDebuggerView.hidden = !stemsOpen;
  els.stemDebuggerToggle.classList.toggle("active", stemsOpen);
}

function renderMetrics() {
  const playable = playableTracks();
  els.trackCount.textContent = state.tracks.length;
  els.mixDuration.textContent = formatTime(getMixDurationSeconds());
  els.crossfadeValue.textContent = `${state.settings.crossfade}s`;
  els.phraseBarsValue.textContent = `${state.settings.phraseBars} bars`;
  els.targetLufsValue.textContent = `${state.settings.targetLufs} LUFS`;
  els.sortButton.disabled = playable.length < 2;
  els.autoHandoffButton.disabled = playable.length < 2 || state.autoHandoff.loading;
  els.playButton.disabled = playable.length < 1;
  els.restartButton.disabled = playable.length < 1;
  els.stopButton.disabled = !state.isPlaying;
  els.exportButton.disabled = playable.length < 1 || state.isExporting;
}

function renderAutoHandoffPanel() {
  if (!els.autoHandoffPanel) return;
  const { loading, rendering, renderedCount, plan, error } = state.autoHandoff;
  if (loading) {
    els.autoHandoffPanel.innerHTML = `<div class="auto-handoff-card">Planning Smart Beat Handoff...</div>`;
    return;
  }
  if (error) {
    els.autoHandoffPanel.innerHTML = `<div class="auto-handoff-card error">${escapeHtml(error)}</div>`;
    return;
  }
  if (!plan) {
    els.autoHandoffPanel.innerHTML = `
      <div class="auto-handoff-card muted">
        <strong>Smart Beat Handoff</strong>
        <span>Generate an Auto-DJ order and stem-aware transition plan after at least two songs finish analysis.</span>
      </div>
    `;
    return;
  }
  els.autoHandoffPanel.innerHTML = `
    <div class="auto-handoff-card">
      <div class="auto-handoff-head">
        <strong>Smart Beat Handoff ${Math.round(plan.score || 0)}/100</strong>
        <span>${escapeHtml(plan.summary || "")}</span>
      </div>
      <button type="button" data-auto-handoff-render ${rendering ? "disabled" : ""}>
        ${rendering ? `Rendering ${renderedCount}/${(plan.transitions || []).length}...` : "Render Real Handoff Audio"}
      </button>
      <div class="auto-handoff-list">
        ${(plan.transitions || []).map(renderAutoHandoffTransition).join("")}
      </div>
    </div>
  `;
}

function renderAutoHandoffTransition(transition) {
  const bed = transition.rhythmBed || {};
  const automation = transition.automation || {};
  const warnings = transition.warnings || [];
  const cueReasons = [
    ...(transition.outgoingCue?.reasons || []).map((item) => `A: ${item}`),
    ...(transition.incomingCue?.reasons || []).map((item) => `B: ${item}`),
  ].slice(0, 4);
  return `
    <article class="auto-handoff-transition">
      <div>
        <strong>${escapeHtml(transitionTypeLabel(transition.type))}</strong>
        <span>${escapeHtml(transition.fromName || "A")} -> ${escapeHtml(transition.toName || "B")}</span>
      </div>
      <div class="auto-handoff-metrics">
        <span>${Math.round(transition.score || 0)}/100</span>
        <span>${transition.barCount || 0} bars</span>
        <span>${formatTime(Number(transition.durationSec) || 0)}</span>
        <span>A OUT ${formatTime(Number(transition.outgoingCue?.time) || 0)}</span>
        <span>B IN ${formatTime(Number(transition.incomingCue?.time) || 0)}</span>
        <span>Bed ${escapeHtml(bed.source || "--")}/${escapeHtml(bed.stem || "--")}</span>
        <span>Bass ${Math.round((automation.bassSwapAt || 0) * 100)}%</span>
        <span>Risk ${escapeHtml(transition.risk || "--")}</span>
        ${transition.renderedPreview?.url ? `<span>Rendered ${escapeHtml(transition.renderedPreview.renderMethod || transition.renderedPreview.method || "audio")}</span>` : ""}
      </div>
      <p>${escapeHtml(transition.explanation || "")}</p>
      ${cueReasons.length ? `<small>${escapeHtml(cueReasons.join(" / "))}</small>` : ""}
      ${warnings.length ? `<small>${escapeHtml(warnings.join(" / "))}</small>` : ""}
    </article>
  `;
}

function transitionTypeLabel(type) {
  return {
    drum_bed_handoff: "Drum Bed Handoff",
    bass_swap_handoff: "Bass Swap Handoff",
    percussive_loop_bridge: "Percussive Loop Bridge",
    vocal_safe_bridge: "Vocal Safe Bridge",
    effect_tail_handoff: "Effect Tail Handoff",
  }[type] || type || "Handoff";
}

function syncMashupSettings() {
  state.mashup.trackAId = els.mashupTrackA.value || null;
  state.mashup.trackBId = els.mashupTrackB.value || null;
  state.mashup.mode = els.mashupMode.value || "auto";
  state.mashup.barsPerSegment = Number(els.mashupBars.value) || 16;
  state.mashup.useStems = els.mashupUseStems.checked;
  state.mashup.vocalPriority = els.mashupVocalPriority.value || "auto";
  state.mashup.bedPreference = els.mashupBedPreference.value || "auto";
  state.mashup.allowHybridBed = els.mashupAllowHybridBed.checked;
  state.mashup.allowVocalPitchShift = els.mashupAllowVocalPitchShift.checked;
  state.mashup.maxVocalStretch = Number(els.mashupMaxVocalStretch.value) || 1.06;
  state.mashup.error = "";
  renderMashupPanel();
}

async function analyzeMashupSegments() {
  const ready = playableTracks();
  if (ready.length < 2) {
    state.mashup.error = "请先上传并完成分析至少两首歌。";
    renderMashupPanel();
    return;
  }
  ensureMashupSelection();
  state.mashup.analyzing = true;
  state.mashup.error = "";
  state.mashup.analysis = null;
  state.mashup.plan = null;
  state.mashup.renderResult = null;
  renderMashupPanel();
  try {
    await assertBackendReachable();
    state.mashup.analysis = await fetchJson("/api/mashup/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(mashupRequestBase()),
    });
    setStatus("Mashup 段落分析完成");
  } catch (error) {
    state.mashup.error = error.message || "Mashup 段落分析失败";
  } finally {
    state.mashup.analyzing = false;
    renderMashupPanel();
  }
}

async function generateMashupPlan() {
  const ready = playableTracks();
  if (ready.length < 2) {
    state.mashup.error = "请先上传并完成分析至少两首歌。";
    renderMashupPanel();
    return;
  }
  ensureMashupSelection();
  state.mashup.planning = true;
  state.mashup.error = "";
  state.mashup.plan = null;
  state.mashup.renderResult = null;
  renderMashupPanel();
  try {
    await assertBackendReachable();
    state.mashup.plan = await fetchJson("/api/mashup/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...mashupRequestBase(),
        mode: state.mashup.mode,
        targetDurationSec: 180,
        vocalPriority: state.mashup.vocalPriority,
        bedPreference: state.mashup.bedPreference,
        allowHybridBed: state.mashup.allowHybridBed,
        allowVocalPitchShift: state.mashup.allowVocalPitchShift,
        maxVocalStretch: state.mashup.maxVocalStretch,
        returnAlternatives: true,
      }),
    });
    setStatus(`Mashup 方案已生成：${state.mashup.plan.score}/100`);
  } catch (error) {
    state.mashup.error = error.message || "Mashup 方案生成失败";
  } finally {
    state.mashup.planning = false;
    renderMashupPanel();
  }
}

async function renderMashupExport() {
  if (!state.mashup.plan?.plan?.length) {
    state.mashup.error = "请先生成拼接方案。";
    renderMashupPanel();
    return;
  }
  state.mashup.rendering = true;
  state.mashup.error = "";
  state.mashup.renderResult = null;
  renderMashupPanel();
  try {
    await assertBackendReachable();
    state.mashup.renderResult = await fetchJson("/api/mashup/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        plan: state.mashup.plan,
        format: "wav",
        targetLufs: -14,
        useStems: state.mashup.useStems,
      }),
    });
    setStatus("Mashup 渲染完成");
  } catch (error) {
    state.mashup.error = error.message || "Mashup 渲染失败";
  } finally {
    state.mashup.rendering = false;
    renderMashupPanel();
  }
}

function mashupRequestBase() {
  return {
    trackAId: state.mashup.trackAId,
    trackBId: state.mashup.trackBId,
    barsPerSegment: state.mashup.barsPerSegment,
    useStems: state.mashup.useStems,
  };
}

function mashupTimelineClick(event) {
  const button = event.target.closest?.("[data-mashup-alt]");
  if (!button) return;
  const index = Number(button.dataset.mashupAlt);
  const alternative = state.mashup.plan?.alternativePlans?.[index];
  if (!alternative) return;
  state.mashup.plan = {
    ...state.mashup.plan,
    ...alternative,
    alternativePlans: state.mashup.plan.alternativePlans,
  };
  state.mashup.renderResult = null;
  renderMashupPanel();
}

function ensureMashupSelection() {
  const ready = playableTracks();
  const ids = ready.map((track) => track.id);
  if (!ids.includes(state.mashup.trackAId)) state.mashup.trackAId = ids[0] || null;
  if (!ids.includes(state.mashup.trackBId) || state.mashup.trackBId === state.mashup.trackAId) {
    state.mashup.trackBId = ids.find((id) => id !== state.mashup.trackAId) || null;
  }
}

function renderMashupPanel() {
  if (!els.mashupTrackA) return;
  const ready = playableTracks();
  ensureMashupSelection();
  const options = ready.length
    ? ready.map((track) => `<option value="${track.id}">${escapeHtml(track.name)}</option>`).join("")
    : `<option value="">需要已分析曲目</option>`;
  els.mashupTrackA.innerHTML = options;
  els.mashupTrackB.innerHTML = options;
  els.mashupTrackA.value = state.mashup.trackAId || "";
  els.mashupTrackB.value = state.mashup.trackBId || "";
  els.mashupMode.value = state.mashup.mode;
  els.mashupBars.value = String(state.mashup.barsPerSegment);
  els.mashupUseStems.checked = state.mashup.useStems;
  els.mashupVocalPriority.value = state.mashup.vocalPriority;
  els.mashupBedPreference.value = state.mashup.bedPreference;
  els.mashupAllowHybridBed.checked = state.mashup.allowHybridBed;
  els.mashupAllowVocalPitchShift.checked = state.mashup.allowVocalPitchShift;
  els.mashupMaxVocalStretch.value = String(state.mashup.maxVocalStretch);

  const validPair = Boolean(state.mashup.trackAId && state.mashup.trackBId && state.mashup.trackAId !== state.mashup.trackBId);
  els.mashupAnalyzeButton.disabled = !validPair || state.mashup.analyzing;
  els.mashupPlanButton.disabled = !validPair || state.mashup.planning;
  els.mashupRenderButton.disabled = !state.mashup.plan?.plan?.length || state.mashup.rendering;
  els.mashupAnalyzeButton.textContent = state.mashup.analyzing ? "分析中..." : "分析段落";
  els.mashupPlanButton.textContent = state.mashup.planning ? "生成中..." : "生成拼接方案";
  els.mashupRenderButton.textContent = state.mashup.rendering ? "渲染中..." : "渲染试听/导出";

  renderMashupFlow();
  renderMashupSegments();
  renderMashupTimeline();
  renderMashupResult();
}

function renderMashupFlow() {
  if (!els.mashupFlow) return;
  const hasPair = Boolean(state.mashup.trackAId && state.mashup.trackBId && state.mashup.trackAId !== state.mashup.trackBId);
  const steps = [
    { label: "1. Pick songs", detail: hasPair ? "Song A/B ready" : "Choose two analyzed tracks", state: hasPair ? "done" : "active" },
    { label: "2. Analyze", detail: state.mashup.analysis ? "Sections, phrases, beds ready" : "Find sections and vocal phrases", state: state.mashup.analysis ? "done" : hasPair ? "active" : "pending" },
    { label: "3. Build plan", detail: state.mashup.plan ? `${Math.round(state.mashup.plan.score || 0)}/100` : "Choose groove bed and handoff", state: state.mashup.plan ? "done" : state.mashup.analysis ? "active" : "pending" },
    { label: "4. Render", detail: state.mashup.renderResult ? "Preview and download ready" : "Export WAV preview", state: state.mashup.renderResult ? "done" : state.mashup.plan ? "active" : "pending" },
  ];
  els.mashupFlow.innerHTML = steps.map((step) => `
    <div class="mashup-flow-step ${step.state}">
      <strong>${escapeHtml(step.label)}</strong>
      <span>${escapeHtml(step.detail)}</span>
    </div>
  `).join("");
}

function renderMashupSegments() {
  if (state.mashup.error) {
    els.mashupSegments.innerHTML = `<div class="mashup-error">${escapeHtml(state.mashup.error)}</div>`;
    return;
  }
  const analysis = state.mashup.analysis;
  if (!analysis) {
    els.mashupSegments.innerHTML = `<div class="mashup-empty">选择两首已上传歌曲后，先分析 8/16 小节段落。</div>`;
    return;
  }
  els.mashupSegments.innerHTML = `
    ${renderMashupSegmentColumn("Song A", analysis.trackA?.segments || [])}
    ${renderMashupSegmentColumn("Song B", analysis.trackB?.segments || [])}
    ${renderSegmentationDebug(analysis.segmentationReport)}
  `;
}

function renderSegmentationDebug(report) {
  if (!report || (!report.trackA && !report.trackB)) return "";
  return `
    <section class="mashup-seg-debug">
      <header><strong>Segmentation Debug</strong><span>multi-scale SSM / novelty / stems</span></header>
      ${renderSegmentationTrackDebug("Song A", report.trackA)}
      ${renderSegmentationTrackDebug("Song B", report.trackB)}
    </section>
  `;
}

function renderSegmentationTrackDebug(title, report) {
  if (!report) return "";
  const sections = report.sections || [];
  const minorSections = report.minorSections || [];
  const phrases = report.vocalPhrases || [];
  const beds = report.grooveBedCandidates || [];
  const safe = report.safeCutPoints || [];
  const warnings = report.warnings || [];
  return `
    <article class="mashup-seg-track">
      <div class="mashup-plan-head">
        <strong>${escapeHtml(title)}</strong>
        <small>${sections.length} major / ${minorSections.length} minor / ${phrases.length} vocal phrases / ${beds.length} beds</small>
        <span>${escapeHtml(report.method || "segmentation")} · ${sections.length} sections · ${phrases.length} vocal phrases · ${beds.length} beds</span>
      </div>
      ${warnings.length ? `<div class="mashup-warnings">${warnings.map((warning) => `<span>${escapeHtml(warning)}</span>`).join("")}</div>` : ""}
      <div class="mashup-quality-report">
        <div><strong>Minor sections</strong>${minorSections.slice(0, 8).map((item) => `<span>${escapeHtml(item.sectionSubLabel || item.sectionLabel || item.label || "unknown")} / b${item.barStart}-${item.barEnd} / ${item.bars} bars / ${escapeHtml(item.arrangementLevel || "layer")} / ${escapeHtml((item.riskFlags || []).join(", ") || "clean")}</span>`).join("")}</div>
        <div><strong>Structural sections</strong>${sections.slice(0, 8).map((item) => `<span>${escapeHtml(item.sectionSubLabel || item.sectionLabel || item.label || "unknown")} · b${item.barStart}-${item.barEnd} · ${Math.round((Number(item.labelConfidence || item.confidence) || 0) * 100)}% · ${escapeHtml((item.labelReasons || item.riskFlags || []).join(", ") || "clean")}</span>`).join("")}</div>
        <div><strong>Vocal phrases</strong>${phrases.slice(0, 8).map((item) => `<span>${escapeHtml(item.id || "phrase")} · ${item.bars} bars · score ${Math.round(item.score || 0)} · ${item.hasPickup ? "pickup " : ""}${item.hasTail ? "tail " : ""}${escapeHtml((item.riskFlags || []).join(", "))}</span>`).join("")}</div>
        <div><strong>Groove beds</strong>${beds.slice(0, 5).map((item) => `<span>${escapeHtml(item.id || "bed")} · ${item.bars} bars · loop ${Math.round((Number(item.loopability) || 0) * 100)} · leak ${Math.round((Number(item.vocalLeakage) || 0) * 100)} · ${Math.round(item.score || 0)}/100</span>`).join("")}</div>
        <div><strong>Safe cut points</strong>${safe.slice(0, 6).map((item) => `<span>${formatTime(item.time)} · ${escapeHtml(item.type)} · ${Math.round((Number(item.score) || 0) * 100)} · ${escapeHtml((item.riskFlags || []).join(", ") || "safe")}</span>`).join("")}</div>
      </div>
    </article>
  `;
}

function renderMashupSegmentColumn(title, segments) {
  return `
    <section class="mashup-segment-column">
      <header><strong>${title}</strong><span>${segments.length} segments</span></header>
      <div class="mashup-segment-list">
        ${segments.map(renderMashupSegmentBlock).join("") || `<div class="mashup-empty">暂无段落</div>`}
      </div>
    </section>
  `;
}

function renderMashupSegmentBlock(segment) {
  const energy = Math.round((Number(segment.energy) || 0) * 100);
  const vocal = Math.round((Number(segment.vocalDensity) || 0) * 100);
  const bass = Math.round((Number(segment.bassEnergy) || 0) * 100);
  const cleanEntry = segment.isCleanEntry ? "clean in" : "risky in";
  const cleanExit = segment.isCleanExit ? "clean out" : "risky out";
  const risks = (segment.riskFlags || []).slice(0, 3);
  const title = segment.sectionSubLabel || segment.sectionLabel || segment.label || "segment";
  const raw = segment.rawLabel && segment.rawLabel !== segment.label ? ` · ${segment.rawLabel}` : "";
  const level = segment.arrangementLevel ? ` · ${segment.arrangementLevel}` : "";
  const reasons = (segment.labelReasons || []).slice(0, 2);
  return `
    <article class="mashup-segment-block ${escapeHtml(segment.source || "")}">
      <div><strong>${escapeHtml(title)}</strong><span>${formatTime(segment.start)}-${formatTime(segment.end)}</span></div>
      <small>${escapeHtml(segment.sectionLabel || segment.label || "segment")}${escapeHtml(level)}${escapeHtml(raw)}</small>
      <div class="mashup-mini-meters">
        <span style="--value:${energy}%">E ${energy}</span>
        <span style="--value:${vocal}%">V ${vocal}</span>
        <span style="--value:${bass}%">B ${bass}</span>
      </div>
      <small>${escapeHtml(segment.camelot || "--")} · ${formatNumber(segment.bpm, 1)} BPM · ${cleanEntry} / ${cleanExit}</small>
      ${reasons.length ? `<div class="mashup-flags">${reasons.map((reason) => `<span>${escapeHtml(reason)}</span>`).join("")}</div>` : ""}
      ${risks.length ? `<div class="mashup-flags">${risks.map((risk) => `<span>${escapeHtml(risk)}</span>`).join("")}</div>` : ""}
    </article>
  `;
}

function renderMashupTimeline() {
  const result = state.mashup.plan;
  if (result?.groovePlan && !result?.plan?.length) {
    const warnings = result.warnings || result.groovePlan.globalWarnings || [];
    els.mashupTimeline.innerHTML = `
      <div class="mashup-plan-head">
        <strong>Groove 人声接力不可用</strong>
        <span>${escapeHtml(result.groovePlan.status || "no_plan")}</span>
      </div>
      <div class="mashup-warnings">${warnings.map((warning) => `<span>${escapeHtml(warning)}</span>`).join("")}</div>
    `;
    return;
  }
  if (!result?.plan?.length) {
    els.mashupTimeline.innerHTML = `<div class="mashup-empty">生成方案后，这里会显示新的拼接时间线。</div>`;
    return;
  }
  if (result.groovePlan?.bed) {
    els.mashupTimeline.innerHTML = renderGrooveMashupTimeline(result);
    return;
  }
  const plan = result.plan;
  const total = Math.max(1, ...plan.map((item) => Number(item.timelineEnd) || 0));
  const report = result.qualityReport || {};
  const transitions = result.transitions || [];
  els.mashupTimeline.innerHTML = `
    <div class="mashup-plan-head">
      <strong>Score ${result.score}/100</strong>
      <span>${escapeHtml(result.mode || state.mashup.mode)} · ${formatTime(total)} · ${formatNumber(result.targetBpm, 1)} BPM</span>
    </div>
    ${report.summary ? `<p class="mashup-summary">${escapeHtml(report.summary)}</p>` : ""}
    <div class="mashup-plan-stage">
      ${plan.map((item) => renderMashupPlanItem(item, total)).join("")}
    </div>
    ${renderMashupQualityReport(report)}
    ${renderMashupTransitionDetails(transitions)}
    ${renderMashupAlternatives(result.alternativePlans || [])}
    ${(result.warnings || []).length ? `<div class="mashup-warnings">${result.warnings.map((warning) => `<span>${escapeHtml(warning)}</span>`).join("")}</div>` : ""}
  `;
}

function renderGrooveMashupTimeline(result) {
  const groove = result.groovePlan || {};
  const bed = groove.bed || {};
  const events = groove.vocalEvents || [];
  const total = Math.max(1, ...events.map((event) => Number(event.timelineEnd) || 0));
  const report = groove.qualityReport || result.qualityReport || {};
  return `
    <div class="mashup-plan-head">
      <strong>Groove 人声接力 · Score ${Math.round(result.score || report.score || 0)}/100</strong>
      <span>${formatNumber(groove.targetBpm || result.targetBpm, 1)} BPM · ${escapeHtml(groove.targetCamelot || result.targetCamelot || "--")}</span>
    </div>
    <div class="mashup-quality-report">
      <div><strong>GrooveBed</strong>
        <span>drums ${escapeHtml(bed.drumsSource || "--")} · bass ${escapeHtml(bed.bassSource || "--")} · other ${escapeHtml(bed.otherSource || "--")}</span>
        <span>loopability ${Math.round((Number(bed.loopability) || 0) * 100)} · vocal leakage ${Math.round((Number(bed.vocalLeakage) || 0) * 100)}</span>
      </div>
      <div><strong>Vocal handoff</strong>
        ${events.slice(0, 8).map((event) => `<span>${escapeHtml(event.source)} ${escapeHtml(event.phraseId)} · x${formatNumber(event.stretchRatio, 3)} · pitch ${formatNumber(event.pitchShiftSemitones, 1)} · ${escapeHtml(event.tailTreatment || "natural")}</span>`).join("")}
      </div>
    </div>
    ${report.summary ? `<p class="mashup-summary">${escapeHtml(report.summary)}</p>` : ""}
    <div class="mashup-plan-stage">
      ${events.map((event) => renderGrooveVocalEvent(event, total)).join("")}
    </div>
    ${renderMashupQualityReport(report)}
    ${renderMashupAlternatives(result.alternativePlans || [])}
    ${(result.warnings || groove.globalWarnings || []).length ? `<div class="mashup-warnings">${(result.warnings || groove.globalWarnings || []).map((warning) => `<span>${escapeHtml(warning)}</span>`).join("")}</div>` : ""}
  `;
}

function renderGrooveVocalEvent(event, total) {
  const left = clamp(((Number(event.timelineStart) || 0) / total) * 100, 0, 100);
  const width = clamp((((Number(event.timelineEnd) || 0) - (Number(event.timelineStart) || 0)) / total) * 100, 1, 100);
  const lane = event.source === "B" ? 1 : 0;
  const warnings = (event.warnings || []).slice(0, 2);
  return `
    <div class="mashup-plan-item lane-${lane} ${escapeHtml(event.source || "")}" style="left:${left}%;width:${width}%">
      <strong>${escapeHtml(event.source)} vocal phrase</strong>
      <span>${escapeHtml(event.handoffToNext || "handoff")} · duck ${formatNumber(event.duckBedDb, 1)} dB</span>
      ${warnings.length ? `<small>${warnings.map(escapeHtml).join(" · ")}</small>` : `<small>${formatTime(event.sourceStart)}-${formatTime(event.sourceEnd)}</small>`}
    </div>
  `;
}

function renderMashupPlanItem(item, total) {
  const left = clamp(((Number(item.timelineStart) || 0) / total) * 100, 0, 100);
  const width = clamp((((Number(item.timelineEnd) || 0) - (Number(item.timelineStart) || 0)) / total) * 100, 1, 100);
  const lane = item.layerMode === "vocals" ? 1 : item.layerMode === "instrumental" || item.layerMode === "drums_bass_other" ? 2 : item.source === "B" ? 1 : 0;
  const transition = transitionName(item.transitionIn);
  const score = Math.round(Number(item.quality?.score) || 0);
  const warnings = (item.quality?.warnings || []).slice(0, 2);
  return `
    <div class="mashup-plan-item lane-${lane} ${escapeHtml(item.source || "")}" style="left:${left}%;width:${width}%">
      <strong>${escapeHtml(item.source)} · ${escapeHtml(item.segmentLabel || item.layerMode)}</strong>
      <span>${escapeHtml(item.layerMode)} · ${escapeHtml(transition)} · ${score}/100</span>
      ${warnings.length ? `<small>${warnings.map(escapeHtml).join(" · ")}</small>` : `<small>${formatTime(item.sourceStart)}-${formatTime(item.sourceEnd)}</small>`}
    </div>
  `;
}

function renderMashupQualityReport(report) {
  if (!report || (!report.strengths?.length && !report.warnings?.length && !report.transitionReports?.length)) return "";
  return `
    <div class="mashup-quality-report">
      ${report.strengths?.length ? `<div><strong>Strengths</strong>${report.strengths.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>` : ""}
      ${report.warnings?.length ? `<div><strong>Warnings</strong>${report.warnings.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>` : ""}
      ${report.transitionReports?.length ? `<div><strong>Transitions</strong>${report.transitionReports
        .slice(0, 4)
        .map((item) => `<span>${escapeHtml(item.from)} -> ${escapeHtml(item.to)} · ${escapeHtml(item.type)} · ${Math.round(item.score || 0)}/100</span>`)
        .join("")}</div>` : ""}
    </div>
  `;
}

function renderMashupTransitionDetails(transitions) {
  if (!transitions.length) return "";
  return `
    <div class="mashup-quality-report">
      <div><strong>Layered transitions</strong>${transitions
        .slice(0, 5)
        .map((item) => `<span>${escapeHtml(item.type)} · ${escapeHtml(item.reason || "")}${item.fallbackType ? ` · fallback ${escapeHtml(item.fallbackType)}` : ""}</span>`)
        .join("")}</div>
    </div>
  `;
}

function renderMashupAlternatives(alternatives) {
  if (!alternatives.length) return "";
  return `
    <div class="mashup-alternatives">
      <strong>Alternative plans</strong>
      ${alternatives.map((plan, index) => `<button type="button" data-mashup-alt="${index}">${escapeHtml(plan.mode || "alt")} · ${Math.round(plan.score || 0)}/100</button>`).join("")}
    </div>
  `;
}

function transitionName(transition) {
  if (!transition) return "none";
  if (typeof transition === "string") return transition;
  return transition.type || "none";
}

function renderMashupResult() {
  const result = state.mashup.renderResult;
  if (!result) {
    els.mashupResult.innerHTML = "";
    return;
  }
  const url = `${API}${result.downloadUrl}`;
  const report = result.report || {};
  const warnings = report.warnings || [];
  const layerLines = report.layers?.slice(0, 6) || [];
  const groove = report.groovePlan || {};
  const renderDuration = report.duration || report.renderStats?.duration || 0;
  els.mashupResult.innerHTML = `
    <div class="mashup-rendered">
      <strong>已渲染 ${escapeHtml(result.report?.filename || "mashup.wav")}</strong>
      <audio controls preload="none" src="${url}"></audio>
      <a href="${url}" target="_blank" rel="noreferrer">下载 WAV</a>
      <span>${formatNumber(report.finalLufs, 1)} LUFS · peak ${formatNumber(report.peak, 3)} · ${formatTime(renderDuration)}</span>
    </div>
    ${groove.bed ? `<div class="mashup-quality-report"><div><strong>Rendered groove</strong><span>bed ${escapeHtml(groove.bed.drumsSource || "--")}/${escapeHtml(groove.bed.bassSource || "--")}/${escapeHtml(groove.bed.otherSource || "--")} · ${groove.vocalEvents?.length || 0} vocal phrases · no full-mix crossfade bed</span></div></div>` : ""}
    ${layerLines.length ? `<div class="mashup-quality-report"><div><strong>Rendered layers</strong>${layerLines.map((layer) => `<span>${escapeHtml(layer.source)} ${escapeHtml(layer.stem)} · ${escapeHtml(layer.role)} · x${formatNumber(layer.stretchRatio, 3)} · pitch ${formatNumber(layer.pitchShiftSemitones, 1)}</span>`).join("")}</div></div>` : ""}
    ${warnings.length ? `<div class="mashup-warnings">${warnings.map((warning) => `<span>${escapeHtml(warning)}</span>`).join("")}</div>` : ""}
  `;
}

function renderStemDebugger() {
  if (!els.stemDebuggerView || state.view !== "stems") return;
  const tracks = playableTracks().filter((track) => track.buffer);
  const fallbackTracks = tracks.length ? tracks : state.tracks.filter((track) => track.buffer);
  const active = activeStemTrack();
  if (active && state.stemDebugger.trackId !== active.localId) state.stemDebugger.trackId = active.localId;
  const references = stemReferenceTracks(active);
  const selectedReference = selectedStemReferenceTrack(active);
  if (selectedReference) state.stemDebugger.referenceTrackId = selectedReference.id;

  els.stemTrackSelect.innerHTML = fallbackTracks.length
    ? fallbackTracks.map((track) => `<option value="${track.localId}" ${track.localId === state.stemDebugger.trackId ? "selected" : ""}>${escapeHtml(track.name)}</option>`).join("")
    : `<option value="">\u672a\u9009\u62e9\u97f3\u9891</option>`;
  els.stemReferenceSelect.innerHTML = references.length
    ? references.map((track) => `<option value="${track.id}" ${track.id === selectedReference?.id ? "selected" : ""}>${escapeHtml(track.name)}</option>`).join("")
    : `<option value="">\u9700\u8981\u7b2c\u4e8c\u9996\u66f2\u4f5c\u53c2\u8003</option>`;

  const hasTrack = Boolean(active?.buffer);
  els.stemPlayButton.disabled = !hasTrack;
  els.stemRestartButton.disabled = !hasTrack;
  els.stemSeparateButton.disabled = !hasTrack || isStemPending(active) || active?.status !== "ready";
  els.stemAutoMixButton.disabled = !hasTrack || !references.length || state.stemDebugger.isAutoMixing || isStemPending(active);
  els.stemStopButton.disabled = !state.stemDebugger.isPlaying;
  els.stemTransportPlay.disabled = !hasTrack;
  els.stemAutoMixButton.classList.toggle("active", state.settings.mixStyleTransfer.enabled);
  els.stemAutoMixButton.textContent = state.stemDebugger.isAutoMixing ? "\u6b63\u5728\u81ea\u52a8\u6df7\u97f3..." : "\u53c2\u8003\u66f2\u81ea\u52a8\u6df7\u97f3";
  els.stemSeparateButton.textContent = isStemPending(active) ? "Demucs ..." : trackHasRealStems(active) ? "\u91cd\u65b0\u5206\u8f68" : "Demucs \u5206\u8f68";
  els.stemPlayButton.textContent = state.stemDebugger.isPlaying ? "\u6682\u505c" : "\u64ad\u653e";
  els.stemTransportPlay.textContent = state.stemDebugger.isPlaying ? "II" : ">";
  renderStemMixResult();

  if (!hasTrack) {
    els.stemDeck.innerHTML = `
      <div class="stem-empty">
        <strong>\u672a\u9009\u62e9\u97f3\u9891</strong>
      </div>
    `;
    renderStemTransport();
    return;
  }

  els.stemDeck.innerHTML = renderStemDebuggerStatus(active) + STEMS.map((stem) => renderStemLane(stem, active)).join("");
  renderStemTransport();
  drawStemWaveforms();
}

function renderStemDebuggerStatus(track) {
  if (!track) return "";
  const ready = trackHasRealStems(track);
  const loading = isStemPending(track);
  const error = track.stemStatus === "error" ? track.stemError : "";
  const label = ready
    ? "Demucs \u771f\u5206\u8f68"
    : loading
      ? track.stemStatus === "queued" ? "Demucs \u7b49\u5f85\u5206\u8f68" : "Demucs \u5206\u8f68\u4e2d"
      : "\u6a21\u62df\u5206\u8f68";
  const note = ready
    ? "\u5f53\u524d\u76d1\u542c vocals / drums / bass / other \u56db\u4e2a\u771f\u5b9e\u97f3\u9891\u6587\u4ef6"
    : loading
      ? "\u9996\u6b21\u751f\u6210\u4f1a\u8f83\u6162\uff0c\u5b8c\u6210\u540e\u4f1a\u81ea\u52a8\u5207\u6362\u5230\u771f\u5206\u8f68"
      : "\u5f53\u524d\u7528\u5168\u66f2\u6ee4\u6ce2\u6a21\u62df\uff0c\u70b9\u51fb Demucs \u5206\u8f68\u540e\u624d\u80fd\u771f\u6b63\u5206\u8f68\u8c03\u8bd5";
  return `
    <div class="stem-mode ${ready ? "ready" : loading ? "loading" : "simulated"}">
      <strong>${label}</strong>
      <span>${escapeHtml(error || note)}${renderStemStyleStatus()}</span>
    </div>
  `;
}

function renderStemStyleStatus() {
  if (!state.settings.mixStyleTransfer.enabled) return "";
  const name = state.settings.mixStyleTransfer.referenceTrackName || "\u53c2\u8003\u66f2";
  return ` · \u53c2\u8003\u66f2\u81ea\u52a8\u6df7\u97f3: ${escapeHtml(name)}`;
}

function renderStemMixResult() {
  if (!els.stemMixResult) return;
  const result = state.settings.mixStyleTransfer.result;
  if (!result && !state.stemDebugger.isAutoMixing) {
    els.stemMixResult.hidden = true;
    els.stemMixResult.innerHTML = "";
    return;
  }
  els.stemMixResult.hidden = false;
  if (state.stemDebugger.isAutoMixing) {
    els.stemMixResult.innerHTML = `
      <div class="stem-result-head">
        <strong>\u53c2\u8003\u66f2\u81ea\u52a8\u6df7\u97f3</strong>
        <span>\u6b63\u5728\u5206\u6790\u53c2\u8003\u66f2\u7279\u5f81\u5e76\u641c\u7d22 DSP \u6df7\u97f3\u53c2\u6570...</span>
      </div>
    `;
    return;
  }
  if (result?.error) {
    els.stemMixResult.innerHTML = `
      <div class="stem-result-head">
        <strong>\u751f\u6210\u5931\u8d25</strong>
        <span>${escapeHtml(result.error)}</span>
      </div>
    `;
    return;
  }
  const summary = result?.summary || {};
  const before = Number(result?.featureDistanceBefore);
  const after = Number(result?.featureDistanceAfter);
  els.stemMixResult.innerHTML = `
    <div class="stem-result-head">
      <strong>\u53c2\u8003: ${escapeHtml(result?.referenceTrackName || state.settings.mixStyleTransfer.referenceTrackName || "")}</strong>
      <span>${Number.isFinite(after) ? `feature distance ${formatNumber(after)}${Number.isFinite(before) ? ` \u2190 ${formatNumber(before)}` : ""}` : "\u5df2\u751f\u6210 DSP \u6df7\u97f3\u53c2\u6570"}</span>
    </div>
    <div class="stem-audio-compare">
      ${result?.rawUrl ? `<label><span>\u5904\u7406\u524d</span><audio controls preload="none" src="${API}${result.rawUrl}"></audio></label>` : ""}
      ${result?.url ? `<label><span>\u81ea\u52a8\u6df7\u97f3\u540e</span><audio controls preload="none" src="${API}${result.url}"></audio></label>` : ""}
    </div>
    <div class="stem-feature-grid">
      ${renderMetricCompare("LUFS", summary.beforeLufs, summary.finalLufs, summary.referenceLufs, -30, -6)}
      ${renderMetricCompare("Crest", summary.beforeCrestDb, summary.finalCrestDb, summary.referenceCrestDb, 0, 24)}
      ${renderMetricCompare("Width", summary.beforeWidth, summary.finalWidth, summary.referenceWidth, 0, 1.5)}
    </div>
    ${renderBandEnergyCompare(result)}
    ${renderStemParamVisual(result?.mixer)}
    <div class="stem-mix-links">
      ${result?.url ? `<a href="${API}${result.url}" target="_blank" rel="noreferrer">\u6253\u5f00\u6df7\u97f3 WAV</a>` : ""}
      ${result?.reportUrl ? `<a href="${API}${result.reportUrl}" target="_blank" rel="noreferrer">mix_report.json</a>` : ""}
    </div>
  `;
}

function renderMetricCompare(label, before, after, reference, min, max) {
  const beforePct = metricPercent(before, min, max);
  const afterPct = metricPercent(after, min, max);
  const referencePct = metricPercent(reference, min, max);
  const delta = Number(after) - Number(before);
  return `
    <section class="stem-metric-card">
      <header><span>${label}</span><b>${formatSignedNumber(delta, label === "Width" ? 3 : 2)}</b></header>
      <div class="stem-metric-track">
        <i class="before" style="width:${beforePct}%"></i>
        <i class="after" style="width:${afterPct}%"></i>
        <em style="left:${referencePct}%"></em>
      </div>
      <footer><span>\u524d ${formatNumber(before, label === "Width" ? 3 : 2)}</span><span>\u540e ${formatNumber(after, label === "Width" ? 3 : 2)}</span><span>\u53c2\u8003 ${formatNumber(reference, label === "Width" ? 3 : 2)}</span></footer>
    </section>
  `;
}

function renderBandEnergyCompare(result) {
  const bands = [
    ["sub", "Sub"],
    ["bass", "Bass"],
    ["low_mid", "Low Mid"],
    ["mid", "Mid"],
    ["high_mid", "High Mid"],
    ["high", "High"],
  ];
  const before = result?.beforeFeatures?.band_energy || {};
  const after = result?.finalFeatures?.band_energy || {};
  const reference = result?.referenceFeatures?.band_energy || {};
  const maxValue = Math.max(0.04, ...bands.flatMap(([key]) => [Number(before[key]) || 0, Number(after[key]) || 0, Number(reference[key]) || 0]));
  return `
    <section class="stem-band-compare">
      <header><strong>\u9891\u6bb5\u80fd\u91cf</strong><span>\u7070=\u5904\u7406\u524d / \u9752=\u81ea\u52a8\u6df7\u97f3 / \u9ec4\u7ebf=\u53c2\u8003\u66f2</span></header>
      ${bands.map(([key, label]) => renderBandRow(label, before[key], after[key], reference[key], maxValue)).join("")}
    </section>
  `;
}

function renderBandRow(label, before, after, reference, maxValue) {
  const beforePct = metricPercent(before, 0, maxValue);
  const afterPct = metricPercent(after, 0, maxValue);
  const referencePct = metricPercent(reference, 0, maxValue);
  return `
    <div class="stem-band-row">
      <span>${label}</span>
      <div class="stem-band-track">
        <i class="before" style="width:${beforePct}%"></i>
        <i class="after" style="width:${afterPct}%"></i>
        <em style="left:${referencePct}%"></em>
      </div>
    </div>
  `;
}

function renderStemParamVisual(mixer) {
  const stems = mixer?.stems || {};
  if (!Object.keys(stems).length) return "";
  return `
    <section class="stem-param-viz">
      <header><strong>DSP \u53c2\u6570\u53d8\u5316</strong><span>gain / pan / EQ / reverb send</span></header>
      ${STEMS.map((stem) => {
        const params = stems[stem.id] || {};
        const gainDb = Number(params.gainDb) || 0;
        const pan = Number.isFinite(params.pan) ? params.pan : 0.5;
        const eq = params.eqDb || {};
        return `
          <div class="stem-param-row">
            <span>${stem.label}</span>
            <div class="stem-gain-axis"><i style="left:${metricPercent(gainDb, -12, 12)}%"></i><b>${formatSignedNumber(gainDb, 1)} dB</b></div>
            <div class="stem-pan-axis"><i style="left:${metricPercent(pan, 0, 1)}%"></i><b>${panLabel(pan)}</b></div>
            <small>L ${formatSignedNumber(eq.low, 1)} / M ${formatSignedNumber(eq.mid, 1)} / H ${formatSignedNumber(eq.high, 1)} / R ${formatNumber(params.reverbSend, 2)}</small>
          </div>
        `;
      }).join("")}
    </section>
  `;
}

function metricPercent(value, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number) || max <= min) return 0;
  return clamp(((number - min) / (max - min)) * 100, 0, 100);
}

function panLabel(value) {
  const pan = Number(value);
  if (!Number.isFinite(pan) || Math.abs(pan - 0.5) < 0.03) return "C";
  return pan < 0.5 ? `L${Math.round((0.5 - pan) * 200)}` : `R${Math.round((pan - 0.5) * 200)}`;
}

function renderStemLane(stem, track) {
  const control = ensureStemControl(stem.id);
  const isReal = Boolean(track?.stems?.[stem.id]?.buffer);
  const pending = !isReal && isStemPending(track);
  const simulated = !isReal && !pending;
  const activeClass = `${control.mute ? " muted" : control.solo ? " solo" : ""}${isReal ? " real" : ""}${pending ? " pending" : ""}${simulated ? " simulated" : ""}`;
  return `
    <section class="stem-lane${activeClass}" style="--stem-color:${isReal ? stem.color : "#737b80"}">
      <div class="stem-control-strip">
        <div class="stem-buttons">
          <button type="button" class="${control.mute ? "active" : ""}" data-stem-action="mute" data-stem="${stem.id}" aria-label="${stem.label} mute">M</button>
          <button type="button" class="${control.solo ? "active" : ""}" data-stem-action="solo" data-stem="${stem.id}" aria-label="${stem.label} solo">S</button>
          <strong>${stem.label}</strong>
          <small>${isReal ? "Demucs" : "\u6a21\u62df"}</small>
        </div>
        <label class="stem-volume">
          <input type="range" min="0" max="1.5" value="${control.gain}" step="0.01" data-stem-volume="${stem.id}" />
          <span data-stem-volume-readout>${Math.round(control.gain * 100)}%</span>
        </label>
      </div>
      <canvas class="stem-wave" width="1400" height="82" data-stem-wave="${stem.id}"></canvas>
    </section>
  `;
}

function renderStemTransport() {
  const track = activeStemTrack();
  const duration = track?.duration || 0;
  const offset = clamp(state.stemDebugger.playbackOffset, 0, duration);
  els.stemProgress.max = duration;
  els.stemProgress.value = offset;
  els.stemTime.textContent = `${formatTime(offset)} / ${formatTime(duration)}`;
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

function renderMixTimeline() {
  const timeline = buildTimeline();
  const total = Math.max(timeline.total, 1);
  const transitions = timeline.items.slice(1);
  els.transitionReadout.textContent = transitions.length
    ? `${transitions.length} 个重叠过渡 · 当前 ${formatActiveTransitionLabel(timeline)}`
    : "上传两首以上歌曲后显示过渡";
  if (!timeline.items.length) {
    els.mixTimeline.innerHTML = `<div class="timeline-empty">还没有可预览的混音时间线</div>`;
    return;
  }
  const playhead = clamp((state.playbackOffset / total) * 100, 0, 100);
  els.mixTimeline.innerHTML = `
    <div class="timeline-stage">
      <div class="timeline-playhead" style="left:${playhead}%"></div>
      ${timeline.items
        .map((item) => {
          const left = (item.start / total) * 100;
          const width = Math.max(1.5, ((item.end - item.start) / total) * 100);
          const fadeInWidth = item.fadeIn ? Math.max(1, (item.fadeIn / total) * 100) : 0;
          const fadeOutLeft = item.fadeOutStart == null ? 0 : ((item.fadeOutStart - item.start) / Math.max(item.end - item.start, 1)) * 100;
          const fadeOutWidth = item.fadeOut ? Math.max(1, (item.fadeOut / (item.end - item.start)) * 100) : 0;
          return `
            <button class="timeline-clip lane-${item.lane}" type="button" data-time="${item.start}" style="left:${left}%;width:${width}%">
              <span>${escapeHtml(item.track.name)}</span>
              ${fadeInWidth ? `<i class="fade-in" style="width:${fadeInWidth / Math.max(width, 1) * 100}%"></i>` : ""}
              ${fadeOutWidth ? `<i class="fade-out" style="left:${fadeOutLeft}%;width:${fadeOutWidth}%"></i>` : ""}
            </button>
          `;
        })
        .join("")}
    </div>
  `;
}

function syncMixTimelinePlayback(timeline = buildTimeline()) {
  const playhead = els.mixTimeline.querySelector(".timeline-playhead");
  if (!playhead) return;
  const total = Math.max(timeline.total, 1);
  const left = clamp((state.playbackOffset / total) * 100, 0, 100);
  playhead.style.left = `${left}%`;
  if (timeline.items.length > 1) {
    els.transitionReadout.textContent = `${timeline.items.length - 1} 个重叠过渡 · 当前 ${formatActiveTransitionLabel(timeline)}`;
  }
}

function formatActiveTransitionLabel(timeline) {
  const transition = activeTransition(timeline);
  if (!transition) return "--";
  return `${transition.prev.track.name} → ${transition.next.track.name}`;
}

function renderDeckMixer() {
  const timeline = buildTimeline();
  const transition = activeTransition(timeline);
  els.jumpToTransition.disabled = !transition;
  if (!transition) {
    els.deckMixer.innerHTML = `<div class="deck-empty">需要至少两首已分析曲目。时间线出现重叠后，这里会显示两台 Deck 的独立音量和 EQ。</div>`;
    return;
  }
  const strategy = resolveMixStrategy(transition.prev.track, transition.next.track);
  els.deckMixer.innerHTML = `
    <div class="transition-explain">
      <div>
        <span class="tiny-label">AI Mix Decision</span>
        <strong>${escapeHtml(strategyLabel(strategy))}</strong>
      </div>
      <div class="decision-list">
        ${strategyActions(strategy).map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
      </div>
      <div class="decision-metrics">
        <span>Overlap ${formatTime(transition.plan?.seconds || transition.next.fadeIn)}</span>
        <span>A OUT ${formatTime(transition.prev.track.outroPoint)}</span>
        <span>B IN ${formatTime(transition.next.track.introPoint)}</span>
        <span>Confidence ${Math.round((transition.plan?.confidence || 0) * 100)}%</span>
      </div>
    </div>
    ${[renderDeck("A", transition.prev), renderDeck("B", transition.next)].join("")}
  `;
}

function renderDeck(label, item) {
  const track = item.track;
  const mixer = ensureTrackMixer(track);
  const role = label === "A" ? "Outgoing" : "Incoming";
  return `
    <article class="deck-card">
      <div class="deck-title">
        <span>Deck ${label} · ${role}</span>
        <strong title="${escapeHtml(track.name)}">${escapeHtml(track.name)}</strong>
      </div>
      <div class="deck-stats">
        <span>${track.bpm || "--"} BPM</span>
        <span>${track.camelot || track.key || "--"}</span>
        <span>${formatTime(item.start)} → ${formatTime(item.end)}</span>
      </div>
      ${renderMixerSlider(track.localId, "gain", "Gain", mixer.gain, 0, 1.4, 0.01, `${Math.round(mixer.gain * 100)}%`)}
      ${renderMixerSlider(track.localId, "low", "Low", mixer.eq.low, -1, 1, 0.01, `${Math.round(mixer.eq.low * 12)} dB`)}
      ${renderMixerSlider(track.localId, "mid", "Mid", mixer.eq.mid, -1, 1, 0.01, `${Math.round(mixer.eq.mid * 12)} dB`)}
      ${renderMixerSlider(track.localId, "high", "High", mixer.eq.high, -1, 1, 0.01, `${Math.round(mixer.eq.high * 12)} dB`)}
      <div class="deck-cues">
        <button type="button" data-action="select" data-id="${track.localId}">编辑波形</button>
        <span>IN ${formatTime(track.introPoint)} / OUT ${formatTime(track.outroPoint)}</span>
      </div>
    </article>
  `;
}

function renderTeachingPanel() {
  els.teachingPanel.hidden = !state.teaching.open;
  els.teachingToggle.classList.toggle("active", state.teaching.open);
  if (!state.teaching.open) return;
  const tracks = playableTracks();
  const current = selectedTrack()?.status === "ready" ? selectedTrack() : tracks[0];
  if (tracks.length < 2 || !current) {
    els.teachingContent.innerHTML = `
      <div class="teaching-empty">
        <strong>上传并分析至少两首歌</strong>
        <span>教学入口会基于当前选中的 A 歌，推荐最适合接进来的 B 歌和具体操作。</span>
      </div>
    `;
    return;
  }

  const currentAnalysis = toTeachingAnalysis(current);
  const candidateAnalyses = tracks.filter((track) => track.localId !== current.localId).map(toTeachingAnalysis);
  const recommendations = recommendNextTracks(currentAnalysis, candidateAnalyses, {
    targetEnergy: state.teaching.targetEnergy,
    beginnerMode: state.teaching.beginnerMode,
    maxComplexity: state.teaching.maxComplexity,
    maxResults: 4,
  });

  els.teachingContent.innerHTML = `
    <div class="teaching-current">
      <div>
        <span class="tiny-label">Current Deck A</span>
        <strong>${escapeHtml(current.name)}</strong>
      </div>
      <div class="teaching-current-meta">
        <span>${current.bpm || "--"} BPM</span>
        <span>${escapeHtml(current.camelot || current.key || "--")}</span>
        <span>OUT ${formatTime(current.outroPoint)}</span>
      </div>
    </div>
    <div class="teaching-grid">
      ${recommendations.map((item, index) => renderTeachingCard(item, index)).join("")}
    </div>
  `;
}

function renderTeachingCard(item, index) {
  const target = state.tracks.find((track) => track.localId === item.track.id);
  const rec = item.bestTransition;
  const explanation = explainTransition(rec);
  const preview = teachingPreviewFor(item.track.id);
  const effective = effectiveTeachingCue(rec, preview);
  const isLoading = state.teaching.loadingPreviewId === item.track.id;
  return `
    <article class="teaching-card">
      <div class="teaching-card-head">
        <div>
          <span class="tiny-label">Recommendation ${index + 1}</span>
          <strong>${escapeHtml(methodLabel(rec.method))}</strong>
        </div>
        <b>${Math.round(item.totalScore * 100)}</b>
      </div>
      <div class="teaching-pair">
        <span class="${effective.adjusted ? "cue-adjusted" : ""}">A OUT ${formatTime(effective.outgoingTime)}</span>
        <span class="${effective.adjusted ? "cue-adjusted" : ""}">B IN ${formatTime(effective.incomingTime)}</span>
        <span>Overlap ${formatTime(effective.overlapDuration)}</span>
        <span>难度 ${rec.difficulty}/5</span>
      </div>
      ${effective.adjusted ? renderCueAdjustmentNote(rec, effective) : ""}
      <h3 title="${escapeHtml(target?.name || item.track.title)}">${escapeHtml(target?.name || item.track.title)}</h3>
      ${renderTeachingDebug(rec.debug)}
      <p class="teaching-reason">${escapeHtml(explanation)}</p>
      <div class="teaching-steps">
        ${rec.stepByStep.slice(0, 5).map(renderTeachingStep).join("")}
      </div>
      <div class="teaching-risk">${escapeHtml(riskLine(rec.method))}</div>
      ${preview ? renderPreviewResult(preview) : ""}
      <div class="teaching-card-actions">
        <button type="button" data-teaching-preview="${escapeHtml(item.track.id)}" ${isLoading ? "disabled" : ""}>${isLoading ? "生成中..." : "生成无缝试听"}</button>
        <button type="button" data-teaching-apply="${escapeHtml(item.track.id)}">使用这个接法</button>
      </div>
    </article>
  `;
}

function renderPreviewResult(preview) {
  const report = preview.processingReport || {};
  const bands = report.incomingBandTrimDb || {};
  const bandSummary = `L${Math.round(bands.lowTrimDb || 0)}/M${Math.round(bands.midTrimDb || 0)}/H${Math.round(bands.highTrimDb || 0)}dB`;
  const guardLabel = report.outgoingVocalGuarded ? "A guard on" : "A guard off";
  const methodLabel = report.strategyAdapted ? `${report.requestedMethod}->${report.renderMethod}` : report.renderMethod || preview.method;
  const overlapLabel = report.requestedOverlapDuration && report.renderOverlapDuration && Math.abs(report.requestedOverlapDuration - report.renderOverlapDuration) > 0.1
    ? `${formatTime(report.requestedOverlapDuration)}->${formatTime(report.renderOverlapDuration)}`
    : formatTime(report.renderOverlapDuration || report.overlapDuration || 0);
  const cueRefinement = report.cueRefinement || {};
  const cueLabel = cueRefinement.enabled
    ? `Cue A${formatSignedSeconds(cueRefinement.outgoingShiftSec)} B${formatSignedSeconds(cueRefinement.incomingShiftSec)}`
    : "Cue original";
  return `
    <div class="preview-result">
      <div>
        <strong>试听已生成</strong>
        <a href="${API}${preview.url}" target="_blank" rel="noreferrer">打开 WAV</a>
      </div>
      ${preview.url ? `
        <audio
          class="preview-audio"
          controls
          controlsList="nodownload"
          preload="metadata"
          src="${escapeHtml(`${API}${preview.url}`)}"
        ></audio>
      ` : ""}
      ${preview.bufferError ? `<small>前端解码失败，但仍可用播放器直接试听：${escapeHtml(preview.bufferError)}</small>` : ""}
      <div class="preview-report">
        <span>${escapeHtml(methodLabel || "")}</span>
        <span>${escapeHtml(cueLabel)}</span>
        <span>Overlap ${overlapLabel}</span>
        <span>Tempo ${Math.round(report.tempoChangePercent || 0)}%</span>
        <span>Drift ${Math.round(report.transientShiftMs || 0)}ms</span>
        <span>Vocal +${Math.round(report.incomingVocalDelayMs || 0)}ms</span>
        <span>Energy ${Math.round(report.incomingEnergyTrimDb || 0)}dB</span>
        <span>Bands ${bandSummary}</span>
        <span>Glue ${Math.round((report.glueWet || 0) * 100)}%</span>
        <span>Bridge ${Math.round((report.rhythmBridgeWet || 0) * 100)}%</span>
        <span>${guardLabel}</span>
        <span>Pitch ${report.pitchShiftSemitones || 0} st</span>
        <span>Vocal ${Math.round((report.vocalConflictAfter || 0) * 100)}%</span>
        <span>Risk ${Math.round((report.riskScore || 0) * 100)}</span>
      </div>
      <p>${escapeHtml(report.explanation || "")}</p>
      ${(preview.warnings || []).length ? `<small>${escapeHtml(preview.warnings.slice(0, 2).join("；"))}</small>` : ""}
    </div>
  `;
}

function teachingPreviewFor(nextId, currentId = selectedTrack()?.localId) {
  const preview = state.teaching.previews[nextId];
  if (!preview) return null;
  if (currentId && preview.outgoingLocalId && preview.outgoingLocalId !== currentId) return null;
  return preview;
}

function serializableTransitionPreview(preview, current, next) {
  return {
    url: preview.url,
    audioPath: preview.audioPath,
    previewStartTime: preview.previewStartTime,
    previewEndTime: preview.previewEndTime,
    outgoingCue: preview.outgoingCue,
    incomingCue: preview.incomingCue,
    alignment: preview.alignment,
    method: preview.method,
    processingReport: preview.processingReport,
    outgoingTrackId: current.id,
    incomingTrackId: next.id,
  };
}

async function hydrateTeachingPreviewAudio(preview) {
  if (!preview?.url || preview.buffer) return;
  try {
    const context = await getAudioContext();
    const response = await fetch(apiUrl(preview.url));
    if (!response.ok) throw new Error("preview audio fetch failed");
    const arrayBuffer = await response.arrayBuffer();
    preview.buffer = await context.decodeAudioData(arrayBuffer.slice(0));
    preview.bufferDuration = preview.buffer.duration;
  } catch (error) {
    preview.bufferError = error.message || "preview audio decode failed";
  }
}

function effectiveTeachingCue(rec, preview) {
  const alignment = preview?.alignment || {};
  const report = preview?.processingReport || {};
  const outgoingTime = Number(preview?.outgoingCue?.time ?? alignment.outgoingExitTime ?? rec.outgoingCue.time);
  const incomingTime = Number(preview?.incomingCue?.time ?? alignment.incomingEntryTime ?? rec.incomingCue.time);
  const overlapDuration = Number(report.renderOverlapDuration ?? alignment.overlapDuration ?? rec.overlapDuration ?? 0);
  const method = report.renderMethod || preview?.method || rec.method;
  const outgoingShift = outgoingTime - Number(rec.outgoingCue.time || 0);
  const incomingShift = incomingTime - Number(rec.incomingCue.time || 0);
  const adjusted = Boolean(
    preview &&
      ((report.cueRefinement || {}).enabled ||
        Math.abs(outgoingShift) > 0.05 ||
        Math.abs(incomingShift) > 0.05 ||
        method !== rec.method ||
        Math.abs(overlapDuration - Number(rec.overlapDuration || 0)) > 0.1),
  );
  return {
    outgoingTime,
    incomingTime,
    overlapDuration,
    method,
    adjusted,
    outgoingShift,
    incomingShift,
    methodAdjusted: method !== rec.method,
  };
}

function renderCueAdjustmentNote(rec, effective) {
  const parts = [];
  if (Math.abs(effective.outgoingShift) > 0.05) parts.push(`A OUT ${formatSignedSeconds(effective.outgoingShift)}`);
  if (Math.abs(effective.incomingShift) > 0.05) parts.push(`B IN ${formatSignedSeconds(effective.incomingShift)}`);
  if (effective.methodAdjusted) parts.push(`${methodLabel(rec.method)} -> ${methodLabel(effective.method)}`);
  return `
    <div class="cue-adjustment-note">
      <strong>实际试听 cue</strong>
      <span>${escapeHtml(parts.length ? parts.join(" · ") : "已按渲染结果同步")}</span>
      <small>使用这个接法时会采用这里的 A OUT / B IN，而不是原始推荐点。</small>
    </div>
  `;
}

function renderTeachingDebug(debug) {
  if (!debug) return "";
  const metrics = [
    ["人声安全", 1 - debug.vocalConflictScore],
    ["乐句", debug.phraseScore],
    ["新手", debug.beginnerScore],
    ["能量", debug.energyScore],
    ["BPM", debug.bpmScore],
    ["调性", debug.keyScore],
  ];
  return `
    <details class="teaching-debug">
      <summary>评分拆解 · ${Math.round(debug.finalScore * 100)} / 100</summary>
      <div class="debug-meter-grid">
        ${metrics.map(([label, value]) => renderDebugMeter(label, value)).join("")}
      </div>
      <div class="debug-notes">
        ${(debug.reasons || []).slice(0, 5).map((reason) => `<span>${escapeHtml(reason)}</span>`).join("")}
      </div>
    </details>
  `;
}

function renderDebugMeter(label, value) {
  const percent = Math.round(clamp(Number(value) || 0, 0, 1) * 100);
  return `
    <div class="debug-meter">
      <span>${escapeHtml(label)}</span>
      <b>${percent}</b>
      <i style="--value:${percent}%"></i>
    </div>
  `;
}

function renderTeachingStep(step) {
  return `
    <div class="teaching-step">
      <span>${step.targetDeck}</span>
      <strong>${escapeHtml(actionLabel(step.action))}</strong>
      <small>${step.atBeatOffset >= 0 ? "+" : ""}${step.atBeatOffset} beat · ${formatSignedTime(step.atTimeOffset)}</small>
      <p>${escapeHtml(step.explanation)}</p>
    </div>
  `;
}

function renderMixerSlider(trackId, param, label, value, min, max, step, readout) {
  return `
    <label class="deck-slider">
      <span>${label}<b data-mixer-readout>${readout}</b></span>
      <input type="range" min="${min}" max="${max}" step="${step}" value="${value}" data-mixer="${param}" data-id="${trackId}" />
    </label>
  `;
}

function drawStemWaveforms() {
  if (state.view !== "stems" || !els.stemDeck) return;
  const track = activeStemTrack();
  const canvases = els.stemDeck.querySelectorAll("canvas[data-stem-wave]");
  canvases.forEach((canvas) => drawStemWaveform(canvas, track, canvas.dataset.stemWave));
  if (isStemPending(track) && !state.stemDebugger.scanFrame) {
    state.stemDebugger.scanFrame = window.requestAnimationFrame(() => {
      state.stemDebugger.scanFrame = null;
      drawStemWaveforms();
    });
  }
}

function drawStemWaveform(canvas, track, stemId) {
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  if (canvas.width !== Math.floor(rect.width * ratio) || canvas.height !== Math.floor(rect.height * ratio)) {
    canvas.width = Math.floor(rect.width * ratio);
    canvas.height = Math.floor(rect.height * ratio);
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#050606";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  for (let x = 0; x < width; x += 28) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  const stem = STEMS.find((item) => item.id === stemId) || STEMS[0];
  const realStem = track?.stems?.[stemId];
  const peaks = realStem?.peaks?.length ? realStem.peaks : track?.peaks || [];
  const pending = !realStem && isStemPending(track);
  const waveColor = realStem ? stem.color : "#737b80";
  if (!track || !peaks.length) {
    ctx.fillStyle = "rgba(255,255,255,0.34)";
    ctx.font = "13px Bahnschrift, Segoe UI, sans-serif";
    ctx.fillText("Waiting for audio", 18, height / 2);
    return;
  }

  const center = height / 2;
  const barWidth = Math.max(1, width / peaks.length);
  const control = ensureStemControl(stem.id);
  ctx.fillStyle = control.mute ? "rgba(120,120,120,0.36)" : waveColor;
  peaks.forEach((value, index) => {
    const shaped = realStem ? Number(value) || 0 : shapeStemPeak(Number(value) || 0, index, stem.id);
    const barHeight = Math.max(1, shaped * height * 0.42);
    ctx.globalAlpha = control.mute ? 0.38 : pending ? 0.24 : 0.72 + Math.min(0.24, shaped * 0.24);
    ctx.fillRect(index * barWidth, center - barHeight, Math.max(1, barWidth * 0.72), barHeight * 2);
  });
  ctx.globalAlpha = 1;
  if (pending) {
    const scanWidth = Math.max(80, width * 0.12);
    const scanX = ((Date.now() / 14) % (width + scanWidth * 2)) - scanWidth;
    const gradient = ctx.createLinearGradient(scanX, 0, scanX + scanWidth, 0);
    gradient.addColorStop(0, "rgba(255,255,255,0)");
    gradient.addColorStop(0.5, "rgba(255,255,255,0.22)");
    gradient.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(scanX, 0, scanWidth, height);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.28)";
  ctx.beginPath();
  ctx.moveTo(0, center);
  ctx.lineTo(width, center);
  ctx.stroke();

  const x = track.duration ? (state.stemDebugger.playbackOffset / track.duration) * width : 0;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(x - 1, 0, 2, height);
}

function shapeStemPeak(value, index, stemId) {
  if (stemId === "bass") return Math.pow(value, 0.82) * (0.78 + Math.sin(index * 0.08) * 0.08);
  if (stemId === "drums") return Math.min(1, Math.pow(value, 1.45) * (1.18 + (index % 11 === 0 ? 0.42 : 0)));
  if (stemId === "vocals") return Math.pow(value, 1.08) * (0.72 + Math.sin(index * 0.045) * 0.18);
  return Math.pow(value, 1.2) * (0.64 + Math.cos(index * 0.035) * 0.12);
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
    els.cueEditor.innerHTML = "";
    drawEmptyWave(ctx, width, height);
    return;
  }
  els.selectedTitle.textContent = track.name;
  els.handleReadout.textContent = `入点 ${formatTime(track.introPoint)} / 出点 ${formatTime(track.outroPoint)}`;
  renderCueEditor(track);

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
    const local = item.sourceStart + state.playbackOffset - item.start;
    const x = (local / track.duration) * width;
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.fillRect(x - 1, 0, 2, height);
  }
}

function renderCueEditor(track) {
  const candidate = track.transition_candidates || {};
  els.cueEditor.innerHTML = `
    <label>
      <span>入点 IN</span>
      <input type="number" min="0" max="${Math.floor(track.duration)}" step="0.1" value="${roundOne(track.introPoint)}" data-cue="intro" />
    </label>
    <label>
      <span>出点 OUT</span>
      <input type="number" min="0" max="${Math.floor(track.duration)}" step="0.1" value="${roundOne(track.outroPoint)}" data-cue="outro" />
    </label>
    <button type="button" data-cue-action="reset">恢复 AI 切点</button>
    <div class="cue-ai">
      <span>AI: ${escapeHtml(candidate.method || "manual")}</span>
      <span>Vocal IN ${formatDensity(candidate.intro_vocal_density)} / OUT ${formatDensity(candidate.outro_vocal_density)}</span>
      <span>Beat ${Math.round((track.beat_confidence || 0) * 100)}%</span>
    </div>
  `;
}

function updateCueEditor(event) {
  const input = event.target.closest("input[data-cue]");
  const track = selectedTrack();
  if (!input || !track) return;
  const value = Number(input.value);
  if (!Number.isFinite(value)) return;
  if (input.dataset.cue === "intro") track.introPoint = clamp(value, 0, Math.max(0, track.outroPoint - 0.5));
  if (input.dataset.cue === "outro") track.outroPoint = clamp(value, Math.min(track.duration - 0.5, track.introPoint + 0.5), track.duration);
  render();
}

function cueEditorClick(event) {
  const button = event.target.closest("button[data-cue-action='reset']");
  const track = selectedTrack();
  if (!button || !track?.transition_candidates) return;
  track.introPoint = clamp(track.transition_candidates.intro ?? track.introPoint, 0, Math.max(0, track.outroPoint - 0.5));
  track.outroPoint = clamp(track.transition_candidates.outro ?? track.outroPoint, Math.min(track.duration - 0.5, track.introPoint + 0.5), track.duration);
  render();
}

function toTeachingAnalysis(track) {
  const bpm = Number(track.bpm) || 120;
  return {
    id: track.localId,
    title: track.name,
    duration: track.duration || 0,
    bpm,
    key: track.key || "unknown",
    camelotKey: track.camelot || undefined,
    energyCurve: teachingEnergyCurve(track),
    sections: teachingSections(track, bpm),
    beatGrid: teachingBeatGrid(track, bpm),
    vocalDensityCurve: teachingVocalCurve(track),
  };
}

function teachingBeatGrid(track, bpm) {
  if (track.beats?.length) {
    return track.beats.map((time, beatIndex) => {
      const barIndex = Math.floor(beatIndex / 4);
      return {
        time,
        beatIndex,
        barIndex,
        phraseIndex: Math.floor(barIndex / 8),
      };
    });
  }
  const beatSeconds = 60 / Math.max(1, bpm);
  const total = Math.max(1, Math.floor((track.duration || 0) / beatSeconds));
  return Array.from({ length: total }, (_, beatIndex) => {
    const barIndex = Math.floor(beatIndex / 4);
    return {
      time: beatIndex * beatSeconds,
      beatIndex,
      barIndex,
      phraseIndex: Math.floor(barIndex / 8),
    };
  });
}

function teachingSections(track, bpm) {
  if (Array.isArray(track.sections) && track.sections.length) {
    return track.sections.map((section) => ({
      type: section.type,
      startTime: Number(section.startTime) || 0,
      endTime: Number(section.endTime) || 0,
      startBeat: Number(section.startBeat) || 0,
      endBeat: Number(section.endBeat) || 0,
      confidence: Number(section.confidence) || 0.5,
    })).filter((section) => section.endTime > section.startTime);
  }
  const duration = Math.max(track.duration || 0, 1);
  const introEnd = clamp(track.introPoint || track.transition_candidates?.intro || duration * 0.12, 2, duration * 0.35);
  const outroStart = clamp(track.outroPoint || track.transition_candidates?.outro || duration * 0.82, duration * 0.55, duration - 1);
  const points = [
    ["intro", 0, introEnd, 0.86],
    ["verse", introEnd, duration * 0.38, 0.58],
    ["chorus", duration * 0.38, duration * 0.58, 0.76],
    ["bridge", duration * 0.58, duration * 0.68, 0.56],
    ["breakdown", duration * 0.68, Math.min(duration * 0.78, outroStart), 0.64],
    ["drop", Math.min(duration * 0.78, outroStart), outroStart, 0.68],
    ["outro", outroStart, duration, 0.82],
  ];
  const beatSeconds = 60 / Math.max(1, bpm);
  return points
    .map(([type, start, end, confidence]) => ({
      type,
      startTime: clamp(Number(start), 0, duration),
      endTime: clamp(Number(end), 0, duration),
      startBeat: Math.round(Number(start) / beatSeconds),
      endBeat: Math.round(Number(end) / beatSeconds),
      confidence: Number(confidence),
    }))
    .filter((section) => section.endTime - section.startTime >= 0.5);
}

function teachingEnergyCurve(track) {
  if (Array.isArray(track.energy_curve) && track.energy_curve.length) {
    return track.energy_curve.map((point) => ({
      time: Number(point.time) || 0,
      energy: clamp(Number(point.energy) || 0, 0, 1),
    }));
  }
  const duration = Math.max(track.duration || 0, 1);
  const base = Number(track.energy) || 0.5;
  const intro = track.transition_candidates?.intro_energy ?? track.energy_profile?.intro_relative_energy ?? base * 0.75;
  const outro = track.transition_candidates?.outro_energy ?? track.energy_profile?.outro_relative_energy ?? base * 0.65;
  return [
    { time: 0, energy: clamp(intro, 0, 1) },
    { time: duration * 0.25, energy: clamp(base * 0.9, 0, 1) },
    { time: duration * 0.5, energy: clamp(Math.max(base, 0.55), 0, 1) },
    { time: duration * 0.72, energy: clamp(base * 1.05, 0, 1) },
    { time: duration, energy: clamp(outro, 0, 1) },
  ];
}

function teachingVocalCurve(track) {
  if (Array.isArray(track.vocal_density_curve) && track.vocal_density_curve.length) {
    return track.vocal_density_curve.map((point) => ({
      time: Number(point.time) || 0,
      density: clamp(Number(point.density) || 0, 0, 1),
    }));
  }
  const duration = Math.max(track.duration || 0, 1);
  const intro = track.transition_candidates?.intro_vocal_density;
  const outro = track.transition_candidates?.outro_vocal_density;
  const fallback = Number.isFinite(intro) || Number.isFinite(outro) ? 0.35 : 0.2;
  return [
    { time: 0, density: Number.isFinite(intro) ? intro : fallback * 0.55 },
    { time: duration * 0.28, density: Math.max(fallback, 0.55) },
    { time: duration * 0.48, density: Math.max(fallback, 0.62) },
    { time: duration * 0.7, density: fallback },
    { time: duration, density: Number.isFinite(outro) ? outro : fallback * 0.5 },
  ];
}

function formatDensity(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "--";
}

function roundOne(value) {
  return Math.round((value || 0) * 10) / 10;
}

function methodLabel(method) {
  return {
    fade: "渐隐",
    end_to_end: "首尾切换",
    quick_cut: "快切",
    beatmix: "对拍混音",
    bass_swap: "低频替换",
    echo_out: "回声退出",
    filter_sweep: "滤波扫频",
    breakdown_switch: "空拍切换",
    wide_bpm_loop: "大 BPM 差 Loop",
    loop_build: "Loop Build",
    instrumental_bridge: "伴奏过桥",
    acapella_mashup: "清唱叠加",
  }[method] || method;
}

function actionLabel(action) {
  return {
    press_play: "按播放",
    set_cue: "设 Cue",
    start_loop: "开 Loop",
    halve_loop: "Loop 减半",
    increase_filter: "加滤波",
    decrease_filter: "收滤波",
    filter_sweep: "扫滤波",
    enable_echo: "开 Echo",
    disable_echo: "关 Echo",
    fade_out: "淡出",
    fade_in: "淡入",
    eq_low_cut: "切低频",
    eq_low_swap: "换低频",
    crossfader_move: "推横推",
    stop_track: "停 A 歌",
  }[action] || action;
}

function riskLine(method) {
  if (method === "beatmix" || method === "bass_swap") return "容易翻车：两个底鼓同时打开。先关 B 低频，到乐句边界再交换。";
  if (method === "quick_cut") return "容易翻车：不在第一拍切。数完当前乐句，在强拍瞬间切过去。";
  if (method === "echo_out") return "容易翻车：Echo 盖住新歌。B 歌进来后要马上收掉 A。";
  if (method === "wide_bpm_loop") return "容易翻车：loop 到人声。只在无人声鼓组上做循环。";
  return "容易翻车：两首歌主唱叠唱。听到人声打架就缩短重叠。";
}

function formatSignedTime(seconds) {
  const sign = seconds >= 0 ? "+" : "-";
  return `${sign}${formatTime(Math.abs(seconds))}`;
}

function formatSignedSeconds(seconds) {
  const value = Number(seconds || 0);
  const sign = value >= 0 ? "+" : "-";
  return `${sign}${Math.abs(value).toFixed(1)}s`;
}

function formatNumber(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "--";
}

function formatSignedNumber(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}`;
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

function lerp(from, to, amount) {
  const t = clamp(Number(amount) || 0, 0, 1);
  return from + (to - from) * t;
}

function dbToGain(db) {
  return 10 ** ((Number(db) || 0) / 20);
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
window.addEventListener("resize", drawStemWaveforms);
