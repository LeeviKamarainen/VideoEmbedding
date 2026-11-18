// The page. Four sources of pictures: the demo video (its frames' embeddings ready in
// data/index.json), a video the visitor picks (indexed here), their photos, or their live camera.
// Each becomes a memory of { t, vector, label, thumb } that the search box ranks, and the
// triggers ("tell me when I see…") watch a playing video or the camera as it goes. Everything
// runs in the browser: EmbeddingGemma 2 on WebGPU in worker.js, the ranking in search.js and the
// triggers in triggers.js.

import { draw, ready, sample, thumbnails, toBlob } from "./frames.js";
import { clock, fromIndex, search, toIndex, worthKeeping } from "./search.js";
import { Watcher } from "./triggers.js";

const MODEL = "onnx-community/embeddinggemma-2-ONNX";
const DEMO_INDEX = "data/index.json";
const CAMERA_EVERY_MS = 500; // two frames a second, as the glasses stream

const $ = (id) => document.getElementById(id);
const els = {
  tabs: [...document.querySelectorAll("[data-source]")],
  panels: [...document.querySelectorAll("[data-panel]")],
  video: $("video"),
  still: $("still"),
  source: $("source"),
  credit: $("credit"),
  footerCredit: $("footer-credit"),
  file: $("file"),
  every: $("every"),
  index: $("index-button"),
  stop: $("stop-button"),
  download: $("download-button"),
  photos: $("photos"),
  camera: $("camera-button"),
  flip: $("flip-button"),
  progress: $("progress"),
  progressText: $("progress-text"),
  model: $("model-status"),
  form: $("search-form"),
  query: $("query"),
  ask: $("ask-button"),
  examples: $("examples"),
  watchForm: $("watch-form"),
  watch: $("watch"),
  watching: $("watching"),
  results: $("results"),
  timeline: $("timeline"),
  error: $("error"),
  alert: $("alert"),
  alertFrame: $("alert-frame"),
  alertText: $("alert-text"),
  alertDetail: $("alert-detail"),
  alertWrong: $("alert-wrong"),
  alertClose: $("alert-close"),
  zoom: $("zoom"),
  zoomImage: $("zoom-image"),
};

// What is being searched now.
const state = {
  source: "demo",
  memory: [], // [{ t, vector, label, thumb? }]
  videoName: "",
  thumbs: null, // thumbnails(src) for a video
  busy: null, // AbortController while indexing
  demo: null, // { src, frames, index } once loaded
  fed: 0, // a playing video: the time up to which its frames went to the watcher
  stream: null, // the camera's MediaStream
  facing: "environment",
  cameraStart: 0,
  cameraTimer: 0,
  cameraBusy: false,
  lastFired: null, // { trigger, score }
};
const watcher = new Watcher();

// ---- The model, in its worker ---------------------------------------------------------------

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const pending = new Map();
let nextId = 1;
worker.onmessage = ({ data }) => {
  if (data.type === "progress") return showDownload(data.loaded, data.total);
  const waiting = pending.get(data.id);
  if (!waiting) return;
  pending.delete(data.id);
  if (data.type === "error") waiting.reject(new Error(data.message));
  else waiting.resolve(data);
};
function ask(message) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...message, id });
  });
}

let loadedVision = null; // null: nothing loaded; false: text only; true: text and images
let loading = null;
async function load(vision) {
  if (loadedVision === true || loadedVision === vision) return;
  if (loading) {
    await loading;
    return load(vision);
  }
  els.model.textContent = vision
    ? "Loading EmbeddingGemma 2 with its image encoder…"
    : "Loading EmbeddingGemma 2…";
  loading = ask({ type: "load", vision });
  try {
    const { device, dtype } = await loading;
    loadedVision = vision;
    els.model.textContent = `EmbeddingGemma 2 ready · ${dtype} · ${
      device === "webgpu" ? "WebGPU" : "CPU (no WebGPU: slow)"
    }`;
  } finally {
    loading = null;
    hideProgress();
  }
}

async function embedText(text) {
  await load(loadedVision ?? false);
  return (await ask({ type: "text", text })).vector;
}

async function embedImage(blob) {
  await load(true);
  return (await ask({ type: "image", blob })).vector;
}

// ---- Progress and errors --------------------------------------------------------------------

function showDownload(loaded, total) {
  if (!total) return;
  const mb = (bytes) => Math.round(bytes / 1e6);
  showProgress(loaded / total, `Downloading the model: ${mb(loaded)} / ${mb(total)} MB (once)`);
}
function showProgress(fraction, text) {
  els.progress.hidden = false;
  els.progress.value = fraction;
  els.progressText.textContent = text;
}
function hideProgress() {
  els.progress.hidden = true;
  els.progressText.textContent = "";
}
function fail(problem) {
  els.error.textContent = problem instanceof Error ? problem.message : String(problem);
  els.error.hidden = false;
}
function clearError() {
  els.error.hidden = true;
}

// ---- Sources --------------------------------------------------------------------------------

function choose(source) {
  if (state.busy) state.busy.abort();
  if (source !== "camera") stopCamera();
  state.source = source;
  for (const tab of els.tabs) tab.setAttribute("aria-selected", String(tab.dataset.source === source));
  for (const panel of els.panels) panel.hidden = panel.dataset.panel !== source;
  clearError();
  hideAlert();
  setMemory([]);
  els.still.hidden = true;
  els.video.hidden = false;
  els.video.controls = source !== "camera";
  els.video.pause();
  els.video.srcObject = null;
  els.video.removeAttribute("src");
  els.video.load();
  if (source === "demo") showDemo();
  else if (source === "video") els.source.textContent = "Choose a video, then index it.";
  else if (source === "photos") {
    els.video.hidden = true;
    els.source.textContent = "Choose photos to search them.";
  } else els.source.textContent = "Start the camera: what it sees is remembered as it goes.";
  showExamples(source === "demo" ? state.demo?.index : null);
  updateControls();
}
for (const tab of els.tabs) tab.addEventListener("click", () => choose(tab.dataset.source));

function setMemory(memory) {
  state.memory = memory;
  state.fed = 0;
  watcher.reset();
  els.results.replaceChildren();
  els.timeline.replaceChildren();
  updateControls();
}

function useVideo(src, name) {
  els.video.srcObject = null;
  els.video.src = src;
  state.videoName = name;
  state.thumbs = thumbnails(src);
}

function updateControls() {
  const busy = Boolean(state.busy);
  els.index.disabled = busy || !els.video.src;
  els.stop.hidden = !busy;
  els.download.disabled = busy || !state.memory.length || state.source !== "video";
  els.query.disabled = !state.memory.length && !busy;
  els.ask.disabled = !state.memory.length;
  els.photos.disabled = busy;
}

// ---- The demo -------------------------------------------------------------------------------

async function loadDemo() {
  try {
    const response = await fetch(DEMO_INDEX);
    if (!response.ok) throw new Error(`no demo (${response.status})`);
    const index = await response.json();
    const src = new URL(index.video, new URL(DEMO_INDEX, location.href)).href;
    state.demo = { src, frames: fromIndex(index), index };
    showCredit(index.credit);
  } catch {
    state.demo = null;
  }
  if (state.source === "demo") choose("demo");
}

function showDemo() {
  if (!state.demo) {
    els.source.textContent = "There is no demo here: pick another source.";
    return;
  }
  useVideo(state.demo.src, state.demo.index.video);
  setMemory(state.demo.frames.map((f) => ({ ...f, label: clock(f.t) })));
  els.source.textContent = `${state.demo.frames.length} frames indexed. Search them, or press play and watch for something.`;
  // The frames' embeddings came with the page; the model that embeds a question has to be
  // downloaded (once, then the browser keeps it). Start now, so the first search does not wait.
  if (loadedVision === null && !loading) {
    load(false).catch((problem) => fail(`The model could not load: ${problem.message}`));
  }
}

function showCredit(credit) {
  if (!credit) return;
  const link = (text, url) => {
    const a = document.createElement("a");
    a.href = url;
    a.textContent = text;
    return a;
  };
  const parts = [
    "Video: ",
    link(`“${credit.title}”`, credit.url),
    ` by ${credit.author}${credit.source ? `, from ${credit.source}` : ""} (`,
    credit.license_url ? link(credit.license, credit.license_url) : credit.license,
    ").",
  ];
  els.credit.replaceChildren(...parts);
  els.footerCredit.replaceChildren(
    "Demo video: ",
    ...parts.slice(1).map((part) => (part instanceof Node ? part.cloneNode(true) : part)),
  );
}

function showExamples(index) {
  const searches = index?.examples ?? [];
  const watches = index?.watch_examples ?? [];
  const button = (text, run) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    b.addEventListener("click", run);
    return b;
  };
  els.examples.replaceChildren(
    ...searches.map((text) =>
      button(text, () => {
        els.query.value = text;
        els.form.requestSubmit();
      }),
    ),
    ...watches.map((text) =>
      button(`👁 ${text}`, () => {
        els.watch.value = text;
        els.watchForm.requestSubmit();
      }),
    ),
  );
}

// ---- Your video -----------------------------------------------------------------------------

els.file.addEventListener("change", () => {
  const file = els.file.files?.[0];
  if (!file) return;
  clearError();
  useVideo(URL.createObjectURL(file), file.name);
  setMemory([]);
  els.source.textContent = `${file.name}: not indexed yet. Press “Index this video”.`;
});

els.index.addEventListener("click", async () => {
  clearError();
  const every = Number(els.every.value);
  const controller = new AbortController();
  state.busy = controller;
  setMemory([]);
  // A video element of its own, so the visible one stays free to play.
  const reader = document.createElement("video");
  reader.muted = true;
  reader.preload = "auto";
  reader.src = els.video.src;
  try {
    await load(true);
    await ready(reader);
    const total = Math.ceil(reader.duration / every);
    const kept = [];
    let seen = 0;
    const started = performance.now();
    for await (const { t, canvas } of sample(reader, every, { signal: controller.signal })) {
      const vector = await embedImage(await toBlob(canvas));
      seen += 1;
      const frame = { t, vector, label: clock(t) };
      if (worthKeeping(frame, kept.at(-1))) {
        kept.push(frame);
        state.thumbs.put(t, canvas);
      }
      const perFrame = (performance.now() - started) / seen / 1000;
      showProgress(
        seen / total,
        `Frame ${seen} / ${total} · ${kept.length} kept · about ${Math.ceil((total - seen) * perFrame)} s left`,
      );
    }
    hideProgress();
    if (state.source !== "video") return;
    setMemory(kept);
    els.source.textContent = `${state.videoName}: ${kept.length} frames indexed${
      controller.signal.aborted ? " (stopped early)" : ""
    }. Search them, or press play and watch for something.`;
  } catch (problem) {
    hideProgress();
    fail(problem);
  } finally {
    state.busy = null;
    reader.removeAttribute("src");
    updateControls();
  }
});

els.stop.addEventListener("click", () => state.busy?.abort());

els.download.addEventListener("click", () => {
  const index = toIndex({ model: MODEL, video: state.videoName, frames: state.memory });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([JSON.stringify(index)], { type: "application/json" }));
  link.download = "index.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});

// A playing video feeds its indexed frames to the watcher as the playhead passes them, so the
// triggers fire as they would on the glasses, with no model work while it plays.
els.video.addEventListener("timeupdate", () => {
  if (state.source === "camera" || !state.memory.length || els.video.paused) return;
  const now = els.video.currentTime;
  for (const frame of state.memory) {
    if (frame.t <= state.fed || frame.t > now) continue;
    const fired = watcher.see(frame.t, frame.vector);
    if (fired) showAlert(fired, frame);
  }
  state.fed = Math.max(state.fed, now);
  showWatching();
});
els.video.addEventListener("seeking", () => {
  // Back in time: what was seen after this has not happened yet.
  if (state.source !== "camera" && els.video.currentTime < state.fed) {
    watcher.reset();
    state.fed = els.video.currentTime;
    showWatching();
  }
});

// ---- Your photos ----------------------------------------------------------------------------

els.photos.addEventListener("change", async () => {
  const files = [...(els.photos.files ?? [])];
  if (!files.length) return;
  clearError();
  const controller = new AbortController();
  state.busy = controller;
  updateControls();
  try {
    await load(true);
    const memory = [...state.memory];
    for (const [i, file] of files.entries()) {
      if (controller.signal.aborted) break;
      showProgress(i / files.length, `Photo ${i + 1} / ${files.length}`);
      const thumb = URL.createObjectURL(file);
      memory.push({ t: memory.length, vector: await embedImage(file), label: file.name, thumb });
    }
    hideProgress();
    if (state.source !== "photos") return;
    setMemory(memory);
    els.source.textContent = `${memory.length} photos indexed. Search them.`;
  } catch (problem) {
    hideProgress();
    fail(problem);
  } finally {
    state.busy = null;
    els.photos.value = "";
    updateControls();
  }
});

// ---- The live camera ------------------------------------------------------------------------

els.camera.addEventListener("click", () => (state.stream ? stopCamera() : startCamera()));
els.flip.addEventListener("click", async () => {
  state.facing = state.facing === "environment" ? "user" : "environment";
  stopCamera(false);
  await startCamera(false);
});

async function startCamera(fresh = true) {
  clearError();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: state.facing, width: { ideal: 1280 } },
      audio: false,
    });
  } catch (problem) {
    fail(`The camera could not start: ${problem.message ?? problem}`);
    return;
  }
  els.video.srcObject = state.stream;
  els.video.muted = true;
  await els.video.play().catch(() => {});
  if (fresh) {
    setMemory([]);
    state.cameraStart = performance.now();
  }
  const cameras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  els.flip.hidden = cameras.length < 2;
  els.camera.textContent = "Stop the camera";
  els.source.textContent = "Loading the image encoder…";
  try {
    await load(true);
  } catch (problem) {
    fail(problem);
    return stopCamera();
  }
  state.cameraTimer = setInterval(cameraTick, CAMERA_EVERY_MS);
}

function stopCamera(final = true) {
  clearInterval(state.cameraTimer);
  state.cameraTimer = 0;
  for (const track of state.stream?.getTracks() ?? []) track.stop();
  state.stream = null;
  if (state.source === "camera") els.video.srcObject = null;
  els.camera.textContent = "Start the camera";
  if (final) els.flip.hidden = true;
}

/** One camera frame: embed it, remember it if it is new, and let the triggers see it. A frame due
 * while the last one is still being embedded is skipped. */
async function cameraTick() {
  if (state.cameraBusy || !state.stream || els.video.readyState < 2) return;
  state.cameraBusy = true;
  try {
    const t = (performance.now() - state.cameraStart) / 1000;
    const canvas = draw(els.video, 768);
    const vector = await embedImage(await toBlob(canvas));
    if (!state.stream) return;
    const frame = { t, vector, label: new Date().toLocaleTimeString() };
    if (worthKeeping(frame, state.memory.at(-1))) {
      frame.thumb = draw(canvas, 240).toDataURL("image/jpeg", 0.8);
      state.memory.push(frame);
      updateControls();
    }
    els.source.textContent = `Live: ${state.memory.length} frames remembered.`;
    const fired = watcher.see(t, vector);
    if (fired) showAlert(fired, { ...frame, thumb: frame.thumb ?? canvas.toDataURL("image/jpeg", 0.8) });
    showWatching();
  } catch (problem) {
    fail(problem);
  } finally {
    state.cameraBusy = false;
  }
}

// ---- Searching ------------------------------------------------------------------------------

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = els.query.value.trim();
  if (!text || !state.memory.length) return;
  clearError();
  els.ask.disabled = true;
  try {
    const started = performance.now();
    const sameMoment = state.source === "photos" ? 0 : undefined;
    const hits = search(await embedText(text), state.memory, 6, sameMoment);
    const ms = Math.round(performance.now() - started);
    showHits(hits, `${hits.length} ${state.source === "photos" ? "photos" : "moments"} for “${text}” in ${ms} ms`);
  } catch (problem) {
    fail(problem);
  } finally {
    updateControls();
  }
});

/** Show a hit: a video plays from there; a photo or a camera frame opens large. */
function open(hit) {
  if (state.source === "demo" || state.source === "video") {
    els.video.currentTime = hit.t;
    void els.video.play().catch(() => {});
  } else if (hit.thumb) {
    els.zoomImage.src = hit.thumb;
    els.zoomImage.alt = hit.label;
    els.zoom.showModal();
  }
}

async function thumbOf(hit) {
  return hit.thumb ?? state.thumbs?.get(hit.t);
}

function showHits(hits, summary) {
  const caption = document.createElement("p");
  caption.className = "summary";
  caption.textContent = summary;
  const cards = hits.map((hit, rank) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = rank === 0 ? "hit best" : "hit";
    const image = document.createElement("img");
    image.alt = hit.label;
    void thumbOf(hit).then((url) => url && (image.src = url));
    const label = document.createElement("span");
    const name = document.createElement("b");
    name.textContent = hit.label;
    const score = document.createElement("small");
    score.textContent = ` ${hit.score.toFixed(3)}`;
    label.append(name, score);
    card.append(image, label);
    card.addEventListener("click", () => open(hit));
    return card;
  });
  els.results.replaceChildren(caption, ...cards);

  // Where the moments are in a video.
  const isVideo = state.source === "demo" || state.source === "video";
  const duration = els.video.duration || Math.max(...state.memory.map((f) => f.t), 1);
  els.timeline.replaceChildren(
    ...(isVideo ? hits : []).map((hit, rank) => {
      const mark = document.createElement("button");
      mark.type = "button";
      mark.className = rank === 0 ? "mark best" : "mark";
      mark.style.left = `${(hit.t / duration) * 100}%`;
      mark.title = hit.label;
      mark.addEventListener("click", () => open(hit));
      return mark;
    }),
  );
  if (isVideo && hits[0]) els.video.currentTime = hits[0].t;
}

// ---- Watching for things --------------------------------------------------------------------

els.watchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = els.watch.value.trim();
  if (!text) return;
  clearError();
  try {
    watcher.add(text, await embedText(text));
    els.watch.value = "";
    showWatching();
  } catch (problem) {
    fail(problem);
  }
});

function showWatching() {
  els.watching.replaceChildren(
    ...watcher.triggers.map((trigger) => {
      const item = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = trigger.text;
      const meter = document.createElement("meter");
      // How close the last frame came to setting it off.
      meter.min = -0.05;
      meter.max = Math.max(trigger.level ?? 0.05, 0.05);
      meter.value = trigger.lastScore ?? meter.min;
      meter.title =
        trigger.level === null
          ? "Measuring the scene: it can fire after a few seconds of frames"
          : `Last frame ${(trigger.lastScore ?? 0).toFixed(3)}, fires at ${trigger.level.toFixed(3)}`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Stop watching for ${trigger.text}`);
      remove.addEventListener("click", () => {
        watcher.remove(trigger.id);
        showWatching();
      });
      item.append(name, meter, remove);
      return item;
    }),
  );
}

function showAlert(fired, frame) {
  state.lastFired = fired;
  els.alertText.textContent = fired.trigger.text;
  els.alertDetail.textContent = `At ${frame.label} · score ${fired.score.toFixed(3)} over ${fired.trigger.level.toFixed(3)}`;
  void thumbOf(frame).then((url) => {
    els.alertFrame.hidden = !url;
    if (url) els.alertFrame.src = url;
  });
  els.alert.hidden = false;
  navigator.vibrate?.(200);
  if ("speechSynthesis" in window) {
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(`I see ${fired.trigger.text}`));
  }
}

function hideAlert() {
  els.alert.hidden = true;
}

els.alertClose.addEventListener("click", hideAlert);
els.alertWrong.addEventListener("click", () => {
  const fired = state.lastFired;
  if (fired) watcher.notThis(fired.trigger.id, fired.score);
  hideAlert();
  showWatching();
});

// ---- Start ----------------------------------------------------------------------------------

if (!navigator.gpu) {
  els.model.textContent = "This browser has no WebGPU: the model will run on the CPU, which is much slower.";
}
if (!navigator.mediaDevices?.getUserMedia) {
  els.tabs.find((tab) => tab.dataset.source === "camera").disabled = true;
}
updateControls();
void loadDemo();
