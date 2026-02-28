import type { Cursor, CDFView, WidgetToken, LanguageModel } from "./types";
import { createBackendClient } from "./remote/backend";
import {
  forceCleanUtf8,
  fromByteLevelModel,
  interpolate,
  trieCache,
} from "./models";
import { normalizeCursor } from "./cursor";
import { buildScene } from "./scene";
import { renderScene } from "./render";

import { createCachedLSTMPredictor } from "./lstm/lstm";

function prefixToString(prefix: readonly WidgetToken[]): string {
  return prefix
    .map((t) =>
      t.type === "codepoint" ? String.fromCodePoint(t.codepoint) : t.label,
    )
    .join("");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/**
 * How many window-heights per second the cursor moves when the mouse
 * is at the edge of the canvas.  With SPEED = 2, full traversal of
 * the visible window takes ~1 second at maximum mouse offset.
 */
const SPEED = 2;

/** Cap dt to avoid huge jumps when the tab regains focus. */
const MAX_DT = 0.05;

async function main() {
  const hashParams = new URLSearchParams(window.location.hash.slice(1));
  let backendUrl = hashParams.get("backendUrl") ?? "http://localhost:8000";
  let remoteModelCallPrefix = hashParams.get("remoteModelCallPrefix") ?? "";
  let sliderValue = parseFloat(hashParams.get("mix") ?? "0");
  if (!Number.isFinite(sliderValue)) sliderValue = 0;
  sliderValue = Math.max(0, Math.min(1, sliderValue));

  // --- Config inputs ---
  const statusEl = document.getElementById("status") as HTMLElement;
  const backendUrlInput = document.getElementById(
    "backend-url",
  ) as HTMLInputElement;
  const modelPrefixInput = document.getElementById(
    "model-prefix",
  ) as HTMLTextAreaElement;
  const mixSlider = document.getElementById("mix-slider") as HTMLInputElement;

  backendUrlInput.value = backendUrl;
  modelPrefixInput.value = remoteModelCallPrefix;
  mixSlider.value = String(sliderValue);

  // --- LSTM side (loaded eagerly) ---
  statusEl.style.display = "";
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  const { predict: lstmPredict } = await createCachedLSTMPredictor(
    base,
    (msg) => {
      statusEl.textContent = msg;
    },
  );
  statusEl.style.display = "none";
  const lstmByteLM: LanguageModel<Uint8Array> = trieCache(
    forceCleanUtf8(lstmPredict),
  );

  // --- Remote side ---
  let currentMinProb = 0;

  function createRemoteLM(): LanguageModel<Uint8Array> {
    const { predictBytes } = createBackendClient(backendUrl);
    const prefixBytes = new TextEncoder().encode(remoteModelCallPrefix);
    return (prefix: Uint8Array) => {
      const full = new Uint8Array(prefixBytes.length + prefix.length);
      full.set(prefixBytes);
      full.set(prefix, prefixBytes.length);
      return predictBytes(full, currentMinProb);
    };
  }

  let remoteLM = createRemoteLM();

  // --- Interpolation ---
  let interpolated = interpolate([
    { model: lstmByteLM, weight: 1 - sliderValue },
    { model: remoteLM, weight: sliderValue },
  ]);

  function rebuildInterpolation() {
    interpolated = interpolate([
      { model: lstmByteLM, weight: 1 - sliderValue },
      { model: remoteLM, weight: sliderValue },
    ]);
  }

  // --- Stable model (never reassigned) ---
  const model: CDFView<readonly WidgetToken[], WidgetToken> =
    fromByteLevelModel((prefix, minProb) => {
      currentMinProb = minProb;
      return interpolated(prefix);
    }, []);

  // --- Hash ---
  function updateHash() {
    const params = new URLSearchParams();
    if (sliderValue !== 0) params.set("mix", String(sliderValue));
    if (backendUrl !== "http://localhost:8000") {
      params.set("backendUrl", backendUrl);
    }
    if (remoteModelCallPrefix)
      params.set("remoteModelCallPrefix", remoteModelCallPrefix);
    window.location.hash = params.toString();
  }

  // --- Config change handlers ---
  function applyConfigChange() {
    const newUrl = backendUrlInput.value;
    const newPrefix = modelPrefixInput.value;
    if (newUrl === backendUrl && newPrefix === remoteModelCallPrefix) return;
    backendUrl = newUrl;
    remoteModelCallPrefix = newPrefix;
    remoteLM = createRemoteLM();
    rebuildInterpolation();
    updateHash();
    renderController?.abort();
    renderController = new AbortController();
    render(renderController.signal);
  }

  backendUrlInput.addEventListener("blur", applyConfigChange);
  modelPrefixInput.addEventListener("blur", applyConfigChange);

  mixSlider.addEventListener("input", () => {
    sliderValue = parseFloat(mixSlider.value);
    rebuildInterpolation();
    updateHash();
    renderController?.abort();
    renderController = new AbortController();
    render(renderController.signal);
  });

  // --- DOM elements ---
  const prefixEl = document.getElementById("prefix-display")!;
  const nodeCanvas = document.getElementById(
    "node-canvas",
  ) as HTMLCanvasElement;
  const labelCanvas = document.getElementById(
    "label-canvas",
  ) as HTMLCanvasElement;
  const nodeCtx = nodeCanvas.getContext("2d")!;
  const labelCtx = labelCanvas.getContext("2d")!;

  // --- Cursor state ---
  let cursor: Cursor<WidgetToken> = { prefix: [], x: 0.5, y: 0.5 };

  // --- Mouse state ---
  let mouseDown = false;
  let mouseX = 0;
  let mouseY = 0;

  function updateMousePos(e: MouseEvent) {
    const rect = nodeCanvas.getBoundingClientRect();
    // Scale from CSS coords to canvas backing-store coords
    mouseX = (e.clientX - rect.left) * (nodeCanvas.width / rect.width);
    mouseY = (e.clientY - rect.top) * (nodeCanvas.height / rect.height);
  }

  nodeCanvas.addEventListener("mousedown", (e) => {
    mouseDown = true;
    updateMousePos(e);
  });

  nodeCanvas.addEventListener("mousemove", (e) => {
    updateMousePos(e);
  });

  window.addEventListener("mouseup", () => {
    mouseDown = false;
  });

  // --- Async render with abort support ---
  let renderController: AbortController | null = null;

  async function render(signal: AbortSignal) {
    const scene = await buildScene(model, cursor, 0.005);
    if (signal.aborted) return;
    await renderScene(
      nodeCtx,
      labelCtx,
      scene,
      nodeCanvas.width,
      nodeCanvas.height,
      signal,
    );
  }

  // --- Monotonic normalizeCursor ---
  let normalizeVersion = 0;

  async function updateAndRender(dx: number, dy: number) {
    const version = ++normalizeVersion;

    const newCursor = await normalizeCursor(model, {
      prefix: cursor.prefix,
      x: cursor.x + dx,
      y: cursor.y + dy,
    });

    if (version !== normalizeVersion) return;
    cursor = newCursor;
    const text = prefixToString(cursor.prefix);
    if (prefixEl.textContent !== text) {
      prefixEl.textContent = text;
      prefixEl.scrollTop = prefixEl.scrollHeight;
    }

    renderController?.abort();
    renderController = new AbortController();
    render(renderController.signal);
  }

  // --- Animation loop ---
  let lastTime = performance.now();

  function frame(now: number) {
    const dt = Math.min((now - lastTime) / 1000, MAX_DT);
    lastTime = now;

    if (mouseDown) {
      const w = nodeCanvas.width;
      const h = nodeCanvas.height;

      // Normalized displacement from center: [-1, 1] on each axis
      const ndx = (mouseX - w / 2) / (w / 2);
      const ndy = (mouseY - h / 2) / (h / 2);

      // halfHeight = size of the window in the cursor's local frame
      const halfHeight = 1 - cursor.x;

      // Velocity in local frame, proportional to displacement and zoom
      let dx = ndx * SPEED * halfHeight * dt;
      const dy = ndy * SPEED * halfHeight * dt;

      // Going left (backward) is easy/safe, so let the user go fast
      if (dx < 0) dx *= 4;

      updateAndRender(dx, dy);
    }

    requestAnimationFrame(frame);
  }

  // Initial render, then start loop
  renderController = new AbortController();
  render(renderController.signal);
  requestAnimationFrame(frame);
}

main();
