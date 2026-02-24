import type { Cursor, LanguageModel, PlainTokenProb } from "./types";
import { createBackendClient } from "./backend";
import { forceCleanUtf8, fromByteLevelModel, trieCache } from "./models";
import { normalizeCursor } from "./cursor";
import { buildScene } from "./scene";
import { renderScene } from "./render";
import { loadSmolLM } from "./smollm";
import { loadModel, softmax } from "./lstm";

function prefixToString(prefix: readonly number[]): string {
  return String.fromCodePoint(...prefix);
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

function createModel(
  backendUrl: string,
  modelCallPrefix: string,
): LanguageModel<readonly number[], number> {
  const { predictBytes } = createBackendClient(backendUrl);
  const prefixBytes = new TextEncoder().encode(modelCallPrefix);

  const byteLevelModel =
    prefixBytes.length > 0
      ? (prefix: Uint8Array, minProb: number) => {
          const full = new Uint8Array(prefixBytes.length + prefix.length);
          full.set(prefixBytes);
          full.set(prefix, prefixBytes.length);
          return predictBytes(full, minProb);
        }
      : predictBytes;

  return fromByteLevelModel(byteLevelModel);
}

async function main() {
  const hashParams = new URLSearchParams(window.location.hash.slice(1));
  let backendUrl = hashParams.get("backendUrl") ?? "http://localhost:8000";
  let modelCallPrefix = hashParams.get("modelCallPrefix") ?? "";
  let mode = hashParams.get("mode") ?? "backend";

  let model = createModel(backendUrl, modelCallPrefix);

  // --- Config inputs ---
  const modeSelect = document.getElementById(
    "inference-mode",
  ) as HTMLSelectElement;
  const backendUrlLabel = document.getElementById(
    "backend-url-label",
  ) as HTMLElement;
  const webgpuStatusEl = document.getElementById(
    "webgpu-status",
  ) as HTMLElement;
  const backendUrlInput = document.getElementById(
    "backend-url",
  ) as HTMLInputElement;
  const modelPrefixInput = document.getElementById(
    "model-prefix",
  ) as HTMLTextAreaElement;
  const prefixLabel = document.getElementById("prefix-label") as HTMLElement;

  backendUrlInput.value = backendUrl;
  modelPrefixInput.value = modelCallPrefix;
  modeSelect.value = mode;

  // WebGPU model caching
  let webgpuModel: LanguageModel<readonly number[], number> | null = null;
  let webgpuLoadPromise: Promise<
    LanguageModel<readonly number[], number>
  > | null = null;

  // LSTM model caching
  let lstmModel: LanguageModel<readonly number[], number> | null = null;
  let lstmLoadPromise: Promise<
    LanguageModel<readonly number[], number>
  > | null = null;

  function updateModeUI() {
    const isBackend = mode === "backend";
    backendUrlLabel.style.display = isBackend ? "" : "none";
    prefixLabel.style.display = isBackend ? "" : "none";
    webgpuStatusEl.style.display = isBackend ? "none" : "";
  }

  function updateHash() {
    const params = new URLSearchParams();
    if (mode !== "backend") params.set("mode", mode);
    if (backendUrl !== "http://localhost:8000") {
      params.set("backendUrl", backendUrl);
    }
    if (modelCallPrefix) params.set("modelCallPrefix", modelCallPrefix);
    window.location.hash = params.toString();
  }

  async function loadWebGPUModel(): Promise<
    LanguageModel<readonly number[], number>
  > {
    if (!webgpuLoadPromise) {
      webgpuLoadPromise = loadSmolLM((msg) => {
        webgpuStatusEl.textContent = msg;
      });
    }
    webgpuModel = await webgpuLoadPromise;
    return webgpuModel;
  }

  async function loadLSTMModel(): Promise<
    LanguageModel<readonly number[], number>
  > {
    if (!lstmLoadPromise) {
      lstmLoadPromise = (async () => {
        webgpuStatusEl.textContent = "Loading LSTM model\u2026";
        const base = import.meta.env.BASE_URL.replace(/\/$/, "");
        const lstm = await loadModel(base, true);
        webgpuStatusEl.textContent = "Ready!";

        const plainModel = async (
          prefix: Uint8Array,
        ): Promise<readonly PlainTokenProb<number>[]> => {
          lstm.reset();
          const logits = lstm.forward(prefix);
          const probs = softmax(logits);
          const result: PlainTokenProb<number>[] = [];
          for (let i = 0; i < probs.length; i++) {
            if (probs[i] > 0) {
              result.push({ token: i, probability: probs[i] });
            }
          }
          return result;
        };

        const cleanModel = trieCache(forceCleanUtf8(plainModel));

        return fromByteLevelModel(async (prefix: Uint8Array) => {
          const dist = await cleanModel(prefix);
          const result: number[] = new Array(256).fill(0);
          for (const { token, probability } of dist) {
            result[token] = probability;
          }
          return result;
        });
      })();
    }
    lstmModel = await lstmLoadPromise;
    return lstmModel;
  }

  function applyConfigChange() {
    const newUrl = backendUrlInput.value;
    const newPrefix = modelPrefixInput.value;
    if (newUrl === backendUrl && newPrefix === modelCallPrefix) return;
    backendUrl = newUrl;
    modelCallPrefix = newPrefix;
    updateHash();

    if (mode === "backend") {
      model = createModel(backendUrl, modelCallPrefix);
      renderController?.abort();
      renderController = new AbortController();
      render(renderController.signal);
    }
  }

  backendUrlInput.addEventListener("blur", applyConfigChange);
  modelPrefixInput.addEventListener("blur", applyConfigChange);

  modeSelect.addEventListener("change", async () => {
    mode = modeSelect.value;
    updateModeUI();
    updateHash();
    if (mode === "webgpu") {
      const loaded = await loadWebGPUModel();
      if (mode !== "webgpu") return;
      model = loaded;
    } else if (mode === "lstm") {
      const loaded = await loadLSTMModel();
      if (mode !== "lstm") return;
      model = loaded;
    } else {
      model = createModel(backendUrl, modelCallPrefix);
    }
    cursor = { prefix: [], x: 0.5, y: 0.5 };
    renderController?.abort();
    renderController = new AbortController();
    render(renderController.signal);
  });

  updateModeUI();

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
  let cursor: Cursor<number> = { prefix: [], x: 0.5, y: 0.5 };

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

  // Initialize model based on mode
  if (mode === "webgpu") {
    model = await loadWebGPUModel();
  } else if (mode === "lstm") {
    model = await loadLSTMModel();
  }

  // Initial render, then start loop
  renderController = new AbortController();
  render(renderController.signal);
  requestAnimationFrame(frame);
}

main();
