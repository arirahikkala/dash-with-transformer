import type {
  Cursor,
  CDFView,
  WidgetToken,
  SpecialToken,
  LanguageModel,
} from "./types";
import { createBackendClient, fetchSpecialTokens } from "./remote/backend";
import {
  byteOnly,
  forceCleanUtf8,
  fromByteLevelModel,
  interpolate,
  passMinProb,
  trieCache,
} from "./models";
import type { ByteLevelModel } from "./models";
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

/**
 * Encode a prefix string into byte values (0–255) and special-token indices
 * (≥ 256) by splitting on special-token labels.
 */
function encodePrefixWithSpecialTokens(
  text: string,
  specialTokens: readonly SpecialToken[],
): number[] {
  if (specialTokens.length === 0 || text.length === 0) {
    return [...new TextEncoder().encode(text)];
  }
  const escaped = specialTokens.map((st) =>
    st.label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
  );
  const pattern = new RegExp(`(${escaped.join("|")})`);
  const parts = text.split(pattern);
  const labelToIndex = new Map(specialTokens.map((st) => [st.label, st.index]));

  const result: number[] = [];
  for (const part of parts) {
    if (part === "") continue;
    const idx = labelToIndex.get(part);
    if (idx !== undefined) {
      result.push(idx);
    } else {
      result.push(...new TextEncoder().encode(part));
    }
  }
  return result;
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
  const lstmByteLM: LanguageModel<readonly number[]> = trieCache(
    forceCleanUtf8(byteOnly(lstmPredict)),
  );

  // --- Remote side ---

  // Mutable array — fromByteLevelModel iterates it on each call,
  // so in-place mutations are picked up automatically.
  const specialTokens: SpecialToken[] = [];

  const prefixLabelEl = document.getElementById(
    "prefix-label",
  ) as HTMLLabelElement;

  function updatePrefixLabel() {
    const base = "Prefix";
    if (specialTokens.length === 0) {
      prefixLabelEl.childNodes[0].textContent = base + "\n";
    } else {
      const labels = specialTokens.map((st) => st.label).join("  ");
      prefixLabelEl.childNodes[0].textContent =
        base + " (special tokens parsed inline: " + labels + ")\n";
    }
  }

  function createRemoteBLM(): ByteLevelModel {
    const { predictBytes } = createBackendClient(backendUrl);
    return (prefix: readonly number[], minProb: number) => {
      const prefixTokens = encodePrefixWithSpecialTokens(
        remoteModelCallPrefix,
        specialTokens,
      );
      return predictBytes([...prefixTokens, ...prefix], minProb);
    };
  }

  let remoteBLM = createRemoteBLM();

  // --- Stable model (never reassigned) ---
  // passMinProb threads minProb from fromByteLevelModel through to remoteBLM.
  // The adapter closes over mutable sliderValue and remoteBLM so that slider
  // changes and backend reconnects are picked up automatically.
  const model: CDFView<readonly WidgetToken[], WidgetToken> =
    fromByteLevelModel(
      passMinProb((remoteLM) => {
        return async (prefix) => {
          const mixed = interpolate([
            { model: lstmByteLM, weight: 1 - sliderValue },
            { model: remoteLM, weight: sliderValue },
          ]);
          return mixed(prefix);
        };
      })((prefix, minProb) => remoteBLM(prefix, minProb)),
      specialTokens,
    );

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

  // --- Fetch special tokens ---
  async function refreshSpecialTokens() {
    try {
      const tokens = await fetchSpecialTokens(backendUrl);
      specialTokens.length = 0;
      specialTokens.push(...tokens);
    } catch {
      specialTokens.length = 0;
    }
    updatePrefixLabel();
  }

  // Fetch special tokens before first render so the initial prefix is parsed correctly
  await refreshSpecialTokens();

  // --- Config change handlers ---
  async function applyConfigChange() {
    const newUrl = backendUrlInput.value;
    const newPrefix = modelPrefixInput.value;
    if (newUrl === backendUrl && newPrefix === remoteModelCallPrefix) return;
    const urlChanged = newUrl !== backendUrl;
    backendUrl = newUrl;
    remoteModelCallPrefix = newPrefix;
    if (urlChanged) await refreshSpecialTokens();
    remoteBLM = createRemoteBLM();
    updateHash();
    rerender();
  }

  backendUrlInput.addEventListener("blur", applyConfigChange);
  modelPrefixInput.addEventListener("blur", applyConfigChange);

  mixSlider.addEventListener("input", () => {
    sliderValue = parseFloat(mixSlider.value);
    updateHash();
    rerender();
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

  function rerender() {
    renderController?.abort();
    renderController = new AbortController();
    const signal = renderController.signal;
    (async () => {
      const scene = await buildScene(model, cursor, 0.015);
      if (signal.aborted) return;
      await renderScene(
        nodeCtx,
        labelCtx,
        scene,
        nodeCanvas.width,
        nodeCanvas.height,
        signal,
      );
    })();
  }

  // --- Abortable normalizeCursor ---
  let normalizeController: AbortController | null = null;

  async function updateAndRender(dx: number, dy: number) {
    normalizeController?.abort();
    normalizeController = new AbortController();
    const signal = normalizeController.signal;

    const newCursor = await normalizeCursor(
      model,
      {
        prefix: cursor.prefix,
        x: cursor.x + dx,
        y: cursor.y + dy,
      },
      signal,
    );

    if (!newCursor) return;
    cursor = newCursor;
    const text = prefixToString(cursor.prefix);
    if (prefixEl.textContent !== text) {
      prefixEl.textContent = text;
      prefixEl.scrollTop = prefixEl.scrollHeight;
    }

    rerender();
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
  rerender();
  requestAnimationFrame(frame);
}

main();
