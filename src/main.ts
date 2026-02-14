import type { LanguageModel, TokenProb, Cursor } from "./types";
import { TrigramModel } from "./trigram";
import { normalizeCursor } from "./cursor";
import { buildScene } from "./scene";
import { renderScene, type RenderOptions } from "./render";

// ---------------------------------------------------------------------------
// Trigram adapter
// ---------------------------------------------------------------------------

/** Printable ASCII (32..126) plus newline (10). */
function isPrintableOrNewline(code: number): boolean {
  return code === 10 || (code >= 32 && code <= 126);
}

/** Wrap a TrigramModel as a generic LanguageModel<number>. */
function wrapTrigramModel(trigram: TrigramModel): LanguageModel<number> {
  return async (prefix: readonly number[]): Promise<readonly TokenProb<number>[]> => {
    let context: string;
    if (prefix.length === 0) {
      context = "  ";
    } else if (prefix.length === 1) {
      context = " " + String.fromCharCode(prefix[0]);
    } else {
      const a = prefix[prefix.length - 2];
      const b = prefix[prefix.length - 1];
      context = String.fromCharCode(a) + String.fromCharCode(b);
    }

    const counts = trigram.predict(context);

    const entries: { token: number; count: number }[] = [];
    let total = 0;
    for (let c = 0; c < counts.length; c++) {
      if (!isPrintableOrNewline(c)) continue;
      if (counts[c] <= 0) continue;
      entries.push({ token: c, count: counts[c] });
      total += counts[c];
    }
    if (total === 0) return [];

    entries.sort((a, b) => a.token - b.token);

    return entries.map((e) => ({
      token: e.token,
      probability: e.count / total,
    }));
  };
}

// ---------------------------------------------------------------------------
// Render helpers
// ---------------------------------------------------------------------------

function labelFor(code: number): string {
  if (code === 32) return "\u25A1"; // □
  if (code === 10) return "\u23CE"; // ⏎
  return String.fromCharCode(code);
}

function colorFor(code: number): string {
  const hue = (code * 137.508) % 360;
  return `hsl(${hue}, 45%, 35%)`;
}

function prefixToDisplayString(prefix: readonly number[]): string {
  return prefix
    .map((c) => {
      if (c === 10) return "\u23CE";
      return String.fromCharCode(c);
    })
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
  const resp = await fetch("/model.bin");
  const buffer = await resp.arrayBuffer();
  const trigram = new TrigramModel(buffer);
  const model = wrapTrigramModel(trigram);

  const prefixEl = document.getElementById("prefix-display")!;
  const canvas = document.getElementById("dasher-canvas") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;

  const renderOpts: RenderOptions<number> = {
    label: labelFor,
    color: colorFor,
  };

  // --- Cursor state ---
  const initialPrefix = "";
  const tokenPrefix = Array.from(initialPrefix).map((ch) => ch.charCodeAt(0));
  let cursor: Cursor<number> = { prefix: tokenPrefix, x: 0.5, y: 0.5 };

  // --- Mouse state ---
  let mouseDown = false;
  let mouseX = 0;
  let mouseY = 0;

  function updateMousePos(e: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    // Scale from CSS coords to canvas backing-store coords
    mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
  }

  canvas.addEventListener("mousedown", (e) => {
    mouseDown = true;
    updateMousePos(e);
  });

  canvas.addEventListener("mousemove", (e) => {
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
    await renderScene(ctx, scene, canvas.width, canvas.height, renderOpts, signal);
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
    prefixEl.textContent = prefixToDisplayString(cursor.prefix);

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
      const w = canvas.width;
      const h = canvas.height;

      // Normalized displacement from center: [-1, 1] on each axis
      const ndx = (mouseX - w / 2) / (w / 2);
      const ndy = (mouseY - h / 2) / (h / 2);

      // halfHeight = size of the window in the cursor's local frame
      const halfHeight = 1 - cursor.x;

      // Velocity in local frame, proportional to displacement and zoom
      const dx = ndx * SPEED * halfHeight * dt;
      const dy = ndy * SPEED * halfHeight * dt;

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
