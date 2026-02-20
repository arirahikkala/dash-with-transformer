import type { Cursor, TokenDisplay } from "./types";
import { predictBytes } from "./backend";
import { fromByteLevelModel } from "./models";
import { normalizeCursor } from "./cursor";
import { buildScene } from "./scene";
import { renderScene } from "./render";

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
  const modelCallPrefix = hashParams.get("modelCallPrefix") ?? "";
  const prefixBytes = new TextEncoder().encode(modelCallPrefix);

  const byteLevelModel =
    prefixBytes.length > 0
      ? (prefix: Uint8Array) => {
          const full = new Uint8Array(prefixBytes.length + prefix.length);
          full.set(prefixBytes);
          full.set(prefix, prefixBytes.length);
          return predictBytes(full);
        }
      : predictBytes;

  const model = fromByteLevelModel(byteLevelModel);
  const display: TokenDisplay<number> = {
    label(cp) {
      if (cp === 32) return "\u25A1"; // □
      if (cp === 10) return "\u23CE"; // ⏎
      return String.fromCodePoint(cp);
    },
    color(cp) {
      const hue = (cp * 137.508) % 360;
      return `hsl(${hue}, 45%, 35%)`;
    },
    prefixToString: (prefix) => String.fromCodePoint(...prefix),
  };

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
      display,
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
    prefixEl.textContent = display.prefixToString(cursor.prefix);

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
