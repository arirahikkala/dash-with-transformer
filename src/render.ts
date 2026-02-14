import type { Scene, SceneNode } from "./types";

/** Callbacks for rendering tokens of type T. */
export interface RenderOptions<T> {
  /** Display label for a token. */
  label: (token: T) => string;
  /** CSS color string for a token. */
  color: (token: T) => string;
}

/** Render nodes as right-aligned squares, parent first then children on top. */
function renderNodes<T>(
  ctx: CanvasRenderingContext2D,
  nodes: SceneNode<T>[],
  width: number,
  height: number,
  opts: RenderOptions<T>,
): void {
  for (const node of nodes) {
    const py0 = node.y0 * height;
    const py1 = node.y1 * height;
    const side = py1 - py0;
    const x0 = width - side;

    // Colored square, right-aligned
    ctx.fillStyle = opts.color(node.token);
    ctx.fillRect(x0, py0, side, side);

    // Subtle border at the top
    ctx.strokeStyle = "rgba(255,255,255,0.1)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, py0);
    ctx.lineTo(width, py0);
    ctx.stroke();

    // Label near the left edge of the square
    if (side >= 10) {
      const fontSize = Math.min(Math.max(side * 0.7, 10), 28);
      ctx.font = `${fontSize}px monospace`;
      ctx.fillStyle = "#e0e0e0";
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      ctx.fillText(opts.label(node.token), x0 + 4, py0 + side / 2);
    }

    // Children paint on top (smaller squares nested inside)
    renderNodes(ctx, node.children, width, height, opts);
  }
}

/** Render a Scene onto a 2D canvas. */
export function renderScene<T>(
  ctx: CanvasRenderingContext2D,
  scene: Scene<T>,
  width: number,
  height: number,
  opts: RenderOptions<T>,
): void {
  // Dark background — gaps and out-of-bounds areas show through as dark space.
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, width, height);

  renderNodes(ctx, scene.children, width, height, opts);

  // -- Crosshairs (always at canvas center) --
  const cx = width / 2;
  const cy = height / 2;

  ctx.strokeStyle = "rgba(255, 60, 60, 0.8)";
  ctx.lineWidth = 1.5;

  ctx.beginPath();
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, height);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(0, cy);
  ctx.lineTo(width, cy);
  ctx.stroke();
}
