import { Scene, SceneNode } from "./scene";

/** Stable hue derived from a character code (golden-angle spread). */
function nodeColor(charCode: number): string {
  const hue = (charCode * 137.508) % 360;
  return `hsl(${hue}, 45%, 35%)`;
}

/** Render nodes as right-aligned squares, parent first then children on top. */
function renderNodes(
  ctx: CanvasRenderingContext2D,
  nodes: SceneNode[],
  width: number,
  height: number,
): void {
  for (const node of nodes) {
    const py0 = node.y0 * height;
    const py1 = node.y1 * height;
    const side = py1 - py0;
    const x0 = width - side;

    // Colored square, right-aligned
    ctx.fillStyle = nodeColor(node.charCode);
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
      ctx.fillText(node.label, x0 + 4, py0 + side / 2);
    }

    // Children paint on top (smaller squares nested inside)
    renderNodes(ctx, node.children, width, height);
  }
}

/** Render a Scene onto a 2D canvas. */
export function renderScene(
  ctx: CanvasRenderingContext2D,
  scene: Scene,
  width: number,
  height: number,
): void {
  // Dark background — gaps show through as dark space.
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, width, height);

  renderNodes(ctx, scene.children, width, height);

  // -- Crosshairs (drawn last, on top of everything) --
  const cx = scene.crosshairs.x * width;
  const cy = scene.crosshairs.y * height;

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
