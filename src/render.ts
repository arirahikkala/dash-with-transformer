import { Scene, SceneNode } from "./scene";

/** Stable hue derived from a character code (golden-angle spread). */
function nodeColor(charCode: number): string {
  const hue = (charCode * 137.508) % 360;
  return `hsl(${hue}, 45%, 35%)`;
}

/** Render the tree nodes recursively, one column per depth level. */
function renderNodes(
  ctx: CanvasRenderingContext2D,
  nodes: SceneNode[],
  depth: number,
  colWidth: number,
  height: number,
): void {
  for (const node of nodes) {
    const x0 = depth * colWidth;
    const py0 = node.y0 * height;
    const py1 = node.y1 * height;
    const bandHeight = py1 - py0;

    // Colored rectangle
    ctx.fillStyle = nodeColor(node.charCode);
    ctx.fillRect(x0, py0, colWidth, bandHeight);

    // Subtle border at the top of each node
    ctx.strokeStyle = "rgba(255,255,255,0.1)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, py0);
    ctx.lineTo(x0 + colWidth, py0);
    ctx.stroke();

    // Label — only if tall enough
    if (bandHeight >= 10) {
      const fontSize = Math.min(Math.max(bandHeight * 0.7, 10), 28);
      ctx.font = `${fontSize}px monospace`;
      ctx.fillStyle = "#e0e0e0";
      ctx.textBaseline = "middle";
      ctx.textAlign = "center";
      ctx.fillText(node.label, x0 + colWidth / 2, py0 + bandHeight / 2);
    }

    // Recurse into children at the next column
    renderNodes(ctx, node.children, depth + 1, colWidth, height);
  }
}

/** Render a Scene onto a 2D canvas. */
export function renderScene(
  ctx: CanvasRenderingContext2D,
  scene: Scene,
  width: number,
  height: number,
): void {
  // Dark background — gaps between nodes (low-probability chars) show through.
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, width, height);

  // Divide canvas into equal columns, one per tree depth level.
  const cols = Math.max(scene.depth, 1);
  const colWidth = width / cols;

  renderNodes(ctx, scene.children, 0, colWidth, height);

  // -- Crosshairs (drawn last, on top of everything) --
  const cx = scene.crosshairs.x * width;
  const cy = scene.crosshairs.y * height;

  ctx.strokeStyle = "rgba(255, 60, 60, 0.8)";
  ctx.lineWidth = 1.5;

  // Vertical line
  ctx.beginPath();
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, height);
  ctx.stroke();

  // Horizontal line
  ctx.beginPath();
  ctx.moveTo(0, cy);
  ctx.lineTo(width, cy);
  ctx.stroke();
}
