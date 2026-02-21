import type { Scene, SceneNode } from "./types";

/** Callbacks for rendering tokens of type T. */
export interface RenderOptions<T> {
  /** Display label for a token. */
  label: (token: T) => string;
  /** CSS color string for a token. */
  color: (token: T) => string;
}

/** Render nodes as right-aligned squares, parent first then children on top. */
async function renderNodes<T>(
  nodeCtx: CanvasRenderingContext2D,
  labelCtx: CanvasRenderingContext2D,
  nodes: AsyncIterable<SceneNode<T>>,
  nodeWidth: number,
  height: number,
  opts: RenderOptions<T>,
  signal: AbortSignal,
  labelMinX: number,
): Promise<void> {
  for await (const node of nodes) {
    if (signal.aborted) return;
    const py0 = node.y0 * height;
    const py1 = node.y1 * height;
    const side = py1 - py0;
    const x0 = nodeWidth - side;

    // Colored square, right-aligned (node canvas)
    nodeCtx.fillStyle = opts.color(node.token);
    nodeCtx.fillRect(x0, py0, side, side);

    // Subtle border at the top (node canvas)
    nodeCtx.strokeStyle = "rgba(0,0,0,0.1)";
    nodeCtx.lineWidth = 1;
    nodeCtx.beginPath();
    nodeCtx.moveTo(x0, py0);
    nodeCtx.lineTo(nodeWidth, py0);
    nodeCtx.stroke();

    // Label on label canvas, pushed right past parent's label
    let childLabelMinX = labelMinX;
    if (side >= 10) {
      const fontSize = Math.min(Math.max(side * 0.7, 10), 28);
      labelCtx.font = `${fontSize}px monospace`;
      labelCtx.fillStyle = "#000";
      labelCtx.textBaseline = "middle";
      labelCtx.textAlign = "left";
      const labelText = opts.label(node.token);
      const labelX = Math.max(x0 + 4, labelMinX);
      const pad = 4;
      const halfFont = fontSize / 2;
      const centerY = py0 + side / 2;
      // Clamp to screen, then clamp to node (node bounds always win)
      const screenClamped = Math.max(
        pad + halfFont,
        Math.min(height - pad - halfFont, centerY),
      );
      const labelY = Math.max(
        py0 + halfFont + pad,
        Math.min(py1 - halfFont - pad, screenClamped),
      );
      labelCtx.fillText(labelText, labelX, labelY);
      childLabelMinX = labelX + labelCtx.measureText(labelText).width + 1;
    }

    // Children paint on top (smaller squares nested inside)
    renderNodes(
      nodeCtx,
      labelCtx,
      node.children,
      nodeWidth,
      height,
      opts,
      signal,
      childLabelMinX,
    );
  }
}

/** Render a Scene onto dual canvases (nodes + labels). */
export async function renderScene<T>(
  nodeCtx: CanvasRenderingContext2D,
  labelCtx: CanvasRenderingContext2D,
  scene: Scene<T>,
  nodeWidth: number,
  height: number,
  opts: RenderOptions<T>,
  signal: AbortSignal,
): Promise<void> {
  // Light background on node canvas
  nodeCtx.fillStyle = "#e8e8e8";
  nodeCtx.fillRect(0, 0, nodeWidth, height);

  // Clear label canvas to transparent
  labelCtx.clearRect(0, 0, labelCtx.canvas.width, height);

  await renderNodes(
    nodeCtx,
    labelCtx,
    scene.children,
    nodeWidth,
    height,
    opts,
    signal,
    0,
  );
}
