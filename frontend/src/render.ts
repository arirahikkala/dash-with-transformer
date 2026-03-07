import type { Scene, SceneNode, WidgetToken } from "./types";

/** A label's bounding box, used to avoid overlapping labels. */
interface LabelExtent {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

/** Display label for a widget token. */
function label(token: WidgetToken): string {
  if (token.type === "special") return token.label;
  const cp = token.codepoint;
  if (cp === 32) return "\u25A1"; // □
  if (cp === 10) return "\u23CE"; // ⏎
  return String.fromCodePoint(cp);
}

/** CSS color for a widget token at a given depth. */
function color(token: WidgetToken, depth: number): string {
  if (token.type === "special") return "hsl(15, 60%, 45%)";
  const cp = token.codepoint;
  // Space: white
  if (cp === 32) return "#ffffff";
  // Lowercase a-z: hue cycles by depth between light green (120) and cyan (180)
  if (cp >= 97 && cp <= 122) {
    const hue = ((depth * 137.508) % 60) + 120;
    return `hsl(${hue}, 40%, 85%)`;
  }
  // Uppercase A-Z: same cycle, slightly darker
  if (cp >= 65 && cp <= 90) {
    const hue = ((depth * 137.508) % 60) + 120;
    return `hsl(${hue}, 40%, 75%)`;
  }
  // Punctuation: dark green
  if (
    (cp >= 33 && cp <= 47) ||
    (cp >= 58 && cp <= 64) ||
    (cp >= 91 && cp <= 96) ||
    (cp >= 123 && cp <= 126)
  ) {
    return `hsl(140, 50%, 30%)`;
  }
  // Everything else (digits, newline, non-ASCII): yellow
  return `hsl(50, 90%, 45%)`;
}

/** Render a node's colored square and top border to the node canvas. */
function renderNodeRect(
  ctx: CanvasRenderingContext2D,
  token: WidgetToken,
  x0: number,
  py0: number,
  side: number,
  nodeWidth: number,
  depth: number,
): void {
  ctx.fillStyle = color(token, depth);
  ctx.fillRect(x0, py0, side, side);

  ctx.strokeStyle = "rgba(0,0,0,0.1)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, py0);
  ctx.lineTo(nodeWidth, py0);
  ctx.stroke();
}

/**
 * Render a node's label (and optional connector spline) to the label canvas.
 * Returns the new extent for this label, or null if the node was too small to label.
 */
function renderNodeLabel(
  ctx: CanvasRenderingContext2D,
  token: WidgetToken,
  x0: number,
  py0: number,
  py1: number,
  side: number,
  height: number,
  labelExtents: LabelExtent[],
): LabelExtent | null {
  if (side < 10) return null;

  const fontSize = Math.min(Math.max(side * 0.7, 10), 28);
  ctx.font = `${fontSize}px monospace`;
  ctx.fillStyle = "#000";
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  const labelText = label(token);
  const textWidth = ctx.measureText(labelText).width;
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
  const ly0 = labelY - halfFont;
  const ly1 = labelY + halfFont;

  // Push labelX as far left as possible while staying in node bounds
  // and not overlapping any ancestor label extent
  let labelX = x0 + 4;
  for (const ext of labelExtents) {
    if (ly0 < ext.y1 && ly1 > ext.y0) {
      labelX = Math.max(labelX, ext.x1);
    }
  }

  // S-spline from parent label to this label when child doesn't
  // vertically overlap parent and is left enough to warrant a connector
  const parentExtent = labelExtents[labelExtents.length - 1] ?? null;
  if (
    parentExtent &&
    !(ly0 < parentExtent.y1 && ly1 > parentExtent.y0) &&
    labelX < parentExtent.x0 + 50
  ) {
    const startX = parentExtent.x0;
    const startY = (parentExtent.y0 + parentExtent.y1) / 2;
    const endX = labelX + textWidth;
    const endY = labelY;
    const bulge = 200;
    const dist = parentExtent.x0 - labelX;
    const alpha = 0.25 * Math.min(1, dist / 50);
    ctx.strokeStyle = `rgba(0,0,0,${alpha})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.bezierCurveTo(startX + bulge, startY, endX - bulge, endY, endX, endY);
    ctx.stroke();
  }

  ctx.fillStyle = "#000";
  ctx.fillText(labelText, labelX, labelY);
  return { x0: labelX, x1: labelX + textWidth + 1, y0: ly0, y1: ly1 };
}

/** Render nodes as right-aligned squares, parent first then children on top. */
async function renderNodes(
  nodeCtx: CanvasRenderingContext2D,
  labelCtx: CanvasRenderingContext2D,
  nodes: AsyncIterable<SceneNode<WidgetToken>>,
  nodeWidth: number,
  height: number,
  signal: AbortSignal,
  labelExtents: LabelExtent[],
  depth: number,
): Promise<void> {
  for await (const node of nodes) {
    if (signal.aborted) return;
    const py0 = node.y0 * height;
    const py1 = node.y1 * height;
    const side = py1 - py0;
    const x0 = nodeWidth - side;

    renderNodeRect(nodeCtx, node.token, x0, py0, side, nodeWidth, depth);

    const thisExtent = renderNodeLabel(
      labelCtx,
      node.token,
      x0,
      py0,
      py1,
      side,
      height,
      labelExtents,
    );
    const childExtents = thisExtent
      ? [...labelExtents, thisExtent]
      : labelExtents;

    renderNodes(
      nodeCtx,
      labelCtx,
      node.children,
      nodeWidth,
      height,
      signal,
      childExtents,
      depth + 1,
    );
  }
}

/** Render a Scene onto dual canvases (nodes + labels). */
export async function renderScene(
  nodeCtx: CanvasRenderingContext2D,
  labelCtx: CanvasRenderingContext2D,
  scene: Scene<WidgetToken>,
  nodeWidth: number,
  height: number,
  signal: AbortSignal,
): Promise<void> {
  // Light background on node canvas
  nodeCtx.fillStyle = "#e8e8e8";
  nodeCtx.fillRect(0, 0, nodeWidth, height);

  // Label canvas is transparent and on top so that both canvases can be rendered async
  labelCtx.clearRect(0, 0, labelCtx.canvas.width, height);

  await renderNodes(
    nodeCtx,
    labelCtx,
    scene.children,
    nodeWidth,
    height,
    signal,
    [],
    scene.prefixLength,
  );
}
