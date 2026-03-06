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

/** Render nodes as right-aligned squares, parent first then children on top. */
async function renderNodes(
  nodeCtx: CanvasRenderingContext2D,
  labelCtx: CanvasRenderingContext2D,
  nodes: AsyncIterable<SceneNode<WidgetToken>>,
  nodeWidth: number,
  height: number,
  signal: AbortSignal,
  labelExtents: LabelExtent[],
  parentExtent: LabelExtent | null,
  depth: number,
): Promise<void> {
  for await (const node of nodes) {
    if (signal.aborted) return;
    const py0 = node.y0 * height;
    const py1 = node.y1 * height;
    const side = py1 - py0;
    const x0 = nodeWidth - side;

    // Colored square, right-aligned (node canvas)
    nodeCtx.fillStyle = color(node.token, depth);
    nodeCtx.fillRect(x0, py0, side, side);

    // Subtle border at the top (node canvas)
    nodeCtx.strokeStyle = "rgba(0,0,0,0.1)";
    nodeCtx.lineWidth = 1;
    nodeCtx.beginPath();
    nodeCtx.moveTo(x0, py0);
    nodeCtx.lineTo(nodeWidth, py0);
    nodeCtx.stroke();

    // Label on label canvas, pushed right only where ancestors vertically overlap
    let childExtents = labelExtents;
    let thisExtent: LabelExtent | null = parentExtent;
    if (side >= 10) {
      const fontSize = Math.min(Math.max(side * 0.7, 10), 28);
      labelCtx.font = `${fontSize}px monospace`;
      labelCtx.fillStyle = "#000";
      labelCtx.textBaseline = "middle";
      labelCtx.textAlign = "left";
      const labelText = label(node.token);
      const textWidth = labelCtx.measureText(labelText).width;
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
        // Check vertical overlap
        if (ly0 < ext.y1 && ly1 > ext.y0) {
          labelX = Math.max(labelX, ext.x1);
        }
      }

      // S-spline from parent label to this label when child is left of parent
      if (parentExtent && labelX < parentExtent.x0) {
        const startX = parentExtent.x0;
        const startY = (parentExtent.y0 + parentExtent.y1) / 2;
        const endX = labelX + textWidth;
        const endY = labelY;
        const bulge = 200;
        labelCtx.strokeStyle = "rgba(0,0,0,0.25)";
        labelCtx.lineWidth = 1;
        labelCtx.beginPath();
        labelCtx.moveTo(startX, startY);
        labelCtx.bezierCurveTo(
          startX + bulge,
          startY,
          endX - bulge,
          endY,
          endX,
          endY,
        );
        labelCtx.stroke();
      }

      labelCtx.fillStyle = "#000";
      labelCtx.fillText(labelText, labelX, labelY);
      thisExtent = { x0: labelX, x1: labelX + textWidth + 1, y0: ly0, y1: ly1 };
      childExtents = [...labelExtents, thisExtent];
    }

    // Children paint on top (smaller squares nested inside)
    renderNodes(
      nodeCtx,
      labelCtx,
      node.children,
      nodeWidth,
      height,
      signal,
      childExtents,
      thisExtent,
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
    null,
    scene.prefixLength,
  );
}
