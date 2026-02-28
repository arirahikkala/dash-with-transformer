import type { Scene, SceneNode, WidgetToken } from "./types";

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
  labelMinX: number,
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

    // Label on label canvas, pushed right past parent's label
    let childLabelMinX = labelMinX;
    if (side >= 10) {
      const fontSize = Math.min(Math.max(side * 0.7, 10), 28);
      labelCtx.font = `${fontSize}px monospace`;
      labelCtx.fillStyle = "#000";
      labelCtx.textBaseline = "middle";
      labelCtx.textAlign = "left";
      const labelText = label(node.token);
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
      signal,
      childLabelMinX,
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

  // Clear label canvas to transparent
  labelCtx.clearRect(0, 0, labelCtx.canvas.width, height);

  await renderNodes(
    nodeCtx,
    labelCtx,
    scene.children,
    nodeWidth,
    height,
    signal,
    0,
    scene.prefixLength,
  );
}
