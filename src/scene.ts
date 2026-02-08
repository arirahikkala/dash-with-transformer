import { TrigramModel } from "./trigram";

/** A node in the recursive prediction tree. */
export interface SceneNode {
  /** The predicted character code. */
  charCode: number;
  /** Display label: □ for space, ⏎ for newline, char otherwise. */
  label: string;
  /** Top edge, 0..1 fraction of the widget height (absolute). */
  y0: number;
  /** Bottom edge, 0..1 fraction of the widget height (absolute). */
  y1: number;
  /** Recursively expanded children (next-character predictions). */
  children: SceneNode[];
}

/** Everything needed to render one frame of the widget. */
export interface Scene {
  /** The current sentence prefix. */
  prefix: string;
  /** Crosshairs position, 0..1 in both axes. */
  crosshairs: { x: number; y: number };
  /** Top-level prediction nodes (first column). */
  children: SceneNode[];
  /** Maximum tree depth, for the renderer to size columns. */
  depth: number;
}

/** Printable ASCII (32..126) plus newline (10). */
function isPrintableOrNewline(code: number): boolean {
  return code === 10 || (code >= 32 && code <= 126);
}

function labelFor(code: number): string {
  if (code === 32) return "\u25A1"; // □
  if (code === 10) return "\u23CE"; // ⏎
  return String.fromCharCode(code);
}

const MAX_DEPTH = 8;

/**
 * Recursively expand children for a given prefix context.
 *
 * All characters (printable + newline) are laid out proportionally in
 * y-space, but only those whose absolute probability meets the threshold
 * produce nodes in the output tree.
 */
function expandChildren(
  prefix: string,
  model: TrigramModel,
  parentY0: number,
  parentY1: number,
  parentAbsProb: number,
  threshold: number,
  currentDepth: number,
): SceneNode[] {
  if (currentDepth >= MAX_DEPTH) return [];

  const padded = prefix.padStart(2, " ");
  const context = padded.slice(-2);
  const counts = model.predict(context);

  // Collect printable + newline entries with positive counts.
  const entries: { charCode: number; count: number }[] = [];
  let total = 0;
  for (let c = 0; c < counts.length; c++) {
    if (!isPrintableOrNewline(c)) continue;
    if (counts[c] <= 0) continue;
    entries.push({ charCode: c, count: counts[c] });
    total += counts[c];
  }
  if (total === 0) return [];

  // Sort by charCode (alphabetical top-to-bottom).
  entries.sort((a, b) => a.charCode - b.charCode);

  const parentSpan = parentY1 - parentY0;
  const nodes: SceneNode[] = [];
  let y = parentY0;

  for (const e of entries) {
    const localProb = e.count / total;
    const span = localProb * parentSpan;
    const nodeY0 = y;
    const nodeY1 = y + span;
    y = nodeY1;

    const absProb = parentAbsProb * localProb;
    if (absProb >= threshold) {
      const children = expandChildren(
        prefix + String.fromCharCode(e.charCode),
        model,
        nodeY0,
        nodeY1,
        absProb,
        threshold,
        currentDepth + 1,
      );
      nodes.push({
        charCode: e.charCode,
        label: labelFor(e.charCode),
        y0: nodeY0,
        y1: nodeY1,
        children,
      });
    }
  }

  return nodes;
}

/** Walk the tree to find its maximum depth. */
function maxDepth(nodes: SceneNode[]): number {
  let d = 0;
  for (const node of nodes) {
    const childDepth = maxDepth(node.children);
    d = Math.max(d, 1 + childDepth);
  }
  return d;
}

/**
 * Given a prefix and a trigram model, produce a Scene describing the
 * recursive next-character prediction tree.
 */
export function buildScene(prefix: string, model: TrigramModel): Scene {
  const children = expandChildren(prefix, model, 0, 1, 1.0, 0.01, 0);
  const depth = maxDepth(children);

  return {
    prefix,
    crosshairs: { x: 0.5, y: 0.5 },
    children,
    depth,
  };
}
