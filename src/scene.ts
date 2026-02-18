/**
 * Scene builder for the Dasher unit-square model.
 *
 * Given a language model, a cursor position, and a minimum displayable
 * height, produces a tree of visible squares for rendering.
 *
 * Three phases:
 *   1. Compute window bounds in cursor-local frame
 *   2. Ascend (exact Rat arithmetic) to find the scene root
 *   3. Descend (float arithmetic) building visible SceneNodes
 */

import {
  type Rat,
  ZERO,
  ONE,
  add,
  mul,
  gte,
  fromFloat,
  toFloat,
} from "./rational";

import type {
  LanguageModel,
  TokenProb,
  Cursor,
  SceneNode,
  Scene,
} from "./types";

// ---------------------------------------------------------------------------
// Phase 2: Ascend to scene root (Rat arithmetic)
// ---------------------------------------------------------------------------

interface AscendResult<T> {
  scenePrefix: readonly T[];
  winTop: number;
  winBot: number;
}

/**
 * Check whether [winTop, winBot] fits inside a single child's range.
 *
 * If a child with probability p contains the full y-range, then
 * p ≥ winBot − winTop = winHeight, so its square (left edge at
 * x = 1 − p) also covers the window's left edge (at x = 1 − winHeight).
 * This guarantees no gap on the left side for any y in the window.
 */
function windowInsideSingleChild<T>(
  dist: readonly TokenProb<T>[],
  winTop: number,
  winBot: number,
): boolean {
  for (const entry of dist) {
    if (entry.end <= entry.start) continue;
    if (winTop >= entry.start && winTop < entry.end) {
      // Found the child containing winTop; does it also contain winBot?
      return winBot <= entry.end;
    }
  }
  return false;
}

/**
 * Walk up from the cursor prefix, transforming window bounds into each
 * parent's frame using exact rational arithmetic, until:
 * (a) the window [winTop, winBot] fits within [0, 1], AND
 * (b) the entire window fits inside a single child of the current node.
 *
 * Condition (b) guarantees that one child's square covers the full
 * window width — no gap can appear on the left side at any y position.
 * The scene root only changes once the covering child is large enough
 * to fill the visible area, eliminating left-side flicker during zoom.
 */
async function ascendToSceneRoot<T>(
  model: LanguageModel<readonly T[], T>,
  prefix: readonly T[],
  winTopFloat: number,
  winBotFloat: number,
  tokEq: (a: T, b: T) => boolean,
): Promise<AscendResult<T>> {
  const mutablePrefix = [...prefix];
  let winTop: Rat = fromFloat(winTopFloat);
  let winBot: Rat = fromFloat(winBotFloat);

  // Check if the window already fits at the starting level with
  // left corners covered by children.
  if (mutablePrefix.length > 0 && gte(winTop, ZERO) && gte(ONE, winBot)) {
    const dist = await model(mutablePrefix, 0, 1, 0);
    if (windowInsideSingleChild(dist, toFloat(winTop), toFloat(winBot))) {
      return {
        scenePrefix: mutablePrefix,
        winTop: toFloat(winTop),
        winBot: toFloat(winBot),
      };
    }
  }

  while (mutablePrefix.length > 0) {
    const lastToken = mutablePrefix.pop()!;
    const dist = await model(mutablePrefix, 0, 1, 0);

    let cumBefore: Rat = ZERO;
    let prob: Rat = ZERO;
    for (const entry of dist) {
      if (tokEq(entry.token, lastToken)) {
        cumBefore = fromFloat(entry.start);
        prob = fromFloat(entry.end - entry.start);
        break;
      }
    }

    // Transform to parent's frame:
    //   parent_winTop = cumBefore + child_winTop * prob
    //   parent_winBot = cumBefore + child_winBot * prob
    winTop = add(cumBefore, mul(winTop, prob));
    winBot = add(cumBefore, mul(winBot, prob));

    // Stop if the window fits vertically AND the children at the
    // window edges are wide enough to cover the left corners.
    // We reuse dist (the parent's distribution) for the check.
    if (gte(winTop, ZERO) && gte(ONE, winBot)) {
      if (windowInsideSingleChild(dist, toFloat(winTop), toFloat(winBot))) {
        break;
      }
    }
  }

  return {
    scenePrefix: mutablePrefix,
    winTop: toFloat(winTop),
    winBot: toFloat(winBot),
  };
}

// ---------------------------------------------------------------------------
// Phase 3: Recursive descent (float arithmetic)
// ---------------------------------------------------------------------------

/**
 * Recursively build visible children for a given prefix.
 *
 * The model handles filtering: we map the visible window back to
 * probability space and pass rangeStart/rangeEnd/minSize so the model
 * only returns entries that are visible and large enough.
 *
 * @param scale   - height of this node in window-relative units
 * @param offset  - y position of this node's top edge in window coords
 * @param absProb - absolute probability of this prefix (in scene root frame)
 */
async function buildChildren<T>(
  model: LanguageModel<readonly T[], T>,
  prefix: readonly T[],
  scale: number,
  offset: number,
  absProb: number,
  minAbsProb: number,
  depth: number,
  maxDepth: number,
): Promise<SceneNode<T>[]> {
  if (depth >= maxDepth) return [];

  // Map window [0,1] back to probability space for filtering.
  // y = offset + cumProb * scale  →  cumProb = (y − offset) / scale
  const rangeStart = -offset / scale;
  const rangeEnd = (1 - offset) / scale;
  const minSize = minAbsProb / absProb;

  const dist = await model(prefix, rangeStart, rangeEnd, minSize);
  if (dist.length === 0) return [];

  const nodes: SceneNode<T>[] = [];

  for (const entry of dist) {
    const p = entry.end - entry.start;
    const y0 = offset + entry.start * scale;
    const y1 = offset + entry.end * scale;
    const childAbsProb = absProb * p;

    const childPrefix = [...prefix, entry.token];
    const children = buildChildren(
      model,
      childPrefix,
      scale * p,
      y0,
      childAbsProb,
      minAbsProb,
      depth + 1,
      maxDepth,
    );

    nodes.push({ token: entry.token, y0, y1, children });
  }

  return nodes;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface BuildSceneOptions {
  /** Maximum recursion depth. Default 100. */
  maxDepth?: number;
  /** Token equality (default: ===). Needed when tokens are objects. */
  tokenEquals?: (a: unknown, b: unknown) => boolean;
}

/**
 * Build a scene from a language model and cursor position.
 *
 * @param model     - The language model
 * @param cursor    - Current cursor position
 * @param minHeight - Minimum displayable height in window coords (e.g. 0.005)
 * @param options   - Optional settings
 */
export async function buildScene<T>(
  model: LanguageModel<readonly T[], T>,
  cursor: Cursor<T>,
  minHeight: number,
  options?: BuildSceneOptions,
): Promise<Scene<T>> {
  const maxD = options?.maxDepth ?? 100;
  const tokEq =
    (options?.tokenEquals as (a: T, b: T) => boolean) ??
    ((a: T, b: T): boolean => a === b);

  // Phase 1: Window bounds in cursor-local frame
  // Window is a square centered on cursor, right edge at x=1
  const halfHeight = 1 - cursor.x;
  const localWinTop = cursor.y - halfHeight;
  const localWinBot = cursor.y + halfHeight;

  // Phase 2: Ascend to scene root
  const { scenePrefix, winTop, winBot } = await ascendToSceneRoot(
    model,
    cursor.prefix,
    localWinTop,
    localWinBot,
    tokEq,
  );

  // Phase 3: Recursive descent from scene root
  // The window spans [winTop, winBot] in the scene root's frame.
  // We need to map that to [0, 1] in window coordinates.
  // windowY = (rootY - winTop) / (winBot - winTop)
  const winHeight = winBot - winTop;
  const scale = 1 / winHeight;
  const offset = -winTop * scale;
  const minAbsProb = minHeight * winHeight;

  const children = await buildChildren(
    model,
    scenePrefix,
    scale,
    offset,
    1.0,
    minAbsProb,
    0,
    maxD,
  );

  return { children };
}
