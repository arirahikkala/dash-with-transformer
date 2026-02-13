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

import { type LanguageModel, type Cursor } from "./cursor";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A node in the recursive prediction tree. */
export interface SceneNode<T> {
  /** The token this node represents. */
  token: T;
  /** Top edge in window-relative coordinates [0,1]. */
  y0: number;
  /** Bottom edge in window-relative coordinates [0,1]. */
  y1: number;
  /** Recursively expanded children (next-token predictions). */
  children: SceneNode<T>[];
}

/** Everything needed to render one frame of the widget. */
export interface Scene<T> {
  /** Top-level prediction nodes. */
  children: SceneNode<T>[];
  /** Maximum tree depth, for the renderer to size columns. */
  depth: number;
}

// ---------------------------------------------------------------------------
// Phase 2: Ascend to scene root (Rat arithmetic)
// ---------------------------------------------------------------------------

interface AscendResult<T> {
  scenePrefix: readonly T[];
  winTop: number;
  winBot: number;
}

/**
 * Walk up from the cursor prefix, transforming window bounds into each
 * parent's frame using exact rational arithmetic, until the window
 * [winTop, winBot] fits entirely within [0, 1] or we reach the root.
 */
function ascendToSceneRoot<T>(
  model: LanguageModel<T>,
  prefix: readonly T[],
  winTopFloat: number,
  winBotFloat: number,
  tokEq: (a: T, b: T) => boolean,
): AscendResult<T> {
  const mutablePrefix = [...prefix];
  let winTop: Rat = fromFloat(winTopFloat);
  let winBot: Rat = fromFloat(winBotFloat);

  while (mutablePrefix.length > 0) {
    // If window fits inside [0, 1], we're done.
    // winTop >= 0 AND winBot <= 1  (i.e. 1 >= winBot)
    if (gte(winTop, ZERO) && gte(ONE, winBot)) break;

    const lastToken = mutablePrefix.pop()!;
    const dist = model(mutablePrefix);

    let cumBefore: Rat = ZERO;
    let prob: Rat = ZERO;
    for (const entry of dist) {
      if (tokEq(entry.token, lastToken)) {
        prob = fromFloat(entry.probability);
        break;
      }
      cumBefore = add(cumBefore, fromFloat(entry.probability));
    }

    // Transform to parent's frame:
    //   parent_winTop = cumBefore + child_winTop * prob
    //   parent_winBot = cumBefore + child_winBot * prob
    winTop = add(cumBefore, mul(winTop, prob));
    winBot = add(cumBefore, mul(winBot, prob));
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
 * @param scale   - height of this node in window-relative units
 * @param offset  - y position of this node's top edge in window coords
 * @param absProb - absolute probability of this prefix (in scene root frame)
 */
function buildChildren<T>(
  model: LanguageModel<T>,
  prefix: readonly T[],
  scale: number,
  offset: number,
  absProb: number,
  minAbsProb: number,
  depth: number,
  maxDepth: number,
): SceneNode<T>[] {
  if (depth >= maxDepth) return [];

  const dist = model(prefix);
  if (dist.length === 0) return [];

  const nodes: SceneNode<T>[] = [];
  let cumProb = 0;

  for (const entry of dist) {
    const p = entry.probability;
    if (p <= 0) continue;

    const y0 = offset + cumProb * scale;
    const y1 = offset + (cumProb + p) * scale;
    cumProb += p;

    // Cull if entirely off-screen
    if (y1 <= 0 || y0 >= 1) continue;

    // Cull if too small
    const childAbsProb = absProb * p;
    if (childAbsProb < minAbsProb) continue;

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
// Depth measurement
// ---------------------------------------------------------------------------

/** Walk the tree to find its maximum depth. */
function maxDepth<T>(nodes: SceneNode<T>[]): number {
  let d = 0;
  for (const node of nodes) {
    const childDepth = maxDepth(node.children);
    d = Math.max(d, 1 + childDepth);
  }
  return d;
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
export function buildScene<T>(
  model: LanguageModel<T>,
  cursor: Cursor<T>,
  minHeight: number,
  options?: BuildSceneOptions,
): Scene<T> {
  const maxD = options?.maxDepth ?? 100;
  const tokEq = (options?.tokenEquals as (a: T, b: T) => boolean) ??
    ((a: T, b: T): boolean => a === b);

  // Phase 1: Window bounds in cursor-local frame
  // Window is a square centered on cursor, right edge at x=1
  const halfHeight = 1 - cursor.x;
  const localWinTop = cursor.y - halfHeight;
  const localWinBot = cursor.y + halfHeight;

  // Phase 2: Ascend to scene root
  const { scenePrefix, winTop, winBot } = ascendToSceneRoot(
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

  const children = buildChildren(
    model,
    scenePrefix,
    scale,
    offset,
    1.0,
    minAbsProb,
    0,
    maxD,
  );

  return { children, depth: maxDepth(children) };
}
