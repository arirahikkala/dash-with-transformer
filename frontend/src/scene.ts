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

import type { LanguageModel, Cursor, SceneNode, Scene } from "./types";
import { first } from "./types";

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
 * parent's frame using exact rational arithmetic, until one level past
 * where the window [winTop, winBot] first fits within [0, 1]. This
 * ensures the node that fully covers the view is a child of the scene
 * root (and thus gets rendered), rather than being the scene root itself.
 *
 * Each ascent step uses a specificToken lookup for the child we came
 * from — since we ascend from a normalized cursor, the cursor is always
 * inside that child, never in an ancestor's sibling.
 */
async function ascendToSceneRoot<T>(
  model: LanguageModel<readonly T[], T>,
  prefix: readonly T[],
  winTopFloat: number,
  winBotFloat: number,
): Promise<AscendResult<T>> {
  const mutablePrefix = [...prefix];
  let winTop: Rat = fromFloat(winTopFloat);
  let winBot: Rat = fromFloat(winBotFloat);
  let fitted = gte(winTop, ZERO) && gte(ONE, winBot);

  while (mutablePrefix.length > 0) {
    const lastToken = mutablePrefix.pop()!;
    const tokenResult = await first(model(mutablePrefix, 0, 1, 0, lastToken));

    let cumBefore: Rat = ZERO;
    let prob: Rat = ZERO;
    if (tokenResult) {
      cumBefore = fromFloat(tokenResult.start);
      prob = fromFloat(tokenResult.end - tokenResult.start);
    }

    // Transform to parent's frame:
    //   parent_winTop = cumBefore + child_winTop * prob
    //   parent_winBot = cumBefore + child_winBot * prob
    winTop = add(cumBefore, mul(winTop, prob));
    winBot = add(cumBefore, mul(winBot, prob));

    // Go one level past the first fit so the node covering the entire
    // view becomes a *child* of the scene root and actually gets rendered.
    if (fitted) {
      break;
    }
    if (gte(winTop, ZERO) && gte(ONE, winBot)) {
      fitted = true;
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
async function* buildChildren<T>(
  model: LanguageModel<readonly T[], T>,
  prefix: readonly T[],
  scale: number,
  offset: number,
  absProb: number,
  minAbsProb: number,
  depth: number,
  maxDepth: number,
): AsyncIterable<SceneNode<T>> {
  if (depth >= maxDepth) return;

  // Map window [0,1] back to probability space for filtering.
  // y = offset + cumProb * scale  →  cumProb = (y − offset) / scale
  const rangeStart = -offset / scale;
  const rangeEnd = (1 - offset) / scale;
  const minSize = minAbsProb / absProb;

  for await (const entry of model(prefix, rangeStart, rangeEnd, minSize)) {
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

    yield { token: entry.token, y0, y1, children };
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface BuildSceneOptions {
  /** Maximum recursion depth. Default 100. */
  maxDepth?: number;
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

  return { children };
}
