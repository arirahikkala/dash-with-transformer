/**
 * Cursor normalization for the unit-square model.
 * See CLAUDE.md for the geometric model and core math concepts.
 */

import type { LanguageModel, Cursor } from "./types";
import { first } from "./types";

export interface NormalizeOptions {
  /** Stop descending when prefix reaches this length.  Default 100. */
  maxDepth?: number;
}

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

/**
 * Normalise a cursor so that `prefix` identifies the smallest square in
 * the model's tree that contains the cursor.
 *
 * Key invariant: the *global* position (inside the unit square) is preserved.
 *
 * Special property: if the input adjustment is (0, 0), the prefix is never
 * changed (unless some token has probability exactly 1).
 */
export async function normalizeCursor<T>(
  model: LanguageModel<readonly T[], T>,
  state: Cursor<T>,
  options?: NormalizeOptions,
): Promise<Cursor<T>> {
  const maxDepth = options?.maxDepth ?? 100000;

  const prefix: T[] = [...state.prefix];
  let x = state.x;
  let y = state.y;

  // Upper bound on iterations: we can ascend at most prefix.length times
  // and descend at most maxDepth times, plus a small constant.
  const limit = state.prefix.length + maxDepth + 2;

  for (let iter = 0; iter < limit; iter++) {
    const oob = x < 0 || x >= 1 || y < 0 || y >= 1;

    // --- Phase 1: ascend if out of the current square ---
    if (oob && prefix.length > 0) {
      const lastToken = prefix.pop()!;
      const tokenResult = await first(model(prefix, 0, 1, 0, lastToken));

      let cumBefore = 0;
      let prob = 0;
      if (tokenResult) {
        cumBefore = tokenResult.start;
        prob = tokenResult.end - tokenResult.start;
      }

      // Map back to parent's coordinate frame.
      //   parent_x = (1 − prob) + child_x · prob
      //   parent_y = cumBefore  + child_y · prob
      x = 1 - prob + x * prob;
      y = cumBefore + y * prob;
      continue;
    }

    // --- Clamp at root if still out of bounds ---
    if (oob && prefix.length === 0) {
      if (x < 0) x = 0;
      else if (x >= 1) x = 1 - Number.EPSILON;
      if (y < 0) y = 0;
      else if (y >= 1) y = 1 - Number.EPSILON;
    }

    // --- Phase 2: try to descend into the smallest containing child ---
    if (prefix.length >= maxDepth) break;

    let descended = false;

    // Since children are squares (width = height = probability p), a child
    // can only contain the cursor if p >= 1 − x (its left edge 1−p <= x).
    // Pass this as minProb to avoid materialising small tokens.
    for await (const entry of model(prefix, y, y, Math.max(0, 1 - x))) {
      const p = entry.end - entry.start;
      if (p <= 0) continue;

      //  Child occupies x ∈ [1−p, 1],  y ∈ [start, end]
      //  in the parent's normalised frame.
      const childXLeft = 1 - p;
      const cumProb = entry.start;
      if (x >= childXLeft && y >= cumProb && y < entry.end) {
        prefix.push(entry.token);
        x = (x - childXLeft) / p;
        y = (y - cumProb) / p;
        descended = true;
        break;
      }
    }

    if (!descended) break;
  }

  return { prefix, x, y };
}

// ---------------------------------------------------------------------------
// Helper: convert cursor to global unit-square coordinates
// ---------------------------------------------------------------------------

/**
 * Map a cursor to global (x, y) in the unit square.
 *
 * Useful for verifying that normalisation preserves position.
 */
export async function cursorToGlobal<T>(
  model: LanguageModel<readonly T[], T>,
  state: Cursor<T>,
): Promise<{ x: number; y: number }> {
  let size = 1;
  let top = 0;

  for (let i = 0; i < state.prefix.length; i++) {
    const parentPrefix = state.prefix.slice(0, i);
    const token = state.prefix[i];
    const tokenResult = await first(model(parentPrefix, 0, 1, 0, token));

    let cumBefore = 0;
    let prob = 0;
    if (tokenResult) {
      cumBefore = tokenResult.start;
      prob = tokenResult.end - tokenResult.start;
    }

    top = top + cumBefore * size;
    size = size * prob;
  }

  return {
    x: 1 - size + state.x * size,
    y: top + state.y * size,
  };
}
