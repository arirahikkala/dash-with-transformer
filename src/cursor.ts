/**
 * Cursor normalization for the Dasher unit-square model.
 * See CLAUDE.md for the geometric model and core math concepts.
 */

import {
  type Rat,
  ZERO,
  ONE,
  add,
  sub,
  mul,
  div,
  lt,
  gte,
  fromFloat,
  toFloat,
} from "./rational";

import type { LanguageModel, Cursor } from "./types";

export interface NormalizeOptions<T> {
  /** Stop descending when prefix reaches this length.  Default 100. */
  maxDepth?: number;
  /** Token equality (default: ===). Needed when tokens are objects. */
  tokenEquals?: (a: T, b: T) => boolean;
}

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

/**
 * Normalise a cursor so that `prefix` identifies the smallest square in
 * the model's tree that contains the cursor.
 *
 * Key invariant: the *global* position (inside the unit square) is preserved
 * exactly in rational arithmetic.  The returned float (x, y) is the nearest
 * float64 to the exact rational result.
 *
 * Special property: if the input adjustment is (0, 0), the prefix is never
 * changed (unless some token has probability exactly 1).
 */
export function normalizeCursor<T>(
  model: LanguageModel<T>,
  state: Cursor<T>,
  options?: NormalizeOptions<T>,
): Cursor<T> {
  const maxDepth = options?.maxDepth ?? 100;
  const tokEq = options?.tokenEquals ?? ((a: T, b: T): boolean => a === b);

  const prefix: T[] = [...state.prefix];
  let x: Rat = fromFloat(state.x);
  let y: Rat = fromFloat(state.y);

  // Upper bound on iterations: we can ascend at most prefix.length times
  // and descend at most maxDepth times, plus a small constant.
  const limit = state.prefix.length + maxDepth + 2;

  for (let iter = 0; iter < limit; iter++) {
    const oob = lt(x, ZERO) || gte(x, ONE) || lt(y, ZERO) || gte(y, ONE);

    // --- Phase 1: ascend if out of the current square ---
    if (oob && prefix.length > 0) {
      const lastToken = prefix.pop()!;
      const dist = model(prefix);

      let cumBefore: Rat = ZERO;
      let prob: Rat = ZERO;
      for (const entry of dist) {
        if (tokEq(entry.token, lastToken)) {
          prob = fromFloat(entry.probability);
          break;
        }
        cumBefore = add(cumBefore, fromFloat(entry.probability));
      }

      // Map back to parent's coordinate frame.
      //   parent_x = (1 − prob) + child_x · prob
      //   parent_y = cumBefore  + child_y · prob
      x = add(sub(ONE, prob), mul(x, prob));
      y = add(cumBefore, mul(y, prob));
      continue;
    }

    // --- Clamp at root if still out of bounds ---
    if (oob && prefix.length === 0) {
      if (lt(x, ZERO)) x = ZERO;
      else if (gte(x, ONE)) x = fromFloat(1 - Number.EPSILON);
      if (lt(y, ZERO)) y = ZERO;
      else if (gte(y, ONE)) y = fromFloat(1 - Number.EPSILON);
    }

    // --- Phase 2: try to descend into the smallest containing child ---
    if (prefix.length >= maxDepth) break;

    const dist = model(prefix);
    if (dist.length === 0) break;

    let cumProb: Rat = ZERO;
    let descended = false;

    for (const entry of dist) {
      if (entry.probability <= 0) continue;
      const p = fromFloat(entry.probability);

      //  Child occupies x ∈ [1−p, 1],  y ∈ [cumProb, cumProb+p]
      //  in the parent's normalised frame.
      const childXLeft = sub(ONE, p);
      if (gte(x, childXLeft) && gte(y, cumProb) && lt(y, add(cumProb, p))) {
        prefix.push(entry.token);
        x = div(sub(x, childXLeft), p);
        y = div(sub(y, cumProb), p);
        descended = true;
        break;
      }
      cumProb = add(cumProb, p);
    }

    if (!descended) break;
  }

  return { prefix, x: toFloat(x), y: toFloat(y) };
}

// ---------------------------------------------------------------------------
// Helper: convert cursor to global unit-square coordinates
// ---------------------------------------------------------------------------

/**
 * Map a cursor to global (x, y) in the unit square.
 *
 * Uses exact rational arithmetic internally, returns float64.
 * Useful for verifying that normalisation preserves position.
 */
export function cursorToGlobal<T>(
  model: LanguageModel<T>,
  state: Cursor<T>,
  tokenEquals?: (a: T, b: T) => boolean,
): { x: number; y: number } {
  const eq = tokenEquals ?? ((a: T, b: T): boolean => a === b);

  let size: Rat = ONE;
  let top: Rat = ZERO;

  for (let i = 0; i < state.prefix.length; i++) {
    const parentPrefix = state.prefix.slice(0, i);
    const dist = model(parentPrefix);
    const token = state.prefix[i];

    let cumBefore: Rat = ZERO;
    let prob: Rat = ZERO;
    for (const entry of dist) {
      if (eq(entry.token, token)) {
        prob = fromFloat(entry.probability);
        break;
      }
      cumBefore = add(cumBefore, fromFloat(entry.probability));
    }

    top = add(top, mul(cumBefore, size));
    size = mul(size, prob);
  }

  return {
    x: toFloat(add(sub(ONE, size), mul(fromFloat(state.x), size))),
    y: toFloat(add(top, mul(fromFloat(state.y), size))),
  };
}
