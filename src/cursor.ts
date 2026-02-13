/**
 * Cursor normalization for the Dasher unit-square model.
 *
 * Geometric model
 * ===============
 * An autoregressive language model induces a recursive tiling of the unit
 * square [0,1]×[0,1].  Every text prefix p maps to a *square*:
 *
 *   width = height = P(p)          (joint probability of the prefix)
 *   right edge at x = 1
 *   left  edge at x = 1 − P(p)
 *   y-position determined by cumulative conditional probabilities
 *
 * Inside each square, the next-token distribution carves out child squares
 * stacked vertically (in the order the model returns them).  A child for
 * token c with conditional probability p_c occupies, in the *parent's*
 * normalised [0,1]×[0,1] coordinate frame:
 *
 *   x ∈ [1 − p_c,  1]
 *   y ∈ [cumBefore,  cumBefore + p_c]
 *
 * Because every child is narrower than its parent (unless p_c = 1), there is
 * a "gap" on the left side of every square that no child covers.
 *
 * Cursor state
 * ============
 * A cursor is (prefix, x, y) where x and y are in the current square's
 * normalised frame:  x = 0 is the left edge, y = 0 is the top edge.
 *
 * The normaliser finds the *smallest* (deepest) square that contains the
 * cursor, adjusting the prefix and (x,y) so that the global position is
 * preserved.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * One entry in a next-token probability distribution.
 * The order of entries determines the top-to-bottom layout.
 */
export interface TokenProb<T> {
  readonly token: T;
  readonly probability: number;
}

/**
 * A language model: given a prefix, return the next-token distribution.
 * Probabilities must be positive and sum to 1.
 */
export type LanguageModel<T> = (prefix: readonly T[]) => readonly TokenProb<T>[];

/** Cursor state: a discrete prefix plus a continuous adjustment. */
export interface CursorState<T> {
  readonly prefix: readonly T[];
  /** 0 = left edge of the prefix's square, 1 = right edge. */
  readonly x: number;
  /** 0 = top edge of the prefix's square, 1 = bottom edge. */
  readonly y: number;
}

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
 * up to floating-point precision.
 *
 * Special property: if the input adjustment is (0, 0), the prefix is never
 * changed (unless some token has probability exactly 1).
 */
export function normalizeCursor<T>(
  model: LanguageModel<T>,
  state: CursorState<T>,
  options?: NormalizeOptions<T>,
): CursorState<T> {
  const maxDepth = options?.maxDepth ?? 100;
  const eq = options?.tokenEquals ?? ((a: T, b: T): boolean => a === b);

  const prefix: T[] = [...state.prefix];
  let { x, y } = state;

  // Upper bound on iterations: we can ascend at most prefix.length times
  // and descend at most maxDepth times, plus a small constant.
  const limit = state.prefix.length + maxDepth + 2;

  for (let iter = 0; iter < limit; iter++) {
    const outOfBounds = x < 0 || x >= 1 || y < 0 || y >= 1;

    // --- Phase 1: ascend if out of the current square ---
    if (outOfBounds && prefix.length > 0) {
      const lastToken = prefix.pop()!;
      const dist = model(prefix);

      let cumBefore = 0;
      let prob = 0;
      for (const entry of dist) {
        if (eq(entry.token, lastToken)) {
          prob = entry.probability;
          break;
        }
        cumBefore += entry.probability;
      }

      // Map back to parent's coordinate frame.
      //   parent_x = (1 − prob) + child_x · prob
      //   parent_y = cumBefore  + child_y · prob
      x = (1 - prob) + x * prob;
      y = cumBefore + y * prob;
      continue;
    }

    // --- Clamp at root if still out of bounds ---
    if (outOfBounds && prefix.length === 0) {
      x = Math.max(0, Math.min(x, 1 - Number.EPSILON));
      y = Math.max(0, Math.min(y, 1 - Number.EPSILON));
    }

    // --- Phase 2: try to descend into the smallest containing child ---
    if (prefix.length >= maxDepth) break;

    const dist = model(prefix);
    if (dist.length === 0) break;

    let cumProb = 0;
    let descended = false;

    for (const entry of dist) {
      const p = entry.probability;
      if (p <= 0) continue;

      //  Child occupies x ∈ [1−p, 1],  y ∈ [cumProb, cumProb+p]
      //  in the parent's normalised frame.
      if (x >= 1 - p && y >= cumProb && y < cumProb + p) {
        prefix.push(entry.token);
        x = (x - (1 - p)) / p;
        y = (y - cumProb) / p;
        descended = true;
        break;
      }
      cumProb += p;
    }

    if (!descended) break;
  }

  return { prefix, x, y };
}

// ---------------------------------------------------------------------------
// Helper: convert cursor state to global unit-square coordinates
// ---------------------------------------------------------------------------

/**
 * Map a cursor state to global (x, y) in the unit square.
 *
 * Useful for verifying that normalisation preserves position.
 */
export function cursorToGlobal<T>(
  model: LanguageModel<T>,
  state: CursorState<T>,
  tokenEquals?: (a: T, b: T) => boolean,
): { x: number; y: number } {
  const eq = tokenEquals ?? ((a: T, b: T): boolean => a === b);

  let size = 1; // P(prefix so far)
  let top = 0;  // y-offset of current square's top edge in global coords

  for (let i = 0; i < state.prefix.length; i++) {
    const parentPrefix = state.prefix.slice(0, i);
    const dist = model(parentPrefix);
    const token = state.prefix[i];

    let cumBefore = 0;
    let prob = 0;
    for (const entry of dist) {
      if (eq(entry.token, token)) {
        prob = entry.probability;
        break;
      }
      cumBefore += entry.probability;
    }

    top += cumBefore * size;
    size *= prob;
  }

  return {
    x: (1 - size) + state.x * size,
    y: top + state.y * size,
  };
}
