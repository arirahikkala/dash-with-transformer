/**
 * Cursor normalization for the unit-square model.
 * See CLAUDE.md for the geometric model and core math concepts.
 */

import type { CDFView, Cursor } from "./types";
import { first } from "./types";

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
 *
 * Returns `null` if aborted via the signal.
 */
export async function normalizeCursor<T>(
  model: CDFView<readonly T[], T>,
  state: Cursor<T>,
  signal?: AbortSignal,
): Promise<Cursor<T> | null> {
  const prefix: T[] = [...state.prefix];
  let x = state.x;
  let y = state.y;

  for (;;) {
    if (signal?.aborted) return null;

    const oob = x < 0 || x >= 1 || y < 0 || y >= 1;

    // --- Phase 1: ascend if out of the current square ---
    if (oob && prefix.length > 0) {
      const lastToken = prefix.pop()!;
      const tokenResult = await first(model(prefix, 0, 1, 0, lastToken));
      if (signal?.aborted) return null;

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
    const entry = await first(model(prefix, y, y, 1 - x));
    if (signal?.aborted) return null;
    if (!entry) break;

    const p = entry.end - entry.start;
    prefix.push(entry.token);
    x = (x - (1 - p)) / p;
    y = (y - entry.start) / p;
  }

  return { prefix, x, y };
}
