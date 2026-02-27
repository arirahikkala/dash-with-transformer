/**
 * Core domain types for the unit-square model.
 */

/**
 * One entry in a next-token probability distribution, placed on the
 * cumulative-probability line.  `start` and `end` are the entry's
 * extent on [0, 1]; the token's probability is `end − start`.
 *
 * A node's extents depend only on the prefix — the query parameters
 * (rangeStart, rangeEnd, minProb) only govern whether it is listed.
 */
export interface TokenCDFExtent<T> {
  readonly token: T;
  readonly start: number;
  readonly end: number;
}

/**
 * A plain token-probability pair, before cumulative extents are computed.
 */
export interface TokenProb<T> {
  readonly token: T;
  readonly probability: number;
}

/**
 * A simple language model that returns a plain probability distribution
 * (without cumulative extents or filtering).  Use `adaptModel` to convert
 * to a full `CDFView`.
 */
export type LanguageModel<P, T> = (
  prefix: P,
) => Promise<readonly TokenProb<T>[]>;

/**
 * Given a prefix and visibility constraints, finds the matching entries from the next-token distribution.
 *
 * Implementations must ensure that:
 * - token extents do not overlap
 * - token extents have no holes between successive entries in the full distribution
 * - token extents don't vary between different calls with the same prefix; only the ordering and presence may change
 *
 * Tokens may be yielded in any order.
 *
 * @param prefix        - The token prefix.
 * @param rangeStart    - Only return entries overlapping [rangeStart, rangeEnd].
 * @param rangeEnd      - Upper bound of the visible range.
 * @param minProb       - Only return entries with (end − start) ≥ minProb.
 * @param specificToken - If set, return only this token's extent (ignoring
 *                        range/size filters) with minimal computation.
 *
 * @template P - The type of the prefix (e.g. `string`, `readonly number[]`).
 * @template T - The type of each next-token.
 */
export type CDFView<P, T> = (
  prefix: P,
  rangeStart: number,
  rangeEnd: number,
  minProb: number,
  specificToken?: T,
) => AsyncIterable<TokenCDFExtent<T>>;

/**
 * Adapt a simple probability-list model into a full CDFView
 * that computes cumulative extents and handles range/size filtering.
 */
export function adaptModel<P, T>(
  inner: (prefix: P) => Promise<readonly TokenProb<T>[]>,
): CDFView<P, T> {
  return async function* (
    prefix,
    rangeStart,
    rangeEnd,
    minProb,
    specificToken,
  ) {
    const dist = await inner(prefix);
    let cum = 0;
    for (const entry of dist) {
      const start = cum;
      const end = cum + entry.probability;
      cum = end;
      if (specificToken !== undefined) {
        if (entry.token === specificToken) {
          yield { token: entry.token, start, end };
          return;
        }
        continue;
      }
      if (end < rangeStart || start > rangeEnd) continue;
      if (entry.probability < minProb) continue;
      yield { token: entry.token, start, end };
    }
  };
}

/** Cursor: a discrete prefix plus a continuous adjustment. */
export interface Cursor<T> {
  readonly prefix: readonly T[];
  /** 0 = left edge of the prefix's square, 1 = right edge. */
  readonly x: number;
  /** 0 = top edge of the prefix's square, 1 = bottom edge. */
  readonly y: number;
}

/** A node in the recursive prediction tree. */
export interface SceneNode<T> {
  /** The token this node represents. */
  token: T;
  /** Top edge in window-relative coordinates [0,1]. */
  y0: number;
  /** Bottom edge in window-relative coordinates [0,1]. */
  y1: number;
  /** Recursively expanded children (next-token predictions). */
  children: AsyncIterable<SceneNode<T>>;
}

/** Everything needed to render one frame of the widget. */
export interface Scene<T> {
  /** Top-level prediction nodes. */
  children: AsyncIterable<SceneNode<T>>;
  /** Length of the prefix at the scene root. */
  prefixLength: number;
}

/** Extract the first element from an async iterable, or undefined if empty. */
export async function first<T>(iter: AsyncIterable<T>): Promise<T | undefined> {
  for await (const item of iter) {
    return item;
  }
  return undefined;
}
