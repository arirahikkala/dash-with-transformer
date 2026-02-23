/**
 * Core domain types for the unit-square model.
 */

/**
 * One entry in a next-token probability distribution, placed on the
 * cumulative-probability line.  `start` and `end` are the entry's
 * extent on [0, 1]; the token's probability is `end − start`.
 *
 * A node's extents depend only on the prefix — the query parameters
 * (rangeStart, rangeEnd, minSize) only govern whether it is listed.
 */
export interface TokenProb<T> {
  readonly token: T;
  readonly start: number;
  readonly end: number;
}

/**
 * A plain token-probability pair, before cumulative extents are computed.
 */
export interface PlainTokenProb<T> {
  readonly token: T;
  readonly probability: number;
}

/**
 * A simple language model that returns a plain probability distribution
 * (without cumulative extents or filtering).  Use `adaptModel` to convert
 * to a full `LanguageModel`.
 */
export type PlainLanguageModel<P, T> = (
  prefix: P,
) => Promise<readonly PlainTokenProb<T>[]>;

/**
 * A language model: given a prefix and visibility constraints, return
 * the matching entries from the next-token distribution.
 *
 * Returned entries must not overlap, must have no holes between
 * successive entries in the full distribution, and their extents must depend
 * only on the prefix. Entries may be returned in any order.
 *
 * @param prefix        - The token prefix.
 * @param rangeStart    - Only return entries overlapping [rangeStart, rangeEnd].
 * @param rangeEnd      - Upper bound of the visible range.
 * @param minSize       - Only return entries with (end − start) ≥ minSize.
 * @param specificToken - If set, return only this token's extent (ignoring
 *                        range/size filters) with minimal computation.
 *
 * @template P - The type of the prefix (e.g. `string`, `readonly number[]`).
 * @template T - The type of each next-token.
 */
export type LanguageModel<P, T> = (
  prefix: P,
  rangeStart: number,
  rangeEnd: number,
  minSize: number,
  specificToken?: T,
) => AsyncIterable<TokenProb<T>>;

/**
 * Adapt a simple probability-list model into a full LanguageModel
 * that computes cumulative extents and handles range/size filtering.
 */
export function adaptModel<P, T>(
  inner: (prefix: P) => Promise<readonly PlainTokenProb<T>[]>,
): LanguageModel<P, T> {
  return async function* (
    prefix,
    rangeStart,
    rangeEnd,
    minSize,
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
      if (entry.probability < minSize) continue;
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

/** Display helpers for rendering tokens of type T. */
export interface TokenDisplay<T> {
  label: (token: T) => string;
  color: (token: T) => string;
  prefixToString: (prefix: readonly T[]) => string;
}

/** Everything needed to render one frame of the widget. */
export interface Scene<T> {
  /** Top-level prediction nodes. */
  children: AsyncIterable<SceneNode<T>>;
}

/** Extract the first element from an async iterable, or undefined if empty. */
export async function first<T>(iter: AsyncIterable<T>): Promise<T | undefined> {
  for await (const item of iter) {
    return item;
  }
  return undefined;
}
