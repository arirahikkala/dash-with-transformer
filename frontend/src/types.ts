/**
 * Core domain types for the unit-square model.
 */

/** A Unicode codepoint token. */
export interface UnicodeCodepoint {
  readonly type: "codepoint";
  readonly codepoint: number;
}

/** A special token with a model-specific index and human-readable label. */
export interface SpecialToken {
  readonly type: "special";
  readonly index: number;
  readonly label: string;
}

/**
 * A token displayed in the widget: either a Unicode codepoint or a special
 * token (e.g. `<im_start>`, `<eos>`).
 */
export type WidgetToken = UnicodeCodepoint | SpecialToken;

/**
 * One entry in a next-token probability distribution, placed on the
 * cumulative-probability line.  The token occupies the half-open interval
 * [start, end); its probability is `end − start`.
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
 * A simple language model that returns a plain probability distribution
 * (without cumulative extents or filtering).  Use `adaptModel` (in models.ts) to convert
 * to a full `CDFView`.
 *
 * Implementations must return the *full* next-token distribution: one entry
 * per token in the model's alphabet, in a fixed canonical order (e.g. byte
 * 0, 1, …, 255 for a byte-level model), including tokens with zero
 * probability.  Callers may rely on positional indexing: `dist[t]` is the
 * probability of token `t`.
 */
export type LanguageModel<P> = (prefix: P) => Promise<readonly number[]>;

/**
 * Given a prefix and visibility constraints, finds the matching entries from the next-token distribution.
 *
 * Implementations must ensure that:
 * - token extents do not overlap
 * - token extents don't vary between different calls with the same prefix; only the ordering and presence may change
 * - each token yielded per call is unique
 * - tokens are yielded as soon as they can be (don't wait for earlier tokens)
 *
 * Tokens may be yielded in any order.
 *
 * @param prefix        - The token prefix.
 * @param rangeStart    - Only return entries whose half-open extent [start, end)
 *                        overlaps the point or range [rangeStart, rangeEnd].
 * @param rangeEnd      - Inclusive upper bound of the query range.
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
