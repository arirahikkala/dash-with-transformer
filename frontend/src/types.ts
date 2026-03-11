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
 * Branding infrastructure for language model normalization state.
 *
 * A `LanguageModel` is branded as normalized (probabilities sum to 1).
 * An `UnnormalizedLanguageModel` may have probabilities that don't sum to 1
 * (e.g. after zeroing out illegal bytes or combining byte + special-token
 * probabilities).  Use `normalize()` (in models.ts) to convert.
 */
declare const __normalized: unique symbol;
declare const __unnormalized: unique symbol;

type RawModel<P> = (prefix: P) => Promise<readonly number[]>;

/**
 * A language model whose output distribution sums to 1.
 *
 * Implementations must return the *full* next-token distribution: one entry
 * per token in the model's alphabet, in a fixed canonical order (e.g. byte
 * 0, 1, …, 255 for a byte-level model), including tokens with zero
 * probability.  Callers may rely on positional indexing: `dist[t]` is the
 * probability of token `t`.
 */
export type LanguageModel<P> = RawModel<P> & { readonly [__normalized]: true };

/**
 * A language model whose output probabilities may not sum to 1.
 * Must be passed through `normalize()` before use as a `LanguageModel`.
 */
export type UnnormalizedLanguageModel<P> = RawModel<P> & {
  readonly [__unnormalized]: true;
};

/** Brand a raw distribution function as a normalized LanguageModel. */
export function asNormalized<P>(fn: RawModel<P>): LanguageModel<P> {
  return fn as LanguageModel<P>;
}

/** Brand a raw distribution function as an UnnormalizedLanguageModel. */
export function asUnnormalized<P>(
  fn: RawModel<P>,
): UnnormalizedLanguageModel<P> {
  return fn as UnnormalizedLanguageModel<P>;
}

/**
 * A view into the cumulative distribution function of a language model's
 * next-token distribution at a given prefix.
 *
 * Two operations:
 * - `slice`: stream entries that overlap a CDF range and meet a minimum
 *   probability threshold.
 * - `token`: look up a single token's extent with minimal computation.
 *
 * Implementations must ensure that:
 * - token extents do not overlap
 * - token extents don't vary between different calls with the same prefix;
 *   only the ordering and presence may change
 * - each token yielded per call is unique
 * - tokens are yielded as soon as they can be (don't wait for earlier tokens)
 *
 * Tokens may be yielded in any order.
 *
 * @template P - The type of the prefix (e.g. `string`, `readonly number[]`).
 * @template T - The type of each next-token.
 */
export interface CDFView<P, T> {
  /**
   * Stream entries whose half-open extent [start, end) overlaps
   * [rangeStart, rangeEnd] and whose probability (end − start) ≥ minProb.
   */
  slice(
    prefix: P,
    rangeStart: number,
    rangeEnd: number,
    minProb: number,
  ): AsyncIterable<TokenCDFExtent<T>>;

  /**
   * Look up a single token's cumulative extent.
   * Returns null if the token has zero probability.
   */
  token(prefix: P, specificToken: T): Promise<TokenCDFExtent<T> | null>;
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
