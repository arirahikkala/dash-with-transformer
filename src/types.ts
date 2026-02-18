/**
 * Core domain types for the Dasher unit-square model.
 */

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
 *
 * @template P - The type of the prefix (e.g. `string`, `readonly number[]`).
 * @template T - The type of each next-token.
 */
export type LanguageModel<P, T> = (
  prefix: P,
) => Promise<readonly TokenProb<T>[]>;

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
  children: Promise<SceneNode<T>[]>;
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
  children: SceneNode<T>[];
}
