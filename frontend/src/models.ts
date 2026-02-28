/**
 * Adapt a byte-level UTF-8 language model into a codepoint-level model.
 * Also: interpolation between arbitrary language models.
 */
import { mergeAsyncIterables } from "./async-iterables";
import { createTrieCache } from "./trie-cache";
import { type CDFView, type LanguageModel, type TokenCDFExtent } from "./types";

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

/** Encode a codepoint sequence to UTF-8 bytes. */
function codepointsToUtf8(codepoints: readonly number[]): Uint8Array {
  return new TextEncoder().encode(String.fromCodePoint(...codepoints));
}

/** Number of bytes in a UTF-8 sequence given the lead byte, or 0 if invalid. */
function utf8SeqLength(leadByte: number): number {
  if (leadByte <= 0x7f) return 1;
  if (leadByte >= 0xc2 && leadByte <= 0xdf) return 2;
  if (leadByte >= 0xe0 && leadByte <= 0xef) return 3;
  if (leadByte >= 0xf0 && leadByte <= 0xf4) return 4;
  return 0;
}

/** Decode a complete UTF-8 byte sequence to a Unicode codepoint. */
function decodeUtf8Bytes(bytes: number[]): number {
  switch (bytes.length) {
    case 1:
      return bytes[0];
    case 2:
      return ((bytes[0] & 0x1f) << 6) | (bytes[1] & 0x3f);
    case 3:
      return (
        ((bytes[0] & 0x0f) << 12) | ((bytes[1] & 0x3f) << 6) | (bytes[2] & 0x3f)
      );
    case 4:
      return (
        ((bytes[0] & 0x07) << 18) |
        ((bytes[1] & 0x3f) << 12) |
        ((bytes[2] & 0x3f) << 6) |
        (bytes[3] & 0x3f)
      );
    default:
      throw new Error(`Invalid UTF-8 sequence length: ${bytes.length}`);
  }
}

// ---------------------------------------------------------------------------
// Multi-byte expansion
// ---------------------------------------------------------------------------

/**
 * A byte-level language model.  Returns exactly 256 probabilities (bytes
 * 0–255) in order, including zero-probability bytes.  `dist[b]` is the
 * probability of byte `b`.
 */
export type ByteLevelModel = (
  bytePrefix: Uint8Array,
  minProb: number,
) => Promise<readonly number[]>;

/**
 * Thread a `minProb` parameter through a chain of LanguageModel adapters.
 *
 * `ByteLevelModel` is just `LanguageModel` with an extra `minProb` hint.
 * This lifts a `LanguageModel → LanguageModel` adapter into a
 * `ByteLevelModel → ByteLevelModel` adapter by capturing minProb via
 * closure before entering the adapter chain.
 *
 * The minProb value is captured synchronously when the returned function is
 * called, before any awaits in the adapter chain, so concurrent calls with
 * different minProb values are safe.
 *
 * Usage:
 *   fromByteLevelModel(passMinProb(m => trieCache(forceCleanUtf8(m)))(predictBytes))
 */
export function passMinProb(
  adapt: (model: LanguageModel<Uint8Array>) => LanguageModel<Uint8Array>,
): (inner: ByteLevelModel) => ByteLevelModel {
  return (inner) => {
    let currentMinProb = 0;
    const adapted = adapt(async (prefix) => inner(prefix, currentMinProb));
    return async (prefix, minProb) => {
      currentMinProb = minProb;
      return adapted(prefix);
    };
  };
}

/**
 * Recursively expand a partial UTF-8 sequence into filtered TokenCDFExtent entries
 * by querying the byte-level model for continuation bytes.
 *
 * At each level the cumulative positions of all sub-groups are computed
 * (so that later sub-groups have correct absolute positions even when
 * earlier ones are skipped), and only sub-groups that overlap the visible
 * range and meet the minimum-size threshold are recursed into.
 *
 * Yields entries as soon as they are ready, in no particular order.
 * All continuation queries at the same depth run in parallel.
 */
async function* expandMultiByte(
  model: (prefix: Uint8Array, minProb: number) => Promise<readonly number[]>,
  bytePrefix: Uint8Array,
  partialBytes: number[],
  totalBytes: number,
  probSoFar: number,
  cumStart: number,
  rangeStart: number,
  rangeEnd: number,
  minProb: number,
): AsyncGenerator<TokenCDFExtent<number>> {
  if (partialBytes.length === totalBytes) {
    const start = cumStart;
    const end = cumStart + probSoFar;
    if (end < rangeStart || start > rangeEnd) return;
    if (end - start < minProb) return;
    yield { token: decodeUtf8Bytes(partialBytes), start, end };
    return;
  }

  const queryPrefix = new Uint8Array([...bytePrefix, ...partialBytes]);
  const dist = await model(queryPrefix, minProb);

  // Single pass: accumulate cumulative positions for ALL non-zero
  // continuation bytes (so later sub-groups are positioned correctly),
  // but only recurse into those that pass the range/size filters.
  const subtrees: AsyncIterable<TokenCDFExtent<number>>[] = [];
  let cum = cumStart;
  for (let b = 0; b < 256; b++) {
    if (dist[b] === 0) continue;
    const subProb = probSoFar * dist[b];
    const subCumStart = cum;
    cum += subProb;

    if (cum < rangeStart || subCumStart > rangeEnd) continue;
    if (cum - subCumStart < minProb) continue;

    subtrees.push(
      expandMultiByte(
        model,
        bytePrefix,
        [...partialBytes, b],
        totalBytes,
        subProb,
        subCumStart,
        rangeStart,
        rangeEnd,
        minProb,
      ),
    );
  }

  yield* mergeAsyncIterables(subtrees);
}

// ---------------------------------------------------------------------------
// Specific-token lookup
// ---------------------------------------------------------------------------

/**
 * Look up a single codepoint's cumulative extent in the byte-level model.
 * Returns the TokenCDFExtent for the codepoint, or null if the model assigns it
 * zero probability at any byte level.
 */
async function lookupSpecificToken(
  model: (prefix: Uint8Array, minProb: number) => Promise<readonly number[]>,
  bytePrefix: Uint8Array,
  codepoint: number,
): Promise<TokenCDFExtent<number> | null> {
  const targetBytes = [...codepointsToUtf8([codepoint])];

  // Fire all byte-level queries in parallel — the prefix for level i
  // is bytePrefix + targetBytes[0..i), which is known upfront.
  const dists = await Promise.all(
    targetBytes.map((_, i) =>
      model(new Uint8Array([...bytePrefix, ...targetBytes.slice(0, i)]), 2),
    ),
  );

  // Process sequentially to compute cumulative position.
  let cumStart = 0;
  let probSoFar = 1;

  for (let i = 0; i < targetBytes.length; i++) {
    const dist = dists[i];
    const targetByte = targetBytes[i];

    if (dist[targetByte] === 0) return null;

    for (let b = 0; b < targetByte; b++) {
      if (dist[b] > 0) {
        cumStart += probSoFar * dist[b];
      }
    }
    probSoFar *= dist[targetByte];
  }

  return { token: codepoint, start: cumStart, end: cumStart + probSoFar };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Adapt a byte-level UTF-8 model into a codepoint-level CDFView.
 *
 * The byte-level model predicts the next byte given a UTF-8 byte prefix.
 * It must return exactly 256 probabilities summing to 1, with zero
 * probability for any illegal UTF-8 continuation.
 *
 * Byte-level model calls are minimised:
 * - 1 call for the first-byte distribution per query.
 * - Multi-byte groups entirely outside [rangeStart, rangeEnd] are skipped.
 * - Multi-byte groups with total probability < minProb are skipped
 *   (every codepoint in the group must be ≤ the group total).
 * - The same pruning is applied recursively at every continuation-byte
 *   level, so e.g. a 3-byte group only queries the third byte for
 *   second-byte sub-groups that overlap the range and meet minProb.
 *
 * Calls are maximally parallel: all continuation queries at the same
 * depth run concurrently, yielding results as they arrive.
 *
 * Cumulative extents are deterministic (the loops accumulate over ALL
 * non-zero bytes in fixed order, regardless of the query window) so
 * that a codepoint's position depends only on the prefix.
 */
export function fromByteLevelModel(
  byteLevelModel: ByteLevelModel,
): CDFView<readonly number[], number> {
  return async function* (
    prefix: readonly number[],
    rangeStart: number,
    rangeEnd: number,
    minProb: number,
    specificToken?: number,
  ) {
    const bytePrefix = codepointsToUtf8(prefix);

    // Fast path: look up a single codepoint's extent.
    if (specificToken !== undefined) {
      const result = await lookupSpecificToken(
        byteLevelModel,
        bytePrefix,
        specificToken,
      );
      if (result) yield result;
      return;
    }

    // 1. First-byte distribution (1 model call).
    const firstByteDist = await byteLevelModel(bytePrefix, minProb);

    // 2. Cumulative start position for every first-byte group.
    //    P(all codepoints starting with byte b) = P(b), so the
    //    cumulative over first bytes gives correct group boundaries.
    const firstByteCumStart: number[] = new Array(256);
    let cum = 0;
    for (let b = 0; b < 256; b++) {
      firstByteCumStart[b] = cum;
      if (firstByteDist[b] > 0) {
        cum += firstByteDist[b];
      }
    }

    // 3. Single-byte codepoints (0x00–0x7F): yield directly.
    for (let b = 0; b <= 0x7f; b++) {
      if (firstByteDist[b] === 0) continue;
      const start = firstByteCumStart[b];
      const end = firstByteCumStart[b] + firstByteDist[b];
      if (end < rangeStart || start > rangeEnd) continue;
      if (end - start < minProb) continue;
      yield { token: b, start, end };
    }

    // 4. Multi-byte groups: expand in parallel, yield as each resolves.
    const expansions: AsyncIterable<TokenCDFExtent<number>>[] = [];

    for (let b = 0xc0; b <= 0xff; b++) {
      if (firstByteDist[b] === 0) continue;
      const totalBytes = utf8SeqLength(b);
      if (totalBytes === 0) continue; // invalid lead byte

      const groupEnd = firstByteCumStart[b] + firstByteDist[b];

      // Skip groups entirely outside the visible range.
      if (groupEnd < rangeStart || firstByteCumStart[b] > rangeEnd) continue;
      // Skip groups where every codepoint is below minProb.
      if (firstByteDist[b] < minProb) continue;

      expansions.push(
        expandMultiByte(
          byteLevelModel,
          bytePrefix,
          [b],
          totalBytes,
          firstByteDist[b],
          firstByteCumStart[b],
          rangeStart,
          rangeEnd,
          minProb,
        ),
      );
    }

    yield* mergeAsyncIterables(expansions);
  };
}

// ---------------------------------------------------------------------------
// Language model interpolation
// ---------------------------------------------------------------------------

/**
 * Interpolate between one or more language models, creating a per-conditional
 * weighted mixture.
 *
 * Models may have different vocabulary sizes; the result is in the space of
 * the largest vocabulary, with missing tokens treated as probability 0.
 *
 * Components with weight 0 are not queried at all.
 *
 * Weights are normalized to sum to 1. Negative weights throw an error.
 */
export function interpolate<P>(
  components: { model: LanguageModel<P>; weight: number }[],
): LanguageModel<P> {
  if (components.length === 0) {
    throw new Error("interpolate requires at least one model");
  }
  for (const { weight } of components) {
    if (weight < 0) {
      throw new Error(`Negative weight: ${weight}`);
    }
  }
  const active = components.filter((c) => c.weight > 0);
  if (active.length === 0) {
    throw new Error("Total weight must be positive");
  }
  if (active.length === 1) {
    return active[0].model;
  }
  const totalWeight = active.reduce((s, c) => s + c.weight, 0);
  const weights = active.map((c) => c.weight / totalWeight);
  const models = active.map((c) => c.model);

  return async (prefix) => {
    const dists = await Promise.all(models.map((m) => m(prefix)));
    const len = Math.max(...dists.map((d) => d.length));
    const result = new Array(len).fill(0);
    for (let i = 0; i < dists.length; i++) {
      const d = dists[i];
      for (let t = 0; t < d.length; t++) {
        result[t] += weights[i] * d[t];
      }
    }
    return result;
  };
}

// ---------------------------------------------------------------------------
// UTF-8 validity filtering for byte-level models
// ---------------------------------------------------------------------------

/**
 * Return a predicate that accepts exactly the byte values that are legal
 * as the next byte in a UTF-8 stream whose tail is `prefix`.
 */
function legalUtf8NextByte(prefix: Uint8Array): (byte: number) => boolean {
  const len = prefix.length;

  // Scan backwards (up to 3 bytes) to find the lead byte of the
  // current (potentially incomplete) character.
  let leadByte = -1;
  let consumed = 0; // bytes consumed from lead to end of prefix
  for (let back = 0; back < Math.min(4, len); back++) {
    const b = prefix[len - 1 - back];
    if (b < 0x80) {
      // ASCII — complete single-byte character; we're at a boundary.
      break;
    }
    if (b >= 0xc0) {
      // Lead byte found.
      const seqLen = utf8SeqLength(b);
      consumed = back + 1;
      if (seqLen > 0 && consumed < seqLen) {
        leadByte = b;
      }
      break;
    }
    // Continuation byte (0x80–0xBF) — keep scanning.
  }

  if (leadByte === -1) {
    // At character boundary: legal starts are ASCII + valid lead bytes.
    return (b: number) =>
      b <= 0x7f ||
      (b >= 0xc2 && b <= 0xdf) ||
      (b >= 0xe0 && b <= 0xef) ||
      (b >= 0xf0 && b <= 0xf4);
  }

  // Mid-character: we need a continuation byte.
  // The first continuation after the lead may have a restricted range.
  if (consumed === 1) {
    if (leadByte === 0xe0) return (b: number) => b >= 0xa0 && b <= 0xbf;
    if (leadByte === 0xed) return (b: number) => b >= 0x80 && b <= 0x9f;
    if (leadByte === 0xf0) return (b: number) => b >= 0x90 && b <= 0xbf;
    if (leadByte === 0xf4) return (b: number) => b >= 0x80 && b <= 0x8f;
  }

  // General continuation byte.
  return (b: number) => b >= 0x80 && b <= 0xbf;
}

/**
 * Byte-trie cache for a LanguageModel over byte prefixes.
 * Each unique prefix is computed at most once; subsequent queries
 * return the cached result.  Old entries are evicted automatically.
 */
export function trieCache(
  model: LanguageModel<Uint8Array>,
): LanguageModel<Uint8Array> {
  const cache = createTrieCache<Promise<readonly number[]>>();
  return (prefix: Uint8Array) => cache.getOrSet(prefix, () => model(prefix));
}

/**
 * Wrap a byte-level model to zero out any next-byte predictions that
 * would violate UTF-8 encoding rules, and renormalise the distribution.
 */
export function forceCleanUtf8(
  model: LanguageModel<Uint8Array>,
): LanguageModel<Uint8Array> {
  return async (prefix: Uint8Array) => {
    const dist = await model(prefix);
    const isLegal = legalUtf8NextByte(prefix);

    let total = 0;
    for (let b = 0; b < dist.length; b++) {
      if (isLegal(b)) total += dist[b];
    }
    if (total === 0) return dist.map(() => 0);

    return dist.map((p, b) => (isLegal(b) ? p / total : 0));
  };
}
