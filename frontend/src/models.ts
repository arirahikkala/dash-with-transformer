/**
 * Adapt a byte-level UTF-8 language model into a codepoint-level model.
 * Also: interpolation between arbitrary language models.
 */
import {
  mergeAsyncIterables,
  raceAsyncIterables,
  racePromises,
} from "./async-iterables";
import { createTrieCache } from "./trie-cache";
import {
  first,
  type LanguageModel,
  type PlainLanguageModel,
  type PlainTokenProb,
  type TokenProb,
} from "./types";

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

type ByteLevelModel = (
  bytePrefix: Uint8Array,
  minProb: number,
) => Promise<number[]>;

/**
 * Recursively expand a partial UTF-8 sequence into filtered TokenProb entries
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
  model: ByteLevelModel,
  bytePrefix: Uint8Array,
  partialBytes: number[],
  totalBytes: number,
  probSoFar: number,
  cumStart: number,
  rangeStart: number,
  rangeEnd: number,
  minProb: number,
): AsyncGenerator<TokenProb<number>> {
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
  const subtrees: AsyncIterable<TokenProb<number>>[] = [];
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
 * Returns the TokenProb for the codepoint, or null if the model assigns it
 * zero probability at any byte level.
 */
async function lookupSpecificToken(
  model: ByteLevelModel,
  bytePrefix: Uint8Array,
  codepoint: number,
): Promise<TokenProb<number> | null> {
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
 * Adapt a byte-level UTF-8 model into a codepoint-level LanguageModel.
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
): LanguageModel<readonly number[], number> {
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
    const expansions: AsyncIterable<TokenProb<number>>[] = [];

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
 * All models must have their alphabets in the same order: they may yield
 * values in different orders, but the CDF must be stacked in the same token
 * ordering.  Under this assumption, a token's interpolated extent is simply
 * the weighted combination of its extents across all models:
 *
 *     start_mix = Σ_i  w_i * start_i
 *     end_mix   = Σ_i  w_i * end_i
 *
 * The implementation streams results from all models in parallel:
 * 1. All models are queried with the caller's minProb (any token with
 *    mixture probability >= minProb must have prob >= minProb in at least
 *    one model, since Σw_i = 1).
 * 2. Results are raced; as soon as a token has been received from all
 *    models, its interpolated extent is yielded immediately.
 * 3. After all streams exhaust, remainder tokens (not returned by every
 *    model) are resolved via parallel specificToken queries.
 *
 * Weights are normalized to sum to 1. Negative weights throw an error.
 */
export function interpolate<P, T>(
  components: { model: LanguageModel<P, T>; weight: number }[],
): LanguageModel<P, T> {
  if (components.length === 1) {
    return components[0].model;
  }
  if (components.length === 0) {
    throw new Error("interpolate requires at least one model");
  }
  for (const { weight } of components) {
    if (weight < 0) {
      throw new Error(`Negative weight: ${weight}`);
    }
  }
  const totalWeight = components.reduce((s, c) => s + c.weight, 0);
  if (totalWeight === 0) {
    throw new Error("Total weight must be positive");
  }
  const weights = components.map((c) => c.weight / totalWeight);
  const models = components.map((c) => c.model);
  const n = models.length;

  return async function* (
    prefix,
    rangeStart,
    rangeEnd,
    minProb,
    specificToken?,
  ) {
    // Fast path: look up a single token's extent in all models.
    if (specificToken !== undefined) {
      const entries = await Promise.all(
        models.map((m) => first(m(prefix, 0, 0, 0, specificToken))),
      );
      if (entries.some((e) => !e)) return;
      let start = 0;
      let end = 0;
      for (let i = 0; i < n; i++) {
        start += weights[i] * entries[i]!.start;
        end += weights[i] * entries[i]!.end;
      }
      yield { token: specificToken, start, end };
      return;
    }

    // Query all models with full range and the caller's minProb.
    // Any token with mixture probability >= minProb must have probability
    // >= minProb in at least one model (since Σw_i = 1).
    const maps: Map<T, TokenProb<T>>[] = models.map(() => new Map());
    const yielded = new Set<T>();

    for await (const { value: entry, index } of raceAsyncIterables(
      models.map((m) => m(prefix, 0, 1, minProb)),
    )) {
      maps[index].set(entry.token, entry);

      // Check if all models have now reported this token.
      if (maps.every((m) => m.has(entry.token))) {
        const token = entry.token;
        let start = 0;
        let end = 0;
        for (let i = 0; i < n; i++) {
          const e = maps[i].get(token)!;
          start += weights[i] * e.start;
          end += weights[i] * e.end;
        }
        if (end >= rangeStart && start <= rangeEnd && end - start >= minProb) {
          yielded.add(token);
          yield { token, start, end };
        }
      }
    }

    // Remainder: tokens that appeared in some but not all models' output.
    // For each, query the missing models with specificToken.
    const allTokens = new Set<T>();
    for (const m of maps) {
      for (const token of m.keys()) allTokens.add(token);
    }

    const remainderPromises: Promise<TokenProb<T> | null>[] = [];

    for (const token of allTokens) {
      if (yielded.has(token)) continue;
      remainderPromises.push(
        Promise.all(
          models.map((m, i) =>
            maps[i].has(token)
              ? Promise.resolve(maps[i].get(token)!)
              : first(m(prefix, 0, 1, 0, token)),
          ),
        ).then((entries) => {
          if (entries.some((e) => !e)) return null;
          let start = 0;
          let end = 0;
          for (let i = 0; i < n; i++) {
            start += weights[i] * entries[i]!.start;
            end += weights[i] * entries[i]!.end;
          }
          if (end < rangeStart || start > rangeEnd || end - start < minProb)
            return null;
          return { token, start, end };
        }),
      );
    }

    for await (const result of racePromises(remainderPromises)) {
      if (result) yield result;
    }
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
 * Byte-trie cache for a PlainLanguageModel over byte prefixes.
 * Each unique prefix is computed at most once; subsequent queries
 * return the cached result.  Old entries are evicted automatically.
 */
export function trieCache(
  model: PlainLanguageModel<Uint8Array, number>,
): PlainLanguageModel<Uint8Array, number> {
  const cache = createTrieCache<readonly PlainTokenProb<number>[]>();
  return async (prefix: Uint8Array) => {
    const cached = cache.get(prefix);
    if (cached !== undefined) return cached;
    const result = await model(prefix);
    cache.set(prefix, result);
    return result;
  };
}

/**
 * Wrap a byte-level model to filter out any next-byte predictions that
 * would violate UTF-8 encoding rules, and renormalise the distribution.
 */
export function forceCleanUtf8(
  model: PlainLanguageModel<Uint8Array, number>,
): PlainLanguageModel<Uint8Array, number> {
  return async (prefix: Uint8Array) => {
    const dist = await model(prefix);
    const isLegal = legalUtf8NextByte(prefix);

    const filtered = dist.filter(({ token }) => isLegal(token));

    const total = filtered.reduce(
      (sum, { probability }) => sum + probability,
      0,
    );
    if (total === 0) return [];

    return filtered.map(({ token, probability }) => ({
      token,
      probability: probability / total,
    }));
  };
}
