/**
 * Adapt a byte-level UTF-8 language model into a codepoint-level model.
 * Also: interpolation between arbitrary language models.
 */
import {
  mergeAsyncIterables,
  raceAsyncIterables,
  racePromises,
} from "./async-iterables";
import { first, type LanguageModel, type TokenProb } from "./types";

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
  minSize: number,
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
  minSize: number,
): AsyncGenerator<TokenProb<number>> {
  if (partialBytes.length === totalBytes) {
    const start = cumStart;
    const end = cumStart + probSoFar;
    if (end < rangeStart || start > rangeEnd) return;
    if (end - start < minSize) return;
    yield { token: decodeUtf8Bytes(partialBytes), start, end };
    return;
  }

  const queryPrefix = new Uint8Array([...bytePrefix, ...partialBytes]);
  const dist = await model(queryPrefix, minSize);

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
    if (cum - subCumStart < minSize) continue;

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
        minSize,
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
 * - Multi-byte groups with total probability < minSize are skipped
 *   (every codepoint in the group must be ≤ the group total).
 * - The same pruning is applied recursively at every continuation-byte
 *   level, so e.g. a 3-byte group only queries the third byte for
 *   second-byte sub-groups that overlap the range and meet minSize.
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
    minSize: number,
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
    const firstByteDist = await byteLevelModel(bytePrefix, minSize);

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
      if (end - start < minSize) continue;
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
      // Skip groups where every codepoint is below minSize.
      if (firstByteDist[b] < minSize) continue;

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
          minSize,
        ),
      );
    }

    yield* mergeAsyncIterables(expansions);
  };
}

// UTF-16 char code → Unicode codepoint adapter
// ---------------------------------------------------------------------------

/** Encode a codepoint sequence to UTF-16 char codes. */
function codepointsToCharCodes(codepoints: readonly number[]): number[] {
  const result: number[] = [];
  for (const cp of codepoints) {
    if (cp <= 0xffff) {
      result.push(cp);
    } else {
      result.push(0xd800 + ((cp - 0x10000) >> 10));
      result.push(0xdc00 + ((cp - 0x10000) & 0x3ff));
    }
  }
  return result;
}

/**
 * Expand a high surrogate entry into codepoint entries by querying the
 * char code model for low surrogates.
 */
async function* expandSurrogate(
  charCodeModel: LanguageModel<readonly number[], number>,
  charCodePrefix: readonly number[],
  highEntry: TokenProb<number>,
  rangeStart: number,
  rangeEnd: number,
  minSize: number,
): AsyncGenerator<TokenProb<number>> {
  const high = highEntry.token;
  const groupSize = highEntry.end - highEntry.start;

  const subRangeStart = Math.max(0, (rangeStart - highEntry.start) / groupSize);
  const subRangeEnd = Math.min(1, (rangeEnd - highEntry.start) / groupSize);
  const subMinSize = minSize / groupSize;

  for await (const lowEntry of charCodeModel(
    [...charCodePrefix, high],
    subRangeStart,
    subRangeEnd,
    subMinSize,
  )) {
    const low = lowEntry.token;
    if (low < 0xdc00 || low > 0xdfff) continue;

    const codepoint = 0x10000 + ((high - 0xd800) << 10) + (low - 0xdc00);
    yield {
      token: codepoint,
      start: highEntry.start + lowEntry.start * groupSize,
      end: highEntry.start + lowEntry.end * groupSize,
    };
  }
}

/**
 * Adapt a UTF-16 char code level model into a Unicode codepoint level model.
 *
 * BMP codepoints (U+0000–U+D7FF, U+E000–U+FFFF) pass through directly.
 * Non-BMP codepoints (U+10000–U+10FFFF) are decoded from surrogate pairs.
 */
export function fromCharCodeModel(
  charCodeModel: LanguageModel<readonly number[], number>,
): LanguageModel<readonly number[], number> {
  return async function* (
    prefix: readonly number[],
    rangeStart: number,
    rangeEnd: number,
    minSize: number,
    specificToken?: number,
  ) {
    const charCodePrefix = codepointsToCharCodes(prefix);

    if (specificToken !== undefined) {
      if (specificToken <= 0xffff) {
        yield* charCodeModel(charCodePrefix, 0, 0, 0, specificToken);
      } else {
        const high = 0xd800 + ((specificToken - 0x10000) >> 10);
        const low = 0xdc00 + ((specificToken - 0x10000) & 0x3ff);
        const highEntry = await first(
          charCodeModel(charCodePrefix, 0, 0, 0, high),
        );
        if (!highEntry) return;
        const groupSize = highEntry.end - highEntry.start;
        const lowEntry = await first(
          charCodeModel([...charCodePrefix, high], 0, 0, 0, low),
        );
        if (!lowEntry) return;
        yield {
          token: specificToken,
          start: highEntry.start + lowEntry.start * groupSize,
          end: highEntry.start + lowEntry.end * groupSize,
        };
      }
      return;
    }

    const expansions: AsyncIterable<TokenProb<number>>[] = [];
    for await (const entry of charCodeModel(
      charCodePrefix,
      rangeStart,
      rangeEnd,
      minSize,
    )) {
      const cc = entry.token;
      if (cc >= 0xd800 && cc <= 0xdbff) {
        expansions.push(
          expandSurrogate(
            charCodeModel,
            charCodePrefix,
            entry,
            rangeStart,
            rangeEnd,
            minSize,
          ),
        );
      } else if (cc >= 0xdc00 && cc <= 0xdfff) {
        // lone low surrogate: skip
      } else {
        yield { token: cc, start: entry.start, end: entry.end };
      }
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
 * 1. All models are queried with the caller's minSize (any token with
 *    mixture probability >= minSize must have prob >= minSize in at least
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
    minSize,
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

    // Query all models with full range and the caller's minSize.
    // Any token with mixture probability >= minSize must have probability
    // >= minSize in at least one model (since Σw_i = 1).
    const maps: Map<T, TokenProb<T>>[] = models.map(() => new Map());
    const yielded = new Set<T>();

    for await (const { value: entry, index } of raceAsyncIterables(
      models.map((m) => m(prefix, 0, 1, minSize)),
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
        if (end >= rangeStart && start <= rangeEnd && end - start >= minSize) {
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
          if (end < rangeStart || start > rangeEnd || end - start < minSize)
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
