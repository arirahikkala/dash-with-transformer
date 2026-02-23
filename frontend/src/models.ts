/**
 * Adapt a byte-level UTF-8 language model into a codepoint-level model.
 * Also: interpolation between arbitrary language models.
 */
import type { LanguageModel, TokenProb } from "./types";

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
// Async iterable merge
// ---------------------------------------------------------------------------

/**
 * Merge multiple async iterables, yielding values as soon as any source
 * produces one.  Order across sources is non-deterministic (whichever
 * source's `.next()` settles first wins).
 */
async function* mergeAsyncIterables<T>(
  iterables: AsyncIterable<T>[],
): AsyncGenerator<T> {
  const iterators = iterables.map((iter) => iter[Symbol.asyncIterator]());
  const active = new Map<
    number,
    Promise<{ result: IteratorResult<T>; index: number }>
  >();
  for (let i = 0; i < iterators.length; i++) {
    active.set(
      i,
      iterators[i].next().then((result) => ({ result, index: i })),
    );
  }
  while (active.size > 0) {
    const { result, index } = await Promise.race(active.values());
    if (result.done) {
      active.delete(index);
    } else {
      yield result.value;
      active.set(
        index,
        iterators[index].next().then((r) => ({ result: r, index })),
      );
    }
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

// ---------------------------------------------------------------------------
// Language model interpolation
// ---------------------------------------------------------------------------

/** Collect all items from an async iterable. */
async function collectAll<T>(iter: AsyncIterable<T>): Promise<T[]> {
  const result: T[] = [];
  for await (const item of iter) result.push(item);
  return result;
}

/**
 * Interpolate between two language models, creating a per-conditional mixture.
 *
 * P_mix(token | prefix) = (1 - fraction) * P_a(token | prefix)
 *                        +      fraction  * P_b(token | prefix)
 *
 * Both underlying models are queried in parallel with full range and minSize 0,
 * since the mixture changes cumulative positions.  The caller's rangeStart /
 * rangeEnd / minSize / specificToken filters are applied after merging.
 *
 * Token ordering smoothly interpolates between model A's and model B's orderings
 * via weighted-average sort keys: wA * startA + wB * startB.  Tokens absent from
 * a model use position 1 (end-of-distribution) as the sentinel, so at fraction=0
 * the ordering exactly matches model A, and at fraction=1 it matches model B.
 */
export function interpolate<P, T>(
  a: LanguageModel<P, T>,
  b: LanguageModel<P, T>,
  fraction: number,
): LanguageModel<P, T> {
  const wA = 1 - fraction;
  const wB = fraction;

  return async function* (prefix, rangeStart, rangeEnd, minSize, specificToken?) {
    // Query both models for the full distribution in parallel.
    const [aEntries, bEntries] = await Promise.all([
      collectAll(a(prefix, 0, 1, 0)),
      collectAll(b(prefix, 0, 1, 0)),
    ]);

    // Index model B entries by token.
    const bByToken = new Map<T, TokenProb<T>>();
    for (const e of bEntries) bByToken.set(e.token, e);

    // Merge: compute mixture probabilities and blended sort keys.
    const seen = new Set<T>();
    const merged: { token: T; prob: number; sortKey: number }[] = [];

    for (const ae of aEntries) {
      seen.add(ae.token);
      const pa = ae.end - ae.start;
      const be = bByToken.get(ae.token);
      const pb = be ? be.end - be.start : 0;
      const prob = wA * pa + wB * pb;
      if (prob === 0) continue;
      merged.push({
        token: ae.token,
        prob,
        sortKey: wA * ae.start + wB * (be?.start ?? 1),
      });
    }

    for (const be of bEntries) {
      if (seen.has(be.token)) continue;
      const pb = be.end - be.start;
      const prob = wB * pb;
      if (prob === 0) continue;
      merged.push({
        token: be.token,
        prob,
        sortKey: wA + wB * be.start,
      });
    }

    merged.sort((x, y) => x.sortKey - y.sortKey);

    // Assign cumulative positions and yield.
    let cum = 0;
    for (const entry of merged) {
      const start = cum;
      const end = cum + entry.prob;
      cum = end;

      if (specificToken !== undefined) {
        if (entry.token === specificToken) {
          yield { token: entry.token, start, end };
          return;
        }
        continue;
      }

      if (end < rangeStart || start > rangeEnd) continue;
      if (entry.prob < minSize) continue;
      yield { token: entry.token, start, end };
    }
  };
}
