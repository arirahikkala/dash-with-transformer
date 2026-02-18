/**
 * Adapt a byte-level UTF-8 language model into a codepoint-level model.
 */
import { type Rat, fromFloat, toFloat, add, mul, ZERO } from "./rational";
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
// Multi-byte expansion
// ---------------------------------------------------------------------------

type ByteLevelModel = (bytePrefix: Uint8Array) => Promise<number[]>;

/**
 * Recursively expand a partial UTF-8 sequence into filtered TokenProb entries
 * by querying the byte-level model for continuation bytes.
 *
 * At each level the cumulative positions of all sub-groups are computed
 * (so that later sub-groups have correct absolute positions even when
 * earlier ones are skipped), and only sub-groups that overlap the visible
 * range and meet the minimum-size threshold are recursed into.
 *
 * Returns entries in byte order (= codepoint order within a fixed lead byte).
 * All continuation queries at the same depth run in parallel.
 */
async function expandMultiByte(
  model: ByteLevelModel,
  bytePrefix: Uint8Array,
  partialBytes: number[],
  totalBytes: number,
  probSoFar: Rat,
  cumStart: Rat,
  rangeStart: number,
  rangeEnd: number,
  minSize: number,
): Promise<TokenProb<number>[]> {
  if (partialBytes.length === totalBytes) {
    const start = toFloat(cumStart);
    const end = toFloat(add(cumStart, probSoFar));
    if (end <= rangeStart || start >= rangeEnd) return [];
    if (end - start < minSize) return [];
    return [{ token: decodeUtf8Bytes(partialBytes), start, end }];
  }

  const queryPrefix = new Uint8Array([...bytePrefix, ...partialBytes]);
  const dist = await model(queryPrefix);

  // Single pass: accumulate cumulative positions for ALL non-zero
  // continuation bytes (so later sub-groups are positioned correctly),
  // but only recurse into those that pass the range/size filters.
  const tasks: Promise<TokenProb<number>[]>[] = [];
  let cum = cumStart;
  for (let b = 0; b < 256; b++) {
    if (dist[b] === 0) continue;
    const subProb = mul(probSoFar, fromFloat(dist[b]));
    const subCumStart = cum;
    cum = add(cum, subProb);

    const subStart = toFloat(subCumStart);
    const subEnd = toFloat(cum);
    if (subEnd <= rangeStart || subStart >= rangeEnd) continue;
    if (subEnd - subStart < minSize) continue;

    tasks.push(
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

  const results = await Promise.all(tasks);
  return results.flat();
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
 * depth run concurrently via Promise.all.
 *
 * Cumulative extents use exact rational arithmetic so that a codepoint's
 * position depends only on the prefix, never the query window.
 */
export function fromByteLevelModel(
  byteLevelModel: ByteLevelModel,
): LanguageModel<number[], number> {
  return async (
    prefix: number[],
    rangeStart: number,
    rangeEnd: number,
    minSize: number,
    specificToken?: number,
  ): Promise<readonly TokenProb<number>[]> => {
    const bytePrefix = codepointsToUtf8(prefix);

    // Fast path: look up a single codepoint's extent.
    if (specificToken !== undefined) {
      const targetBytes = [...codepointsToUtf8([specificToken])];
      const leadByte = targetBytes[0];

      const firstByteDist = await byteLevelModel(bytePrefix);
      if (firstByteDist[leadByte] === 0) return [];

      // Cumulative start for the lead byte.
      let cumStart: Rat = ZERO;
      for (let b = 0; b < leadByte; b++) {
        if (firstByteDist[b] > 0) {
          cumStart = add(cumStart, fromFloat(firstByteDist[b]));
        }
      }
      let probSoFar: Rat = fromFloat(firstByteDist[leadByte]);

      // For multi-byte sequences, descend through continuation bytes.
      for (let i = 1; i < targetBytes.length; i++) {
        const queryPrefix = new Uint8Array([
          ...bytePrefix,
          ...targetBytes.slice(0, i),
        ]);
        const dist = await byteLevelModel(queryPrefix);
        const targetByte = targetBytes[i];

        if (dist[targetByte] === 0) return [];

        for (let b = 0; b < targetByte; b++) {
          if (dist[b] > 0) {
            cumStart = add(cumStart, mul(probSoFar, fromFloat(dist[b])));
          }
        }
        probSoFar = mul(probSoFar, fromFloat(dist[targetByte]));
      }

      const start = toFloat(cumStart);
      const end = toFloat(add(cumStart, probSoFar));
      return [{ token: specificToken, start, end }];
    }

    // 1. First-byte distribution (1 model call).
    const firstByteDist = await byteLevelModel(bytePrefix);

    // 2. Exact cumulative start position for every first-byte group.
    //    P(all codepoints starting with byte b) = P(b), so the
    //    cumulative over first bytes gives correct group boundaries.
    const firstByteProbs: Rat[] = new Array(256);
    const firstByteCumStart: Rat[] = new Array(256);
    let cum: Rat = ZERO;
    for (let b = 0; b < 256; b++) {
      firstByteCumStart[b] = cum;
      if (firstByteDist[b] > 0) {
        firstByteProbs[b] = fromFloat(firstByteDist[b]);
        cum = add(cum, firstByteProbs[b]);
      } else {
        firstByteProbs[b] = ZERO;
      }
    }

    const result: TokenProb<number>[] = [];

    // 3. Single-byte codepoints (0x00–0x7F): emit directly.
    for (let b = 0; b <= 0x7f; b++) {
      if (firstByteDist[b] === 0) continue;
      const start = toFloat(firstByteCumStart[b]);
      const end = toFloat(add(firstByteCumStart[b], firstByteProbs[b]));
      if (end <= rangeStart || start >= rangeEnd) continue;
      if (end - start < minSize) continue;
      result.push({ token: b, start, end });
    }

    // 4. Multi-byte groups: decide which need expansion.
    type GroupInfo = {
      firstByte: number;
      totalBytes: number;
      prob: Rat;
      cumStart: Rat;
    };
    const groupsToExpand: GroupInfo[] = [];

    for (let b = 0xc0; b <= 0xff; b++) {
      if (firstByteDist[b] === 0) continue;
      const totalBytes = utf8SeqLength(b);
      if (totalBytes === 0) continue; // invalid lead byte

      const groupStartF = toFloat(firstByteCumStart[b]);
      const groupEndF = toFloat(add(firstByteCumStart[b], firstByteProbs[b]));

      // Skip groups entirely outside the visible range.
      if (groupEndF <= rangeStart || groupStartF >= rangeEnd) continue;
      // Skip groups where every codepoint is below minSize.
      if (firstByteDist[b] < minSize) continue;

      groupsToExpand.push({
        firstByte: b,
        totalBytes,
        prob: firstByteProbs[b],
        cumStart: firstByteCumStart[b],
      });
    }

    // 5. Expand all needed groups in parallel (filtering happens inside).
    const expansions = await Promise.all(
      groupsToExpand.map((g) =>
        expandMultiByte(
          byteLevelModel,
          bytePrefix,
          [g.firstByte],
          g.totalBytes,
          g.prob,
          g.cumStart,
          rangeStart,
          rangeEnd,
          minSize,
        ),
      ),
    );

    // 6. Append filtered results.
    for (const entries of expansions) {
      result.push(...entries);
    }

    return result;
  };
}
