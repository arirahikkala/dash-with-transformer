/**
 * Adapt a UTF-16 char code level model into a Unicode codepoint level model.
 *
 * BMP codepoints (U+0000–U+D7FF, U+E000–U+FFFF) pass through directly.
 * Non-BMP codepoints (U+10000–U+10FFFF) are decoded from surrogate pairs.
 */
import { mergeAsyncIterables } from "../async-iterables";
import { first, type LanguageModel, type TokenProb } from "../types";

// ---------------------------------------------------------------------------
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
  minProb: number,
): AsyncGenerator<TokenProb<number>> {
  const high = highEntry.token;
  const groupSize = highEntry.end - highEntry.start;

  const subRangeStart = Math.max(0, (rangeStart - highEntry.start) / groupSize);
  const subRangeEnd = Math.min(1, (rangeEnd - highEntry.start) / groupSize);
  const subMinProb = minProb / groupSize;

  for await (const lowEntry of charCodeModel(
    [...charCodePrefix, high],
    subRangeStart,
    subRangeEnd,
    subMinProb,
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
    minProb: number,
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
      minProb,
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
            minProb,
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
