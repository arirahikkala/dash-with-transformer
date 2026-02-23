import { describe, it, expect } from "vitest";
import { fromCharCodeModel } from "./models";
import { adaptModel, type PlainTokenProb } from "./types";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/** Collect all items from an async iterable into an array. */
async function collect<T>(iter: AsyncIterable<T>): Promise<T[]> {
  const result: T[] = [];
  for await (const item of iter) result.push(item);
  return result;
}

/** Build a char code LanguageModel from a table keyed by comma-joined hex prefix. */
function makeCharCodeModel(
  table: Record<string, PlainTokenProb<number>[]>,
) {
  return adaptModel<readonly number[], number>(async (prefix) => {
    const key = prefix.map((n) => n.toString(16)).join(",");
    return table[key] ?? [];
  });
}

// ---------------------------------------------------------------------------
// BMP-only pass-through
// ---------------------------------------------------------------------------

describe("BMP-only pass-through", () => {
  const model = makeCharCodeModel({
    "": [
      { token: 97, probability: 0.75 }, // 'a'
      { token: 98, probability: 0.25 }, // 'b'
    ],
  });

  it("codepoint equals char code for BMP", async () => {
    const lm = fromCharCodeModel(model);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toEqual([
      { token: 97, start: 0, end: 0.75 },
      { token: 98, start: 0.75, end: 1.0 },
    ]);
  });
});

// ---------------------------------------------------------------------------
// Surrogate pair expansion
// ---------------------------------------------------------------------------

describe("surrogate pair expansion", () => {
  // U+1F600 (grinning face) → high 0xD83D, low 0xDE00
  const model = makeCharCodeModel({
    "": [{ token: 0xd83d, probability: 1.0 }],
    d83d: [{ token: 0xde00, probability: 1.0 }],
  });

  it("decodes surrogate pair to correct codepoint", async () => {
    const lm = fromCharCodeModel(model);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toBe(0x1f600);
    expect(result[0].start).toBe(0);
    expect(result[0].end).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// Mixed BMP + surrogate pairs
// ---------------------------------------------------------------------------

describe("mixed BMP + surrogate pairs", () => {
  // 'a' (0.5) + U+1F600 (0.5)
  // U+1F600 → high 0xD83D, low 0xDE00
  const model = makeCharCodeModel({
    "": [
      { token: 97, probability: 0.5 },
      { token: 0xd83d, probability: 0.5 },
    ],
    d83d: [{ token: 0xde00, probability: 1.0 }],
  });

  it("yields both BMP and non-BMP codepoints", async () => {
    const lm = fromCharCodeModel(model);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(2);

    const a = result.find((e) => e.token === 97)!;
    expect(a).toEqual({ token: 97, start: 0, end: 0.5 });

    const emoji = result.find((e) => e.token === 0x1f600)!;
    expect(emoji).toEqual({ token: 0x1f600, start: 0.5, end: 1.0 });
  });

  it("produces contiguous entries", async () => {
    const lm = fromCharCodeModel(model);
    const result = await collect(lm([], 0, 1, 0));
    result.sort((a, b) => a.start - b.start);

    expect(result[0].start).toBe(0);
    for (let i = 1; i < result.length; i++) {
      expect(result[i].start).toBe(result[i - 1].end);
    }
    expect(result[result.length - 1].end).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// Multiple non-BMP codepoints via surrogate expansion
// ---------------------------------------------------------------------------

describe("multiple surrogate expansions", () => {
  // Two emoji via same high surrogate 0xD83D:
  // U+1F600 → D83D DE00 (0.5)
  // U+1F601 → D83D DE01 (0.5)
  const model = makeCharCodeModel({
    "": [{ token: 0xd83d, probability: 1.0 }],
    d83d: [
      { token: 0xde00, probability: 0.5 },
      { token: 0xde01, probability: 0.5 },
    ],
  });

  it("expands to correct codepoints with correct extents", async () => {
    const lm = fromCharCodeModel(model);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(2);

    const e0 = result.find((e) => e.token === 0x1f600)!;
    expect(e0.start).toBe(0);
    expect(e0.end).toBe(0.5);

    const e1 = result.find((e) => e.token === 0x1f601)!;
    expect(e1.start).toBe(0.5);
    expect(e1.end).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// Range filtering on surrogates
// ---------------------------------------------------------------------------

describe("range filtering on surrogates", () => {
  // 'a' at [0, 0.5], U+1F600 at [0.5, 0.75], U+1F601 at [0.75, 1.0]
  const model = makeCharCodeModel({
    "": [
      { token: 97, probability: 0.5 },
      { token: 0xd83d, probability: 0.5 },
    ],
    d83d: [
      { token: 0xde00, probability: 0.5 },
      { token: 0xde01, probability: 0.5 },
    ],
  });

  it("returns only codepoints overlapping the range", async () => {
    const lm = fromCharCodeModel(model);
    // range [0.51, 0.74] should only include U+1F600 at [0.5, 0.75]
    const result = await collect(lm([], 0.51, 0.74, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toBe(0x1f600);
  });
});

// ---------------------------------------------------------------------------
// minSize filtering
// ---------------------------------------------------------------------------

describe("minSize filtering on surrogates", () => {
  const model = makeCharCodeModel({
    "": [
      { token: 97, probability: 0.5 },
      { token: 0xd83d, probability: 0.5 },
    ],
    d83d: [
      { token: 0xde00, probability: 0.5 },
      { token: 0xde01, probability: 0.5 },
    ],
  });

  it("excludes expanded surrogates below threshold", async () => {
    const lm = fromCharCodeModel(model);
    // Each emoji has absolute prob 0.25, minSize 0.3 excludes them
    const result = await collect(lm([], 0, 1, 0.3));

    expect(result).toHaveLength(1);
    expect(result[0].token).toBe(97);
  });
});

// ---------------------------------------------------------------------------
// specificToken for BMP
// ---------------------------------------------------------------------------

describe("specificToken for BMP", () => {
  const model = makeCharCodeModel({
    "": [
      { token: 97, probability: 0.75 },
      { token: 98, probability: 0.25 },
    ],
  });

  it("returns correct extent for BMP codepoint", async () => {
    const lm = fromCharCodeModel(model);
    const result = await collect(lm([], 0, 0, 0, 98));

    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({ token: 98, start: 0.75, end: 1.0 });
  });
});

// ---------------------------------------------------------------------------
// specificToken for non-BMP
// ---------------------------------------------------------------------------

describe("specificToken for non-BMP", () => {
  const model = makeCharCodeModel({
    "": [
      { token: 97, probability: 0.5 },
      { token: 0xd83d, probability: 0.5 },
    ],
    d83d: [
      { token: 0xde00, probability: 0.5 },
      { token: 0xde01, probability: 0.5 },
    ],
  });

  it("returns correct extent for surrogate-pair codepoint", async () => {
    const lm = fromCharCodeModel(model);
    const result = await collect(lm([], 0, 0, 0, 0x1f601));

    expect(result).toHaveLength(1);
    expect(result[0].token).toBe(0x1f601);
    expect(result[0].start).toBe(0.75);
    expect(result[0].end).toBe(1.0);
  });

  it("returns empty for non-existent codepoint", async () => {
    const lm = fromCharCodeModel(model);
    const result = await collect(lm([], 0, 0, 0, 0x1f602));

    expect(result).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Extent determinism
// ---------------------------------------------------------------------------

describe("extent determinism", () => {
  const model = makeCharCodeModel({
    "": [
      { token: 97, probability: 0.5 },
      { token: 0xd83d, probability: 0.5 },
    ],
    d83d: [
      { token: 0xde00, probability: 0.5 },
      { token: 0xde01, probability: 0.5 },
    ],
  });

  it("produces same extents regardless of range", async () => {
    const lm = fromCharCodeModel(model);
    const full = await collect(lm([], 0, 1, 0));
    const narrow = await collect(lm([], 0.6, 0.9, 0));

    const emojiFull = full.find((e) => e.token === 0x1f600)!;
    const emojiNarrow = narrow.find((e) => e.token === 0x1f600)!;

    expect(emojiNarrow.start).toBe(emojiFull.start);
    expect(emojiNarrow.end).toBe(emojiFull.end);
  });

  it("produces same extents regardless of minSize", async () => {
    const lm = fromCharCodeModel(model);
    const all = await collect(lm([], 0, 1, 0));
    const filtered = await collect(lm([], 0, 1, 0.2));

    const aAll = all.find((e) => e.token === 97)!;
    const aFiltered = filtered.find((e) => e.token === 97)!;

    expect(aFiltered.start).toBe(aAll.start);
    expect(aFiltered.end).toBe(aAll.end);
  });
});

// ---------------------------------------------------------------------------
// Codepoint prefix encoding
// ---------------------------------------------------------------------------

describe("codepoint prefix encoding", () => {
  it("encodes BMP prefix correctly", async () => {
    const model = makeCharCodeModel({
      "61": [{ token: 98, probability: 1.0 }], // after 'a', predict 'b'
    });
    const lm = fromCharCodeModel(model);
    const result = await collect(lm([0x61], 0, 1, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toBe(98);
  });

  it("encodes non-BMP prefix as surrogate pair", async () => {
    // U+1F600 → D83D DE00
    const model = makeCharCodeModel({
      "d83d,de00": [{ token: 97, probability: 1.0 }],
    });
    const lm = fromCharCodeModel(model);
    const result = await collect(lm([0x1f600], 0, 1, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toBe(97);
  });
});

// ---------------------------------------------------------------------------
// Lone low surrogates skipped
// ---------------------------------------------------------------------------

describe("lone low surrogates skipped", () => {
  it("omits lone low surrogates from results", async () => {
    const model = makeCharCodeModel({
      "": [
        { token: 97, probability: 0.5 },
        { token: 0xdc00, probability: 0.5 }, // lone low surrogate
      ],
    });
    const lm = fromCharCodeModel(model);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toBe(97);
  });
});
