import { describe, it, expect } from "vitest";
import { detokenize } from "./detokenize";
import type { PlainLanguageModel, PlainTokenProb } from "../types";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/** Collect all items from an async iterable into an array. */
async function collect<T>(iter: AsyncIterable<T>): Promise<T[]> {
  const result: T[] = [];
  for await (const item of iter) result.push(item);
  return result;
}

/** Build a mock PlainLanguageModel from a table keyed by JSON-encoded prefix. */
function makePlainModel(
  table: Record<string, PlainTokenProb<string>[]>,
): PlainLanguageModel<readonly string[], string> {
  return async (prefix) => {
    const key = JSON.stringify([...prefix]);
    return table[key] ?? [];
  };
}

// ---------------------------------------------------------------------------
// Single-character tokens
// ---------------------------------------------------------------------------

describe("single-character tokens", () => {
  const vocab = ["a", "b"];
  const model = makePlainModel({
    "[]": [
      { token: "a", probability: 0.75 },
      { token: "b", probability: 0.25 },
    ],
  });

  it("produces correct extents", async () => {
    const lm = detokenize(model, vocab, 8);
    const result = await collect(lm([], 0, 1, 0));

    // sorted by char code: a=97, b=98
    expect(result).toEqual([
      { token: 97, start: 0, end: 0.75 },
      { token: 98, start: 0.75, end: 1.0 },
    ]);
  });

  it("produces contiguous coverage", async () => {
    const lm = detokenize(model, vocab, 8);
    const result = await collect(lm([], 0, 1, 0));

    expect(result[0].start).toBe(0);
    for (let i = 1; i < result.length; i++) {
      expect(result[i].start).toBe(result[i - 1].end);
    }
    expect(result[result.length - 1].end).toBeCloseTo(1.0);
  });
});

// ---------------------------------------------------------------------------
// Multi-character tokens
// ---------------------------------------------------------------------------

describe("multi-character tokens", () => {
  const vocab = ["ab", "cd"];
  const model = makePlainModel({
    "[]": [
      { token: "ab", probability: 0.5 },
      { token: "cd", probability: 0.5 },
    ],
  });

  it("predicts first characters at empty prefix", async () => {
    const lm = detokenize(model, vocab, 8);
    const result = await collect(lm([], 0, 1, 0));

    // 'a'=97 gets P=0.5, 'c'=99 gets P=0.5
    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({ token: 97, start: 0, end: 0.5 });
    expect(result[1]).toEqual({ token: 99, start: 0.5, end: 1.0 });
  });

  it("predicts second character after first", async () => {
    const lm = detokenize(model, vocab, 8);
    // After 'a' (charCode 97), only 'b' (98) should be predicted
    const result = await collect(lm([97], 0, 1, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toBe(98);
    expect(result[0].end).toBeCloseTo(1.0);
  });
});

// ---------------------------------------------------------------------------
// Beam extension (shared prefix)
// ---------------------------------------------------------------------------

describe("beam extension (shared prefix)", () => {
  const vocab = ["a", "ab", "b"];
  const model = makePlainModel({
    "[]": [
      { token: "a", probability: 0.5 },
      { token: "ab", probability: 0.25 },
      { token: "b", probability: 0.25 },
    ],
    '["a"]': [
      { token: "a", probability: 0.5 },
      { token: "ab", probability: 0.25 },
      { token: "b", probability: 0.25 },
    ],
  });

  it("marginalizes over beam at prefix 'a'", async () => {
    const lm = detokenize(model, vocab, 8);
    const result = await collect(lm([97], 0, 1, 0));

    // After char 'a', the beam has:
    // 1. candidate at trie node 'a' (partway through "a"/"ab"), direct advance
    // 2. candidate at trie root (committed token "a"), extension
    // Both contribute to next-char predictions
    expect(result.length).toBeGreaterThan(0);

    // 'a' and 'b' should both get contributions
    const aEntry = result.find((e) => e.token === 97);
    const bEntry = result.find((e) => e.token === 98);
    expect(aEntry).toBeDefined();
    expect(bEntry).toBeDefined();

    // Both should be 0.5 (contribution from trie advance + extension)
    expect(aEntry!.end - aEntry!.start).toBeCloseTo(0.5);
    expect(bEntry!.end - bEntry!.start).toBeCloseTo(0.5);
  });

  it("produces contiguous entries", async () => {
    const lm = detokenize(model, vocab, 8);
    const result = await collect(lm([97], 0, 1, 0));
    result.sort((a, b) => a.start - b.start);

    expect(result[0].start).toBeCloseTo(0);
    for (let i = 1; i < result.length; i++) {
      expect(result[i].start).toBeCloseTo(result[i - 1].end);
    }
    expect(result[result.length - 1].end).toBeCloseTo(1.0);
  });
});

// ---------------------------------------------------------------------------
// Range filtering
// ---------------------------------------------------------------------------

describe("range filtering", () => {
  const vocab = ["a", "b", "c"];
  const model = makePlainModel({
    "[]": [
      { token: "a", probability: 0.5 },
      { token: "b", probability: 0.25 },
      { token: "c", probability: 0.25 },
    ],
  });

  it("excludes entries outside the range", async () => {
    const lm = detokenize(model, vocab, 8);
    // a=[0, 0.5], b=[0.5, 0.75], c=[0.75, 1.0]
    // range [0.51, 0.74] should only return b
    const result = await collect(lm([], 0.51, 0.74, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toBe(98); // 'b'
  });

  it("preserves extents regardless of range", async () => {
    const lm = detokenize(model, vocab, 8);
    const full = await collect(lm([], 0, 1, 0));
    const narrow = await collect(lm([], 0.51, 0.74, 0));

    const bFull = full.find((e) => e.token === 98)!;
    const bNarrow = narrow.find((e) => e.token === 98)!;

    expect(bNarrow.start).toBe(bFull.start);
    expect(bNarrow.end).toBe(bFull.end);
  });
});

// ---------------------------------------------------------------------------
// minProb filtering
// ---------------------------------------------------------------------------

describe("minProb filtering", () => {
  const vocab = ["a", "b", "c"];
  const model = makePlainModel({
    "[]": [
      { token: "a", probability: 0.5 },
      { token: "b", probability: 0.25 },
      { token: "c", probability: 0.25 },
    ],
  });

  it("excludes entries below minProb", async () => {
    const lm = detokenize(model, vocab, 8);
    // a=0.5, b=0.25, c=0.25 — minProb 0.3 keeps only a
    const result = await collect(lm([], 0, 1, 0.3));

    expect(result).toHaveLength(1);
    expect(result[0].token).toBe(97); // 'a'
  });

  it("keeps entries at exactly minProb", async () => {
    const lm = detokenize(model, vocab, 8);
    const result = await collect(lm([], 0, 1, 0.25));
    expect(result).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// specificToken
// ---------------------------------------------------------------------------

describe("specificToken", () => {
  const vocab = ["a", "b", "c"];
  const model = makePlainModel({
    "[]": [
      { token: "a", probability: 0.5 },
      { token: "b", probability: 0.25 },
      { token: "c", probability: 0.25 },
    ],
  });

  it("returns only the requested token's extent", async () => {
    const lm = detokenize(model, vocab, 8);
    const result = await collect(lm([], 0, 0, 0, 98)); // 'b'

    expect(result).toHaveLength(1);
    expect(result[0].token).toBe(98);
    expect(result[0].start).toBe(0.5);
    expect(result[0].end).toBe(0.75);
  });

  it("returns empty if token not in distribution", async () => {
    const lm = detokenize(model, vocab, 8);
    const result = await collect(lm([], 0, 0, 0, 120)); // 'x'

    expect(result).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Contiguity invariant
// ---------------------------------------------------------------------------

describe("contiguity invariant", () => {
  it("entries sorted by start are contiguous (varied models)", async () => {
    const vocabs: string[][] = [
      ["a", "b", "c"],
      ["ab", "cd", "ef"],
      ["a", "ab", "b"],
    ];
    const models = [
      makePlainModel({
        "[]": [
          { token: "a", probability: 0.5 },
          { token: "b", probability: 0.25 },
          { token: "c", probability: 0.25 },
        ],
      }),
      makePlainModel({
        "[]": [
          { token: "ab", probability: 0.5 },
          { token: "cd", probability: 0.25 },
          { token: "ef", probability: 0.25 },
        ],
      }),
      makePlainModel({
        "[]": [
          { token: "a", probability: 0.5 },
          { token: "ab", probability: 0.25 },
          { token: "b", probability: 0.25 },
        ],
        '["a"]': [
          { token: "a", probability: 0.5 },
          { token: "ab", probability: 0.25 },
          { token: "b", probability: 0.25 },
        ],
      }),
    ];

    for (let i = 0; i < vocabs.length; i++) {
      const lm = detokenize(models[i], vocabs[i], 8);
      const result = await collect(lm([], 0, 1, 0));
      result.sort((a, b) => a.start - b.start);

      expect(result[0].start).toBeCloseTo(0);
      for (let j = 1; j < result.length; j++) {
        expect(result[j].start).toBeCloseTo(result[j - 1].end);
      }
      expect(result[result.length - 1].end).toBeCloseTo(1.0);
    }
  });
});

// ---------------------------------------------------------------------------
// Context-sensitivity
// ---------------------------------------------------------------------------

describe("context-sensitivity", () => {
  it("returns different distributions for different prefixes", async () => {
    const vocab = ["a", "b"];
    const model = makePlainModel({
      "[]": [
        { token: "a", probability: 0.75 },
        { token: "b", probability: 0.25 },
      ],
      '["a"]': [
        { token: "a", probability: 0.25 },
        { token: "b", probability: 0.75 },
      ],
    });

    const lm = detokenize(model, vocab, 8);

    // At empty prefix: a dominates
    const r0 = await collect(lm([], 0, 1, 0));
    const a0 = r0.find((e) => e.token === 97)!;
    expect(a0.end - a0.start).toBeCloseTo(0.75);

    // After "a": b dominates (model says P(b|["a"])=0.75)
    // The beam commits "a" at node 'a' and starts fresh with model(["a"])
    const r1 = await collect(lm([97], 0, 1, 0));
    const b1 = r1.find((e) => e.token === 98)!;
    expect(b1.end - b1.start).toBeCloseTo(0.75);
  });
});
