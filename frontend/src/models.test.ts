import { describe, it, expect } from "vitest";
import {
  fromByteLevelModel,
  forceCleanUtf8,
  byteOnly,
  interpolate,
  type ByteLevelModel,
} from "./models";
import {
  type LanguageModel,
  type SpecialToken,
  type UnicodeCodepoint,
  asNormalized,
} from "./types";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/** Shorthand for a UnicodeCodepoint token. */
const cp = (n: number): UnicodeCodepoint => ({
  type: "codepoint",
  codepoint: n,
});

/** Collect all items from an async iterable into an array. */
async function collect<T>(iter: AsyncIterable<T>): Promise<T[]> {
  const result: T[] = [];
  for await (const item of iter) {
    result.push(item);
  }
  return result;
}

/** Build a 256-element probability array with specific non-zero entries. */
function makeDist(entries: Record<number, number>): number[] {
  const dist = new Array(256).fill(0);
  for (const [byte, prob] of Object.entries(entries)) {
    dist[Number(byte)] = prob;
  }
  return dist;
}

/**
 * Build a distribution with byte entries (indices 0–255) and optional
 * extra entries for special tokens (indices ≥ 256).
 */
function makeDistWithSpecial(
  byteEntries: Record<number, number>,
  specialProbs: number[],
): number[] {
  return [...makeDist(byteEntries), ...specialProbs];
}

/** Hex key for a byte prefix (empty prefix → ""). */
function prefixKey(buf: readonly number[]): string {
  return buf.map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Create a mock byte-level model from a hex-key → dist table. */
function makeMockModel(table: Record<string, number[]>): ByteLevelModel {
  return async (prefix: readonly number[]) => {
    const key = prefixKey(prefix);
    const result = table[key];
    if (!result) throw new Error(`Unexpected byte prefix: "${key}"`);
    return result;
  };
}

/** Like makeMockModel but also records the hex key of every call. */
function makeTrackingModel(table: Record<string, number[]>): {
  model: ByteLevelModel;
  calls: string[];
} {
  const calls: string[] = [];
  const model: ByteLevelModel = async (prefix: readonly number[]) => {
    const key = prefixKey(prefix);
    calls.push(key);
    const result = table[key];
    if (!result) throw new Error(`Unexpected byte prefix: "${key}"`);
    return result;
  };
  return { model, calls };
}

// ---------------------------------------------------------------------------
// Single-byte (ASCII) codepoints
// ---------------------------------------------------------------------------

describe("ASCII-only model", () => {
  // Use power-of-2 probabilities for exact float comparisons.
  const table: Record<string, number[]> = {
    "": makeDist({ 97: 0.5, 98: 0.25, 99: 0.25 }), // a, b, c
  };

  it("returns correct tokens and extents", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(3);
    expect(result[0]).toEqual({ token: cp(97), start: 0, end: 0.5 });
    expect(result[1]).toEqual({ token: cp(98), start: 0.5, end: 0.75 });
    expect(result[2]).toEqual({ token: cp(99), start: 0.75, end: 1.0 });
  });

  it("returns tokens in ascending codepoint order", async () => {
    // Deliberately list bytes out of natural order in the dist —
    // the model still iterates 0..255 internally.
    const lm = fromByteLevelModel(
      makeMockModel({ "": makeDist({ 99: 0.25, 97: 0.5, 98: 0.25 }) }),
      [],
    );
    const tokens = (await collect(lm([], 0, 1, 0))).map((e) => e.token);
    expect(tokens).toEqual([cp(97), cp(98), cp(99)]);
  });
});

// ---------------------------------------------------------------------------
// Two-byte codepoints
// ---------------------------------------------------------------------------

describe("two-byte codepoints", () => {
  // 'è' = U+00E8 → C3 A8,  'é' = U+00E9 → C3 A9
  const table: Record<string, number[]> = {
    "": makeDist({ 97: 0.5, 0xc3: 0.5 }),
    c3: makeDist({ 0xa8: 0.5, 0xa9: 0.5 }),
  };

  it("computes product probabilities correctly", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(3);
    expect(result[0]).toEqual({ token: cp(97), start: 0, end: 0.5 }); // a
    expect(result[1]).toEqual({ token: cp(0xe8), start: 0.5, end: 0.75 }); // è
    expect(result[2]).toEqual({ token: cp(0xe9), start: 0.75, end: 1.0 }); // é
  });
});

// ---------------------------------------------------------------------------
// Three-byte codepoints
// ---------------------------------------------------------------------------

describe("three-byte codepoints", () => {
  // '中' = U+4E2D → E4 B8 AD
  const table: Record<string, number[]> = {
    "": makeDist({ 0xe4: 1.0 }),
    e4: makeDist({ 0xb8: 1.0 }),
    e4b8: makeDist({ 0xad: 1.0 }),
  };

  it("resolves to the correct codepoint", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(cp(0x4e2d));
    expect(result[0].start).toBe(0);
    expect(result[0].end).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Four-byte codepoints
// ---------------------------------------------------------------------------

describe("four-byte codepoints", () => {
  // '😀' = U+1F600 → F0 9F 98 80
  const table: Record<string, number[]> = {
    "": makeDist({ 0xf0: 1.0 }),
    f0: makeDist({ 0x9f: 1.0 }),
    f09f: makeDist({ 0x98: 1.0 }),
    f09f98: makeDist({ 0x80: 1.0 }),
  };

  it("resolves to the correct codepoint", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(cp(0x1f600));
    expect(result[0].start).toBe(0);
    expect(result[0].end).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Mixed single + multi-byte
// ---------------------------------------------------------------------------

describe("mixed codepoints", () => {
  // a (0.5) + é (0.25) + 中 (0.25)
  const table: Record<string, number[]> = {
    "": makeDist({ 97: 0.5, 0xc3: 0.25, 0xe4: 0.25 }),
    c3: makeDist({ 0xa9: 1.0 }),
    e4: makeDist({ 0xb8: 1.0 }),
    e4b8: makeDist({ 0xad: 1.0 }),
  };

  it("orders and positions all codepoints correctly", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(3);
    expect(result[0]).toEqual({ token: cp(97), start: 0, end: 0.5 }); // a
    expect(result[1]).toEqual({ token: cp(0xe9), start: 0.5, end: 0.75 }); // é
    expect(result[2]).toEqual({ token: cp(0x4e2d), start: 0.75, end: 1.0 }); // 中
  });

  it("produces contiguous entries", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0, 1, 0));

    expect(result[0].start).toBe(0);
    for (let i = 1; i < result.length; i++) {
      expect(result[i].start).toBe(result[i - 1].end);
    }
    expect(result[result.length - 1].end).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Range filtering
// ---------------------------------------------------------------------------

describe("range filtering", () => {
  // a=[0, 0.5], è=[0.5, 0.75], é=[0.75, 1.0]
  const table: Record<string, number[]> = {
    "": makeDist({ 97: 0.5, 0xc3: 0.5 }),
    c3: makeDist({ 0xa8: 0.5, 0xa9: 0.5 }),
  };

  it("excludes entries outside the range", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    // [0.51, 0.74] excludes a (ends at 0.5) and é (starts at 0.75)
    const result = await collect(lm([], 0.51, 0.74, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(cp(0xe8)); // è
  });

  it("includes entries that partially overlap", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    // [0.4, 0.8) overlaps a, è, é
    const result = await collect(lm([], 0.4, 0.8, 0));
    const tokens = result.map((e) => e.token);
    expect(tokens).toEqual([cp(97), cp(0xe8), cp(0xe9)]);
  });
});

// ---------------------------------------------------------------------------
// Closed-range semantics
// ---------------------------------------------------------------------------

describe("half-open extent semantics", () => {
  // a=[0, 0.5], è=[0.5, 0.75], é=[0.75, 1.0]
  const table: Record<string, number[]> = {
    "": makeDist({ 97: 0.5, 0xc3: 0.5 }),
    c3: makeDist({ 0xa8: 0.5, 0xa9: 0.5 }),
  };

  it("point query returns the containing node", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    // 0.6 is inside è=[0.5, 0.75)
    const result = await collect(lm([], 0.6, 0.6, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(cp(0xe8));
  });

  it("point query at a boundary returns only the token starting there", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    // 0.5 is at the boundary: a=[0, 0.5) ends here, è=[0.5, 0.75) starts here.
    // With half-open extents, only è contains 0.5.
    const result = await collect(lm([], 0.5, 0.5, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(cp(0xe8)); // è (start = 0.5)
  });

  it("point query at a multi-byte boundary returns only the token starting there", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    // 0.75 is at the boundary: è=[0.5, 0.75) ends here, é=[0.75, 1.0) starts here.
    const result = await collect(lm([], 0.75, 0.75, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(cp(0xe9)); // é (start = 0.75)
  });

  it("range query excludes tokens whose extent ends at rangeStart", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    // a=[0, 0.5) ends at 0.5, è=[0.5, 0.75), é=[0.75, 1.0)
    // Range [0.5, 0.75]: è and é overlap, but a does not (end = rangeStart).
    const result = await collect(lm([], 0.5, 0.75, 0));

    expect(result).toHaveLength(2);
    expect(result.map((e) => e.token)).toEqual([cp(0xe8), cp(0xe9)]);
  });

  it("point query at 0 returns the first node", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0, 0, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(cp(97));
  });

  it("point query at 1 returns nothing (past all half-open extents)", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    // All extents are half-open: the last token é=[0.75, 1.0) does not contain 1.0.
    const result = await collect(lm([], 1, 1, 0));

    expect(result).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Half-open range semantics with multi-byte sub-groups
// ---------------------------------------------------------------------------

describe("half-open range with deep multi-byte", () => {
  // 3-byte: E4 → two second bytes, each with one third byte
  // 中 = E4 B8 AD at [0, 0.5),  乀 = E4 B9 80 at [0.5, 1.0)
  const table: Record<string, number[]> = {
    "": makeDist({ 0xe4: 1.0 }),
    e4: makeDist({ 0xb8: 0.5, 0xb9: 0.5 }),
    e4b8: makeDist({ 0xad: 1.0 }),
    e4b9: makeDist({ 0x80: 1.0 }),
  };

  it("point query at sub-group boundary returns only the token starting there", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    // 0.5 is the boundary: 中=[0, 0.5) ends here, 乀=[0.5, 1.0) starts here.
    const result = await collect(lm([], 0.5, 0.5, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(cp(0x4e40)); // 乀
  });

  it("point query inside a sub-group returns just that codepoint", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0.25, 0.25, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(cp(0x4e2d));
  });
});

// ---------------------------------------------------------------------------
// minProb filtering
// ---------------------------------------------------------------------------

describe("minProb filtering", () => {
  const table: Record<string, number[]> = {
    "": makeDist({ 97: 0.5, 0xc3: 0.5 }),
    c3: makeDist({ 0xa8: 0.5, 0xa9: 0.5 }),
  };

  it("excludes entries below minProb", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    // a=0.5, è=0.25, é=0.25  —  minProb 0.3 keeps only a
    const result = await collect(lm([], 0, 1, 0.3));

    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(cp(97));
  });

  it("keeps entries at exactly minProb", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0, 1, 0.25));
    expect(result).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// Extent determinism — same (start, end) regardless of query parameters
// ---------------------------------------------------------------------------

describe("extent determinism", () => {
  const table: Record<string, number[]> = {
    "": makeDist({ 97: 0.5, 0xc3: 0.5 }),
    c3: makeDist({ 0xa8: 0.5, 0xa9: 0.5 }),
  };

  it("produces identical extents for different range queries", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);

    const full = await collect(lm([], 0, 1, 0));
    const narrow = await collect(lm([], 0.6, 0.9, 0));

    // Both queries should include è — verify same extent.
    const eFull = full.find(
      (e) => e.token.type === "codepoint" && e.token.codepoint === 0xe8,
    )!;
    const eNarrow = narrow.find(
      (e) => e.token.type === "codepoint" && e.token.codepoint === 0xe8,
    )!;

    expect(eNarrow.start).toBe(eFull.start);
    expect(eNarrow.end).toBe(eFull.end);
  });

  it("produces identical extents for different minProb queries", async () => {
    const lm = fromByteLevelModel(makeMockModel(table), []);

    const all = await collect(lm([], 0, 1, 0));
    const filtered = await collect(lm([], 0, 1, 0.2));

    const aAll = all.find(
      (e) => e.token.type === "codepoint" && e.token.codepoint === 97,
    )!;
    const aFiltered = filtered.find(
      (e) => e.token.type === "codepoint" && e.token.codepoint === 97,
    )!;

    expect(aFiltered.start).toBe(aAll.start);
    expect(aFiltered.end).toBe(aAll.end);
  });
});

// ---------------------------------------------------------------------------
// Call minimisation
// ---------------------------------------------------------------------------

describe("call minimisation", () => {
  it("makes only 1 call when multi-byte groups are out of range", async () => {
    const table: Record<string, number[]> = {
      "": makeDist({ 97: 0.5, 0xc3: 0.5 }),
      c3: makeDist({ 0xa9: 1.0 }),
    };
    const { model, calls } = makeTrackingModel(table);
    const lm = fromByteLevelModel(model, []);

    // range [0, 0.49] — multi-byte group starts at 0.5, excluded
    await collect(lm([], 0, 0.49, 0));
    expect(calls).toEqual([""]);
  });

  it("skips groups below minProb", async () => {
    const table: Record<string, number[]> = {
      "": makeDist({ 97: 0.5, 0xc3: 0.25, 0xe4: 0.25 }),
      c3: makeDist({ 0xa9: 1.0 }),
      e4: makeDist({ 0xb8: 1.0 }),
      e4b8: makeDist({ 0xad: 1.0 }),
    };
    const { model, calls } = makeTrackingModel(table);
    const lm = fromByteLevelModel(model, []);

    // minProb 0.3 means both multi-byte groups (0.25 each) are skipped
    await collect(lm([], 0, 1, 0.3));
    expect(calls).toEqual([""]);
  });

  it("expands only the groups that overlap the range", async () => {
    const table: Record<string, number[]> = {
      "": makeDist({ 97: 0.25, 0xc2: 0.25, 0xc3: 0.25, 0xe4: 0.25 }),
      c2: makeDist({ 0x80: 1.0 }),
      c3: makeDist({ 0xa9: 1.0 }),
      e4: makeDist({ 0xb8: 1.0 }),
      e4b8: makeDist({ 0xad: 1.0 }),
    };
    // a=[0,.25], U+80=[.25,.5], é=[.5,.75], 中=[.75,1]
    const { model, calls } = makeTrackingModel(table);
    const lm = fromByteLevelModel(model, []);

    // range [0.51, 0.74] — only 0xC3 group overlaps
    await collect(lm([], 0.51, 0.74, 0));

    expect(calls).toContain(""); // first-byte
    expect(calls).toContain("c3"); // expanded
    expect(calls).not.toContain("c2"); // skipped
    expect(calls).not.toContain("e4"); // skipped
  });

  it("skips deep continuation sub-groups outside the range", async () => {
    // 3-byte: E4 → two second bytes, each with one third byte
    // 中 = E4 B8 AD at [0, 0.5],  乀 = E4 B9 80 at [0.5, 1.0]
    const table: Record<string, number[]> = {
      "": makeDist({ 0xe4: 1.0 }),
      e4: makeDist({ 0xb8: 0.5, 0xb9: 0.5 }),
      e4b8: makeDist({ 0xad: 1.0 }),
      e4b9: makeDist({ 0x80: 1.0 }),
    };
    const { model, calls } = makeTrackingModel(table);
    const lm = fromByteLevelModel(model, []);

    // range [0, 0.49] covers only the B8 sub-group
    await collect(lm([], 0, 0.49, 0));

    expect(calls).toContain(""); // first-byte
    expect(calls).toContain("e4"); // second-byte dist
    expect(calls).toContain("e4b8"); // in range → expanded
    expect(calls).not.toContain("e4b9"); // out of range → skipped
  });

  it("skips deep continuation sub-groups below minProb", async () => {
    const table: Record<string, number[]> = {
      "": makeDist({ 0xe4: 1.0 }),
      e4: makeDist({ 0xb8: 0.75, 0xb9: 0.25 }),
      e4b8: makeDist({ 0xad: 1.0 }),
      e4b9: makeDist({ 0x80: 1.0 }),
    };
    const { model, calls } = makeTrackingModel(table);
    const lm = fromByteLevelModel(model, []);

    // minProb 0.3 — B9 sub-group (prob 0.25) is below threshold
    await collect(lm([], 0, 1, 0.3));

    expect(calls).toContain("e4b8"); // 0.75 ≥ 0.3 → expanded
    expect(calls).not.toContain("e4b9"); // 0.25 < 0.3 → skipped
  });

  it("skips 4-byte third-level continuations outside the range", async () => {
    // 😀 = F0 9F 98 80,  😁 = F0 9F 98 81
    // Both live inside F0 → 9F → 98, so they share three levels.
    // Add a second sub-group under F0 9F (byte 99) to test pruning.
    const table: Record<string, number[]> = {
      "": makeDist({ 0xf0: 1.0 }),
      f0: makeDist({ 0x9f: 1.0 }),
      f09f: makeDist({ 0x98: 0.5, 0x99: 0.5 }),
      f09f98: makeDist({ 0x80: 1.0 }), // 😀 at [0, 0.5]
      f09f99: makeDist({ 0x80: 1.0 }), // 🦀 at [0.5, 1] (F0 9F 99 80 = U+1F640)
    };
    const { model, calls } = makeTrackingModel(table);
    const lm = fromByteLevelModel(model, []);

    // range [0, 0.49] — only the 0x98 sub-group
    await collect(lm([], 0, 0.49, 0));

    expect(calls).toContain("f09f98");
    expect(calls).not.toContain("f09f99");
  });
});

// ---------------------------------------------------------------------------
// Parallelism
// ---------------------------------------------------------------------------

describe("parallelism", () => {
  it("queries continuation bytes for different lead bytes concurrently", async () => {
    let activeCalls = 0;
    let maxConcurrency = 0;

    const table: Record<string, number[]> = {
      "": makeDist({ 0xc2: 0.5, 0xc3: 0.5 }),
      c2: makeDist({ 0x80: 1.0 }),
      c3: makeDist({ 0x80: 1.0 }),
    };

    const model: ByteLevelModel = async (prefix: readonly number[]) => {
      const key = prefixKey(prefix);
      activeCalls++;
      maxConcurrency = Math.max(maxConcurrency, activeCalls);
      // Yield so concurrent calls can register.
      await new Promise((resolve) => setTimeout(resolve, 0));
      activeCalls--;
      return table[key]!;
    };

    const lm = fromByteLevelModel(model, []);
    await collect(lm([], 0, 1, 0));

    // The two continuation calls (c2, c3) should overlap.
    expect(maxConcurrency).toBeGreaterThanOrEqual(2);
  });
});

// ---------------------------------------------------------------------------
// Codepoint prefix encoding
// ---------------------------------------------------------------------------

describe("codepoint prefix encoding", () => {
  it("encodes an ASCII prefix", async () => {
    const { model, calls } = makeTrackingModel({
      "61": makeDist({ 98: 1.0 }), // after 'a', predict 'b'
    });
    const lm = fromByteLevelModel(model, []);

    const result = await collect(lm([cp(0x61)], 0, 1, 0));
    expect(calls).toEqual(["61"]);
    expect(result[0].token).toEqual(cp(98));
  });

  it("encodes a multi-byte codepoint prefix", async () => {
    // 'é' = U+00E9 → UTF-8 C3 A9
    const { model, calls } = makeTrackingModel({
      c3a9: makeDist({ 97: 1.0 }),
    });
    const lm = fromByteLevelModel(model, []);

    const result = await collect(lm([cp(0xe9)], 0, 1, 0));
    expect(calls).toEqual(["c3a9"]);
    expect(result[0].token).toEqual(cp(97));
  });

  it("encodes a mixed single/multi-byte prefix", async () => {
    // 'aé' → UTF-8 61 C3 A9
    const { model, calls } = makeTrackingModel({
      "61c3a9": makeDist({ 98: 1.0 }),
    });
    const lm = fromByteLevelModel(model, []);

    const result = await collect(lm([cp(97), cp(0xe9)], 0, 1, 0));
    expect(calls).toEqual(["61c3a9"]);
    expect(result[0].token).toEqual(cp(98));
  });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe("edge cases", () => {
  it("returns empty for an all-zero distribution", async () => {
    const lm = fromByteLevelModel(
      makeMockModel({ "": new Array(256).fill(0) }),
      [],
    );
    const result = await collect(lm([], 0, 1, 0));
    expect(result).toHaveLength(0);
  });

  it("handles a single ASCII token", async () => {
    const lm = fromByteLevelModel(
      makeMockModel({ "": makeDist({ 65: 1 }) }),
      [],
    );
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({ token: cp(65), start: 0, end: 1 });
  });

  it("handles multiple lead bytes for different sequence lengths", async () => {
    // Mix of 2-byte, 3-byte, 4-byte in one distribution
    const table: Record<string, number[]> = {
      "": makeDist({ 0xc3: 0.25, 0xe4: 0.5, 0xf0: 0.25 }),
      c3: makeDist({ 0xa9: 1.0 }),
      e4: makeDist({ 0xb8: 1.0 }),
      e4b8: makeDist({ 0xad: 1.0 }),
      f0: makeDist({ 0x9f: 1.0 }),
      f09f: makeDist({ 0x98: 1.0 }),
      f09f98: makeDist({ 0x80: 1.0 }),
    };
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0, 1, 0));

    expect(result.map((e) => e.token)).toEqual([
      cp(0xe9),
      cp(0x4e2d),
      cp(0x1f600),
    ]);
    expect(result[0].start).toBe(0);
    expect(result[0].end).toBe(0.25);
    expect(result[1].start).toBe(0.25);
    expect(result[1].end).toBe(0.75);
    expect(result[2].start).toBe(0.75);
    expect(result[2].end).toBe(1.0);
  });

  it("handles multiple continuation options in a 3-byte sequence", async () => {
    // E4 → two second bytes, each with one third byte
    const table: Record<string, number[]> = {
      "": makeDist({ 0xe4: 1.0 }),
      e4: makeDist({ 0xb8: 0.5, 0xb9: 0.5 }),
      e4b8: makeDist({ 0xad: 1.0 }),
      e4b9: makeDist({ 0x80: 1.0 }),
    };
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0, 1, 0));

    // E4 B8 AD = U+4E2D ('中'), E4 B9 80 = U+4E40 ('乀')
    expect(result).toHaveLength(2);
    expect(result[0].token).toEqual(cp(0x4e2d));
    expect(result[0].end).toBe(0.5);
    expect(result[1].token).toEqual(cp(0x4e40));
    expect(result[1].start).toBe(0.5);
    expect(result[1].end).toBe(1.0);
  });

  it("handles empty codepoint prefix", async () => {
    const { model, calls } = makeTrackingModel({
      "": makeDist({ 97: 1.0 }),
    });
    const lm = fromByteLevelModel(model, []);

    await collect(lm([], 0, 1, 0));
    expect(calls).toEqual([""]);
  });
});

// ---------------------------------------------------------------------------
// Special tokens
// ---------------------------------------------------------------------------

describe("special tokens", () => {
  const eos: SpecialToken = { type: "special", index: 256, label: "<eos>" };
  const pad: SpecialToken = { type: "special", index: 257, label: "<pad>" };

  it("yields special tokens from the first-byte distribution", async () => {
    const table: Record<string, number[]> = {
      "": makeDistWithSpecial({ 97: 0.5 }, [0.3, 0.2]),
    };
    const lm = fromByteLevelModel(makeMockModel(table), [eos, pad]);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(3);
    expect(result[0]).toEqual({ token: cp(97), start: 0, end: 0.5 });
    // Special tokens follow all bytes in cumulative order
    expect(result[1]).toEqual({ token: eos, start: 0.5, end: 0.8 });
    expect(result[2]).toEqual({ token: pad, start: 0.8, end: 1.0 });
  });

  it("positions special tokens after all byte entries in the CDF", async () => {
    const table: Record<string, number[]> = {
      "": makeDistWithSpecial({ 97: 0.25, 0xc3: 0.25 }, [0.5]),
      c3: makeDist({ 0xa9: 1.0 }),
    };
    const lm = fromByteLevelModel(makeMockModel(table), [eos]);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(3);
    // a at [0, 0.25], é at [0.25, 0.5], <eos> at [0.5, 1.0]
    // (yield order between multi-byte and special tokens is not guaranteed)
    expect(result).toContainEqual({ token: cp(97), start: 0, end: 0.25 });
    expect(result).toContainEqual({
      token: cp(0xe9),
      start: 0.25,
      end: 0.5,
    });
    expect(result).toContainEqual({ token: eos, start: 0.5, end: 1.0 });
  });

  it("ignores special tokens in continuation byte distributions", async () => {
    // The continuation dist has an entry at index 256, which should be ignored
    const table: Record<string, number[]> = {
      "": makeDist({ 0xc3: 1.0 }),
      c3: makeDistWithSpecial({ 0xa9: 0.8 }, [0.2]),
    };
    const lm = fromByteLevelModel(makeMockModel(table), [eos]);
    const result = await collect(lm([], 0, 1, 0));

    // Only the codepoint is yielded; the special token in the continuation is ignored
    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(cp(0xe9));
    // The extent is [0, 0.8] — the 0.2 for the special token is a hole
    expect(result[0].start).toBe(0);
    expect(result[0].end).toBe(0.8);
  });

  it("handles zero-probability special tokens", async () => {
    const table: Record<string, number[]> = {
      "": makeDistWithSpecial({ 97: 1.0 }, [0, 0]),
    };
    const lm = fromByteLevelModel(makeMockModel(table), [eos, pad]);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(cp(97));
  });

  it("applies range filtering to special tokens", async () => {
    const table: Record<string, number[]> = {
      "": makeDistWithSpecial({ 97: 0.5 }, [0.3, 0.2]),
    };
    const lm = fromByteLevelModel(makeMockModel(table), [eos, pad]);
    // range [0.51, 0.79] only overlaps <eos>=[0.5, 0.8]
    const result = await collect(lm([], 0.51, 0.79, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(eos);
  });

  it("applies minProb filtering to special tokens", async () => {
    const table: Record<string, number[]> = {
      "": makeDistWithSpecial({ 97: 0.5 }, [0.3, 0.2]),
    };
    const lm = fromByteLevelModel(makeMockModel(table), [eos, pad]);
    // minProb 0.25 excludes <pad> (0.2) but keeps <eos> (0.3) and a (0.5)
    const result = await collect(lm([], 0, 1, 0.25));

    expect(result).toHaveLength(2);
    expect(result[0].token).toEqual(cp(97));
    expect(result[1].token).toEqual(eos);
  });

  it("looks up a specific special token", async () => {
    const table: Record<string, number[]> = {
      "": makeDistWithSpecial({ 97: 0.5 }, [0.3, 0.2]),
    };
    const lm = fromByteLevelModel(makeMockModel(table), [eos, pad]);
    const result = await collect(lm([], 0, 1, 0, eos));

    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({ token: eos, start: 0.5, end: 0.8 });
  });

  it("looks up a specific special token that has zero probability", async () => {
    const table: Record<string, number[]> = {
      "": makeDistWithSpecial({ 97: 1.0 }, [0]),
    };
    const lm = fromByteLevelModel(makeMockModel(table), [eos]);
    const result = await collect(lm([], 0, 1, 0, eos));

    expect(result).toHaveLength(0);
  });

  it("returns no special tokens when specialTokens list is empty", async () => {
    // Distribution has entries beyond 255 but no specialTokens registered
    const table: Record<string, number[]> = {
      "": makeDistWithSpecial({ 97: 0.5 }, [0.5]),
    };
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(1);
    expect(result[0].token).toEqual(cp(97));
  });

  it("only-special-tokens distribution works", async () => {
    const table: Record<string, number[]> = {
      "": makeDistWithSpecial({}, [0.6, 0.4]),
    };
    const lm = fromByteLevelModel(makeMockModel(table), [eos, pad]);
    const result = await collect(lm([], 0, 1, 0));

    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({ token: eos, start: 0, end: 0.6 });
    expect(result[1]).toEqual({ token: pad, start: 0.6, end: 1.0 });
  });
});

// ---------------------------------------------------------------------------
// specificToken lookup (codepoint)
// ---------------------------------------------------------------------------

describe("specificToken lookup", () => {
  it("looks up a single-byte codepoint", async () => {
    const table: Record<string, number[]> = {
      "": makeDist({ 97: 0.5, 98: 0.25, 99: 0.25 }),
    };
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0, 1, 0, cp(98)));

    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({ token: cp(98), start: 0.5, end: 0.75 });
  });

  it("looks up a multi-byte codepoint", async () => {
    const table: Record<string, number[]> = {
      "": makeDist({ 97: 0.5, 0xc3: 0.5 }),
      c3: makeDist({ 0xa8: 0.5, 0xa9: 0.5 }),
    };
    const lm = fromByteLevelModel(makeMockModel(table), []);
    const result = await collect(lm([], 0, 1, 0, cp(0xe9)));

    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({ token: cp(0xe9), start: 0.75, end: 1.0 });
  });
});

// ---------------------------------------------------------------------------
// interpolate
// ---------------------------------------------------------------------------

describe("interpolate", () => {
  /** Create a LanguageModel that always returns the given distribution. */
  function simpleModel(dist: number[]): LanguageModel<string> {
    return asNormalized(async () => dist);
  }

  it("computes correct mixture probabilities", async () => {
    const a = simpleModel([0, 0.75, 0.25]);
    const b = simpleModel([0, 0.25, 0.75]);

    const mixed = interpolate([
      { model: a, weight: 1 },
      { model: b, weight: 1 },
    ]);
    const result = await mixed("");

    expect(result).toEqual([0, 0.5, 0.5]);
  });

  it("weight=0 on B reproduces model A exactly", async () => {
    const a = simpleModel([0, 0.6, 0.4]);
    const b = simpleModel([0, 0.1, 0.9]);

    const mixed = interpolate([
      { model: a, weight: 1 },
      { model: b, weight: 0 },
    ]);
    const result = await mixed("");

    expect(result).toEqual([0, 0.6, 0.4]);
  });

  it("weight=0 on A reproduces model B exactly", async () => {
    const a = simpleModel([0, 0.6, 0.4]);
    const b = simpleModel([0, 0.3, 0.7]);

    const mixed = interpolate([
      { model: a, weight: 0 },
      { model: b, weight: 1 },
    ]);
    const result = await mixed("");

    expect(result).toEqual([0, 0.3, 0.7]);
  });

  it("weight=0 model is not queried", async () => {
    let called = false;
    const a = simpleModel([0, 0.6, 0.4]);
    const b: LanguageModel<string> = asNormalized(async () => {
      called = true;
      return [0, 0.1, 0.9];
    });

    const mixed = interpolate([
      { model: a, weight: 1 },
      { model: b, weight: 0 },
    ]);
    await mixed("");

    expect(called).toBe(false);
  });

  it("produces probabilities summing to 1", async () => {
    const a = simpleModel([0, 0.3, 0.5, 0.2]);
    const b = simpleModel([0, 0.4, 0.4, 0.2]);

    const mixed = interpolate([
      { model: a, weight: 0.6 },
      { model: b, weight: 0.4 },
    ]);
    const result = await mixed("");
    const sum = result.reduce((s, p) => s + p, 0);

    expect(sum).toBeCloseTo(1);
  });

  it("handles different vocabulary sizes", async () => {
    const a = simpleModel([0.5, 0.5]);
    const b = simpleModel([0.25, 0.25, 0.25, 0.25]);

    const mixed = interpolate([
      { model: a, weight: 1 },
      { model: b, weight: 1 },
    ]);
    const result = await mixed("");

    // a contributes [0.25, 0.25, 0, 0], b contributes [0.125, 0.125, 0.125, 0.125]
    expect(result).toHaveLength(4);
    expect(result[0]).toBeCloseTo(0.375);
    expect(result[1]).toBeCloseTo(0.375);
    expect(result[2]).toBeCloseTo(0.125);
    expect(result[3]).toBeCloseTo(0.125);
  });

  it("queries all models in parallel", async () => {
    let activeCalls = 0;
    let maxConcurrency = 0;

    const makeSlowModel = (dist: number[]): LanguageModel<string> =>
      asNormalized(async () => {
        activeCalls++;
        maxConcurrency = Math.max(maxConcurrency, activeCalls);
        await new Promise((resolve) => setTimeout(resolve, 10));
        activeCalls--;
        return dist;
      });

    const a = makeSlowModel([0, 0.5, 0.5]);
    const b = makeSlowModel([0, 0.5, 0.5]);

    const mixed = interpolate([
      { model: a, weight: 1 },
      { model: b, weight: 1 },
    ]);
    await mixed("");

    expect(maxConcurrency).toBe(2);
  });

  it("throws on empty components", () => {
    expect(() => interpolate([])).toThrow("at least one model");
  });

  it("throws on negative weight", () => {
    expect(() =>
      interpolate([{ model: simpleModel([1]), weight: -1 }]),
    ).toThrow("Negative weight");
  });

  it("throws when all weights are zero", () => {
    expect(() =>
      interpolate([
        { model: simpleModel([1]), weight: 0 },
        { model: simpleModel([1]), weight: 0 },
      ]),
    ).toThrow("Total weight must be positive");
  });
});

// ---------------------------------------------------------------------------
// forceCleanUtf8
// ---------------------------------------------------------------------------

describe("forceCleanUtf8", () => {
  /**
   * Create a mock LanguageModel from a hex-key → sparse entries table.
   * Each value is padded to a full 256-entry ordered distribution.
   */
  function makePlainModel(
    table: Record<string, Record<number, number>>,
  ): LanguageModel<readonly number[]> {
    return asNormalized(async (prefix: readonly number[]) => {
      const key = prefixKey(prefix);
      const sparse = table[key];
      if (!sparse) throw new Error(`Unexpected prefix: "${key}"`);
      return makeDist(sparse);
    });
  }

  it("zeroes out continuation bytes at a character boundary (unnormalized)", async () => {
    const model = makePlainModel({
      "": {
        0x61: 0.5, // 'a' — legal lead
        0x80: 0.25, // continuation — illegal at boundary
        0xc3: 0.25, // 2-byte lead — legal
      },
    });
    const dist = await forceCleanUtf8(model)([]);

    expect(dist).toHaveLength(256);
    expect(dist[0x61]).toBeCloseTo(0.5);
    expect(dist[0x80]).toBe(0);
    expect(dist[0xc3]).toBeCloseTo(0.25);
  });

  it("zeroes out lead bytes when a continuation byte is expected (unnormalized)", async () => {
    // Prefix [0xC3] → mid-2-byte-sequence, need one continuation
    const model = makePlainModel({
      c3: {
        0xa9: 0.5, // valid continuation
        0x61: 0.25, // ASCII — illegal here
        0xc3: 0.25, // lead — illegal here
      },
    });
    const dist = await forceCleanUtf8(model)([0xc3]);

    expect(dist).toHaveLength(256);
    expect(dist[0xa9]).toBeCloseTo(0.5);
    expect(dist[0x61]).toBe(0);
    expect(dist[0xc3]).toBe(0);
  });

  it("enforces restricted range after E0 (rejects overlong encodings)", async () => {
    const model = makePlainModel({
      e0: {
        0x80: 0.5, // < 0xA0 → overlong
        0xa0: 0.5, // legal
      },
    });
    const dist = await forceCleanUtf8(model)([0xe0]);

    expect(dist).toHaveLength(256);
    expect(dist[0x80]).toBe(0);
    expect(dist[0xa0]).toBeCloseTo(0.5);
  });

  it("enforces restricted range after F0 and F4", async () => {
    // After F0: first continuation must be 0x90–0xBF
    const f0Model = makePlainModel({
      f0: {
        0x80: 0.5, // too low
        0x90: 0.5, // legal
      },
    });
    const f0Dist = await forceCleanUtf8(f0Model)([0xf0]);
    expect(f0Dist).toHaveLength(256);
    expect(f0Dist[0x80]).toBe(0);
    expect(f0Dist[0x90]).toBeCloseTo(0.5);

    // After F4: first continuation must be 0x80–0x8F
    const f4Model = makePlainModel({
      f4: {
        0x80: 0.5, // legal
        0x90: 0.5, // beyond U+10FFFF
      },
    });
    const f4Dist = await forceCleanUtf8(f4Model)([0xf4]);
    expect(f4Dist).toHaveLength(256);
    expect(f4Dist[0x80]).toBeCloseTo(0.5);
    expect(f4Dist[0x90]).toBe(0);
  });

  it("passes through an already-clean distribution unchanged", async () => {
    const model = makePlainModel({
      "": {
        0x61: 0.5,
        0x62: 0.5,
      },
    });
    const dist = await forceCleanUtf8(model)([]);

    expect(dist).toHaveLength(256);
    expect(dist[0x61]).toBe(0.5);
    expect(dist[0x62]).toBe(0.5);
  });

  it("allows generic continuation bytes in later positions of a multi-byte sequence", async () => {
    // Prefix [0xE4, 0xB8] — 3-byte sequence, need one more continuation
    const model = makePlainModel({
      e4b8: {
        0x80: 0.25,
        0xad: 0.25,
        0xbf: 0.25,
        0x61: 0.25, // illegal: not a continuation
      },
    });
    const dist = await forceCleanUtf8(model)([0xe4, 0xb8]);

    expect(dist).toHaveLength(256);
    expect(dist[0x80]).toBeCloseTo(0.25);
    expect(dist[0xad]).toBeCloseTo(0.25);
    expect(dist[0xbf]).toBeCloseTo(0.25);
    expect(dist[0x61]).toBe(0);
  });
});

describe("byteOnly", () => {
  it("passes through byte-only prefixes unchanged", async () => {
    const calls: number[][] = [];
    const inner: LanguageModel<Uint8Array> = asNormalized(async (prefix) => {
      calls.push(Array.from(prefix));
      return [0.5, 0.5];
    });
    const adapted = byteOnly(inner);
    await adapted([0x41, 0x42]);
    expect(calls).toEqual([[0x41, 0x42]]);
  });

  it("strips values > 255 from the prefix", async () => {
    const calls: number[][] = [];
    const inner: LanguageModel<Uint8Array> = asNormalized(async (prefix) => {
      calls.push(Array.from(prefix));
      return [1.0];
    });
    const adapted = byteOnly(inner);
    await adapted([0x41, 256, 0x42, 1000]);
    expect(calls).toEqual([[0x41, 0x42]]);
  });

  it("forwards the distribution from the inner model", async () => {
    const inner: LanguageModel<Uint8Array> = asNormalized(async () => [
      0.3, 0.7,
    ]);
    const dist = await byteOnly(inner)([300, 0x61]);
    expect(dist).toEqual([0.3, 0.7]);
  });
});
