import { describe, it, expect } from "vitest";
import type { LanguageModel, Cursor } from "./types";
import { adaptModel } from "./types";
import { normalizeCursor, cursorToGlobal } from "./cursor";

// ---------------------------------------------------------------------------
// Test language models
// ---------------------------------------------------------------------------

/** Uniform binary: A and B each with probability 0.5. */
const binary = adaptModel<readonly string[], string>(async () => [
  { token: "A", probability: 0.5 },
  { token: "B", probability: 0.5 },
]);

/** Asymmetric binary: A = 0.8, B = 0.2. */
const asym = adaptModel<readonly string[], string>(async () => [
  { token: "A", probability: 0.8 },
  { token: "B", probability: 0.2 },
]);

/** Three tokens. */
const ternary = adaptModel<readonly string[], string>(async () => [
  { token: "X", probability: 0.2 },
  { token: "Y", probability: 0.5 },
  { token: "Z", probability: 0.3 },
]);

/** Context-sensitive: distribution depends on the last token. */
const contextual = adaptModel<readonly string[], string>(async (prefix) => {
  if (prefix.length === 0)
    return [
      { token: "A", probability: 0.6 },
      { token: "B", probability: 0.4 },
    ];
  if (prefix[prefix.length - 1] === "A")
    return [
      { token: "A", probability: 0.3 },
      { token: "B", probability: 0.7 },
    ];
  return [
    { token: "A", probability: 0.5 },
    { token: "B", probability: 0.5 },
  ];
});

/** Deterministic: single token with probability 1. */
const deterministic = adaptModel<readonly string[], string>(async () => [
  { token: "A", probability: 1.0 },
]);

/** Empty distribution — no continuations. */
const empty = adaptModel<readonly string[], string>(async () => []);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Assert that two cursor states point at the same global location. */
async function expectSameGlobal<T>(
  model: LanguageModel<readonly T[], T>,
  a: Cursor<T>,
  b: Cursor<T>,
) {
  const ga = await cursorToGlobal(model, a);
  const gb = await cursorToGlobal(model, b);
  expect(gb.x).toBeCloseTo(ga.x, 6);
  expect(gb.y).toBeCloseTo(ga.y, 6);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("normalizeCursor", () => {
  // -- Identity property: (0, 0) is always stable --

  describe("(0, 0) identity", () => {
    it("empty prefix stays empty", async () => {
      const r = await normalizeCursor(binary, { prefix: [], x: 0, y: 0 });
      expect(r.prefix).toEqual([]);
      expect(r.x).toBe(0);
      expect(r.y).toBe(0);
    });

    it("non-empty prefix is preserved", async () => {
      const r = await normalizeCursor(binary, { prefix: ["A"], x: 0, y: 0 });
      expect(r.prefix).toEqual(["A"]);
      expect(r.x).toBe(0);
      expect(r.y).toBe(0);
    });

    it("deep prefix is preserved", async () => {
      const r = await normalizeCursor(binary, {
        prefix: ["A", "B", "A"],
        x: 0,
        y: 0,
      });
      expect(r.prefix).toEqual(["A", "B", "A"]);
      expect(r.x).toBe(0);
      expect(r.y).toBe(0);
    });
  });

  // -- Descent --

  describe("descent", () => {
    it("descends one level into the first child (binary)", async () => {
      // Binary: A occupies x∈[0.5,1], y∈[0,0.5)
      // Cursor at (0.7, 0.3): inside A.
      const r = await normalizeCursor(binary, { prefix: [], x: 0.7, y: 0.3 });
      expect(r.prefix).toEqual(["A"]);
      expect(r.x).toBeCloseTo(0.4); // (0.7−0.5)/0.5
      expect(r.y).toBeCloseTo(0.6); // 0.3/0.5
    });

    it("descends one level into the second child (binary)", async () => {
      // B occupies x∈[0.5,1], y∈[0.5,1)
      const r = await normalizeCursor(binary, { prefix: [], x: 0.6, y: 0.7 });
      expect(r.prefix).toEqual(["B"]);
      expect(r.x).toBeCloseTo(0.2); // (0.6−0.5)/0.5
      expect(r.y).toBeCloseTo(0.4); // (0.7−0.5)/0.5
    });

    it("descends multiple levels", async () => {
      // x=0.9, y=0.1.  Each step enters A (p=0.5).
      //   → (0.8, 0.2) → (0.6, 0.4) → (0.2, 0.8)  x<0.5 → stop
      const r = await normalizeCursor(binary, { prefix: [], x: 0.9, y: 0.1 });
      expect(r.prefix).toEqual(["A", "A", "A"]);
      expect(r.x).toBeCloseTo(0.2);
      expect(r.y).toBeCloseTo(0.8);
    });

    it("descends into the correct child of a ternary model", async () => {
      // Ternary: X=0.2 y∈[0,0.2), Y=0.5 y∈[0.2,0.7), Z=0.3 y∈[0.7,1)
      // Cursor at (0.6, 0.4) — should land in Y (x≥0.5, y∈[0.2,0.7)).
      const r = await normalizeCursor(ternary, { prefix: [], x: 0.6, y: 0.4 });
      expect(r.prefix).toEqual(["Y"]);
      expect(r.x).toBeCloseTo((0.6 - 0.5) / 0.5); // 0.2
      expect(r.y).toBeCloseTo((0.4 - 0.2) / 0.5); // 0.4
    });

    it("descends with a context-sensitive model", async () => {
      // Root: A=0.6 y∈[0,0.6), B=0.4 y∈[0.6,1).
      // (0.7, 0.2): x≥0.4 and y<0.6 → enter A.
      //   new: x=(0.7−0.4)/0.6=0.5, y=0.2/0.6=1/3
      // After A: A=0.3 y∈[0,0.3), B=0.7 y∈[0.3,1).
      //   x=0.5≥0.3, y=1/3≥0.3 → enter B.
      //   new: x=(0.5−0.3)/0.7=2/7, y=(1/3−0.3)/0.7=1/21
      const r = await normalizeCursor(contextual, {
        prefix: [],
        x: 0.7,
        y: 0.2,
      });
      expect(r.prefix).toEqual(["A", "B"]);
      expect(r.x).toBeCloseTo(2 / 7);
      expect(r.y).toBeCloseTo(1 / 21);
    });

    it("descends with asymmetric probabilities", async () => {
      // A=0.8 x≥0.2 y∈[0,0.8),  B=0.2 x≥0.8 y∈[0.8,1)
      // (0.5, 0.5): enter A →  x=(0.5−0.2)/0.8=0.375,  y=0.5/0.8=0.625
      // enter A again →         x=(0.375−0.2)/0.8=0.21875, y=0.625/0.8=0.78125
      // enter A again →         x=(0.21875−0.2)/0.8≈0.0234, y=0.78125/0.8≈0.977
      // x<0.2 → stop.
      const r = await normalizeCursor(asym, { prefix: [], x: 0.5, y: 0.5 });
      expect(r.prefix).toEqual(["A", "A", "A"]);
      expect(r.x).toBeCloseTo(0.0234375);
      expect(r.y).toBeCloseTo(0.9765625);
    });
  });

  // -- Gap (no descent) --

  describe("gap", () => {
    it("stays at root when x is in the gap", async () => {
      // Binary: children start at x=0.5.  Cursor at x=0.3 — gap.
      const r = await normalizeCursor(binary, { prefix: [], x: 0.3, y: 0.3 });
      expect(r.prefix).toEqual([]);
      expect(r.x).toBe(0.3);
      expect(r.y).toBe(0.3);
    });

    it("stays in a non-root square when in its gap", async () => {
      const r = await normalizeCursor(binary, {
        prefix: ["A"],
        x: 0.3,
        y: 0.3,
      });
      expect(r.prefix).toEqual(["A"]);
      expect(r.x).toBe(0.3);
      expect(r.y).toBe(0.3);
    });
  });

  // -- Ascent --

  describe("ascent", () => {
    it("ascends and enters the next sibling on y overflow", async () => {
      // prefix=['A'], y=1.1 → leave A, enter B.
      // Ascend: x=0.5+0.3·0.5=0.65, y=0+1.1·0.5=0.55
      // B: x≥0.5, y∈[0.5,1) → enter B.
      //   x=(0.65−0.5)/0.5=0.3, y=(0.55−0.5)/0.5=0.1
      const r = await normalizeCursor(binary, {
        prefix: ["A"],
        x: 0.3,
        y: 1.1,
      });
      expect(r.prefix).toEqual(["B"]);
      expect(r.x).toBeCloseTo(0.3);
      expect(r.y).toBeCloseTo(0.1);
    });

    it("ascends and enters the previous sibling on y underflow", async () => {
      // prefix=['B'], y=−0.1 → leave B, enter A.
      // Ascend: x=0.5+0.3·0.5=0.65, y=0.5+(−0.1)·0.5=0.45
      // A: x≥0.5, y∈[0,0.5) → enter A.
      //   x=(0.65−0.5)/0.5=0.3, y=0.45/0.5=0.9
      const r = await normalizeCursor(binary, {
        prefix: ["B"],
        x: 0.3,
        y: -0.1,
      });
      expect(r.prefix).toEqual(["A"]);
      expect(r.x).toBeCloseTo(0.3);
      expect(r.y).toBeCloseTo(0.9);
    });

    it("ascends to the gap on x underflow", async () => {
      // prefix=['A'], x=−0.1 → leave A.
      // Ascend: x=0.5+(−0.1)·0.5=0.45, y=0+0.3·0.5=0.15
      // x=0.45 < 0.5 — in the root's gap.
      const r = await normalizeCursor(binary, {
        prefix: ["A"],
        x: -0.1,
        y: 0.3,
      });
      expect(r.prefix).toEqual([]);
      expect(r.x).toBeCloseTo(0.45);
      expect(r.y).toBeCloseTo(0.15);
    });

    it("ascends multiple levels if needed", async () => {
      // prefix=['A','A'], y=−0.2.  First ascend to ['A'], then
      // y is still < 0, so ascend again to [].
      // Step 1: A (in ['A'] dist) has cum=0, prob=0.5
      //   x = 0.5 + 0.3·0.5 = 0.65,  y = 0 + (−0.2)·0.5 = −0.1
      // Step 2: still y<0. A (in root dist) has cum=0, prob=0.5
      //   x = 0.5 + 0.65·0.5 = 0.825,  y = 0 + (−0.1)·0.5 = −0.05
      // Now at root, y<0 → clamp to 0.
      // Then descend: x=0.825 ≥ 0.5, y=0 ∈ [0, 0.5) → A.
      //   x=(0.825−0.5)/0.5=0.65, y=0/0.5=0
      // In A: x=0.65 ≥ 0.5, y=0 ∈ [0, 0.5) → A.
      //   x=(0.65−0.5)/0.5=0.3, y=0
      // In AA: x=0.3 < 0.5 → stop.
      const r = await normalizeCursor(binary, {
        prefix: ["A", "A"],
        x: 0.3,
        y: -0.2,
      });
      expect(r.prefix).toEqual(["A", "A"]);
      expect(r.x).toBeCloseTo(0.3);
      expect(r.y).toBeCloseTo(0);
    });
  });

  // -- Global position preservation --

  describe("global position preservation", () => {
    const cases: Array<{
      name: string;
      model: LanguageModel<readonly string[], string>;
      state: Cursor<string>;
    }> = [
      {
        name: "simple descent",
        model: binary,
        state: { prefix: [], x: 0.7, y: 0.3 },
      },
      {
        name: "deep descent",
        model: binary,
        state: { prefix: [], x: 0.9, y: 0.1 },
      },
      {
        name: "ascend + re-descend (y overflow)",
        model: binary,
        state: { prefix: ["A"], x: 0.3, y: 1.1 },
      },
      {
        name: "ascend + re-descend (y underflow)",
        model: binary,
        state: { prefix: ["B"], x: 0.3, y: -0.1 },
      },
      {
        name: "ascend to gap",
        model: binary,
        state: { prefix: ["A"], x: -0.1, y: 0.3 },
      },
      {
        name: "asymmetric deep descent",
        model: asym,
        state: { prefix: [], x: 0.5, y: 0.5 },
      },
      {
        name: "ternary descent",
        model: ternary,
        state: { prefix: [], x: 0.6, y: 0.4 },
      },
      {
        name: "context-sensitive descent",
        model: contextual,
        state: { prefix: [], x: 0.7, y: 0.2 },
      },
      {
        name: "already normalised",
        model: binary,
        state: { prefix: ["A"], x: 0.3, y: 0.3 },
      },
    ];

    for (const { name, model, state } of cases) {
      it(`preserves position: ${name}`, async () => {
        const result = await normalizeCursor(model, state);
        await expectSameGlobal(model, state, result);
      });
    }
  });

  // -- Edge cases --

  describe("edge cases", () => {
    it("empty distribution — no descent possible", async () => {
      const r = await normalizeCursor(empty, { prefix: [], x: 0.5, y: 0.5 });
      expect(r.prefix).toEqual([]);
      expect(r.x).toBe(0.5);
      expect(r.y).toBe(0.5);
    });

    it("deterministic model stops at maxDepth", async () => {
      const r = await normalizeCursor(
        deterministic,
        { prefix: [], x: 0.5, y: 0.5 },
        { maxDepth: 5 },
      );
      expect(r.prefix).toHaveLength(5);
      expect(r.prefix).toEqual(["A", "A", "A", "A", "A"]);
      // x and y are unchanged because p=1: (x−0)/1 = x each step.
      expect(r.x).toBeCloseTo(0.5);
      expect(r.y).toBeCloseTo(0.5);
    });

    it("clamps at root on extreme out-of-bounds", async () => {
      const r = await normalizeCursor(binary, { prefix: [], x: -5, y: 3 });
      // Clamped to (0, 1−ε), then descend.
      expect(r.x).toBeGreaterThanOrEqual(0);
      expect(r.x).toBeLessThan(1);
      expect(r.y).toBeGreaterThanOrEqual(0);
      expect(r.y).toBeLessThan(1);
    });

    it("handles cursor exactly on the y boundary between tokens", async () => {
      // y = 0.5 exactly: top boundary of B (inclusive), not A.
      const r = await normalizeCursor(binary, { prefix: [], x: 0.7, y: 0.5 });
      expect(r.prefix).toEqual(["B"]);
    });
  });

  // -- Generics: non-string tokens --

  describe("generic token types", () => {
    it("works with numeric tokens", async () => {
      const numModel = adaptModel<readonly number[], number>(async () => [
        { token: 1, probability: 0.4 },
        { token: 2, probability: 0.6 },
      ]);
      // Token 1: x∈[0.6,1] y∈[0,0.4).  Token 2: x∈[0.4,1] y∈[0.4,1).
      // (0.5, 0.5): token 1 x≥0.6? no.  token 2 x≥0.4 and y∈[0.4,1)? yes.
      const r = await normalizeCursor(numModel, { prefix: [], x: 0.5, y: 0.5 });
      expect(r.prefix).toEqual([2]);
      expect(r.x).toBeCloseTo(1 / 6);
      expect(r.y).toBeCloseTo(1 / 6);
    });

    it("ascends correctly with object tokens (reference-equal)", async () => {
      type Tok = { id: number };
      const tok1: Tok = { id: 1 };
      const tok2: Tok = { id: 2 };
      const objModel = adaptModel<readonly Tok[], Tok>(async () => [
        { token: tok1, probability: 0.5 },
        { token: tok2, probability: 0.5 },
      ]);

      // prefix=[tok1], y=1.1 → ascend, enter tok2.
      const r = await normalizeCursor(objModel, {
        prefix: [tok1],
        x: 0.3,
        y: 1.1,
      });
      expect(r.prefix).toHaveLength(1);
      expect(r.prefix[0]).toBe(tok2);
      expect(r.x).toBeCloseTo(0.3);
      expect(r.y).toBeCloseTo(0.1);
    });
  });
});

// ---------------------------------------------------------------------------
// cursorToGlobal
// ---------------------------------------------------------------------------

describe("cursorToGlobal", () => {
  it("root with (0,0) maps to global (0,0)", async () => {
    const g = await cursorToGlobal(binary, { prefix: [], x: 0, y: 0 });
    expect(g.x).toBeCloseTo(0);
    expect(g.y).toBeCloseTo(0);
  });

  it("root with (1,1) maps to global (1,1)", async () => {
    const g = await cursorToGlobal(binary, { prefix: [], x: 1, y: 1 });
    expect(g.x).toBeCloseTo(1);
    expect(g.y).toBeCloseTo(1);
  });

  it("prefix=[A] with (0,0) maps to A's top-left corner", async () => {
    // Binary: A has size 0.5, left edge at 0.5, top edge at 0.
    const g = await cursorToGlobal(binary, { prefix: ["A"], x: 0, y: 0 });
    expect(g.x).toBeCloseTo(0.5);
    expect(g.y).toBeCloseTo(0);
  });

  it("prefix=[B] with (0,0) maps to B's top-left corner", async () => {
    // Binary: B has size 0.5, left edge at 0.5, top edge at 0.5.
    const g = await cursorToGlobal(binary, { prefix: ["B"], x: 0, y: 0 });
    expect(g.x).toBeCloseTo(0.5);
    expect(g.y).toBeCloseTo(0.5);
  });

  it("composes correctly for a two-token prefix", async () => {
    // prefix=[A, B] with asymmetric model: A=0.8, B=0.2.
    // A square: size=0.8, top=0.
    // In A's dist, B: cum=0.8, prob=0.2 → AB square: size=0.16, top=0+0.8·0.8=0.64.
    // (0, 0) → global (1−0.16, 0.64) = (0.84, 0.64).
    const g = await cursorToGlobal(asym, { prefix: ["A", "B"], x: 0, y: 0 });
    expect(g.x).toBeCloseTo(0.84);
    expect(g.y).toBeCloseTo(0.64);
  });
});
