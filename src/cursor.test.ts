import { describe, it, expect } from "vitest";
import {
  normalizeCursor,
  cursorToGlobal,
  type LanguageModel,
  type CursorState,
} from "./cursor";

// ---------------------------------------------------------------------------
// Test language models
// ---------------------------------------------------------------------------

/** Uniform binary: A and B each with probability 0.5. */
const binary: LanguageModel<string> = () => [
  { token: "A", probability: 0.5 },
  { token: "B", probability: 0.5 },
];

/** Asymmetric binary: A = 0.8, B = 0.2. */
const asym: LanguageModel<string> = () => [
  { token: "A", probability: 0.8 },
  { token: "B", probability: 0.2 },
];

/** Three tokens. */
const ternary: LanguageModel<string> = () => [
  { token: "X", probability: 0.2 },
  { token: "Y", probability: 0.5 },
  { token: "Z", probability: 0.3 },
];

/** Context-sensitive: distribution depends on the last token. */
const contextual: LanguageModel<string> = (prefix) => {
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
};

/** Deterministic: single token with probability 1. */
const deterministic: LanguageModel<string> = () => [
  { token: "A", probability: 1.0 },
];

/** Empty distribution — no continuations. */
const empty: LanguageModel<string> = () => [];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Assert that two cursor states point at the same global location. */
function expectSameGlobal<T>(
  model: LanguageModel<T>,
  a: CursorState<T>,
  b: CursorState<T>,
  tokenEquals?: (x: T, y: T) => boolean,
) {
  const ga = cursorToGlobal(model, a, tokenEquals);
  const gb = cursorToGlobal(model, b, tokenEquals);
  expect(gb.x).toBeCloseTo(ga.x, 10);
  expect(gb.y).toBeCloseTo(ga.y, 10);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("normalizeCursor", () => {
  // -- Identity property: (0, 0) is always stable --

  describe("(0, 0) identity", () => {
    it("empty prefix stays empty", () => {
      const r = normalizeCursor(binary, { prefix: [], x: 0, y: 0 });
      expect(r.prefix).toEqual([]);
      expect(r.x).toBe(0);
      expect(r.y).toBe(0);
    });

    it("non-empty prefix is preserved", () => {
      const r = normalizeCursor(binary, { prefix: ["A"], x: 0, y: 0 });
      expect(r.prefix).toEqual(["A"]);
      expect(r.x).toBe(0);
      expect(r.y).toBe(0);
    });

    it("deep prefix is preserved", () => {
      const r = normalizeCursor(binary, {
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
    it("descends one level into the first child (binary)", () => {
      // Binary: A occupies x∈[0.5,1], y∈[0,0.5)
      // Cursor at (0.7, 0.3): inside A.
      const r = normalizeCursor(binary, { prefix: [], x: 0.7, y: 0.3 });
      expect(r.prefix).toEqual(["A"]);
      expect(r.x).toBeCloseTo(0.4); // (0.7−0.5)/0.5
      expect(r.y).toBeCloseTo(0.6); // 0.3/0.5
    });

    it("descends one level into the second child (binary)", () => {
      // B occupies x∈[0.5,1], y∈[0.5,1)
      const r = normalizeCursor(binary, { prefix: [], x: 0.6, y: 0.7 });
      expect(r.prefix).toEqual(["B"]);
      expect(r.x).toBeCloseTo(0.2); // (0.6−0.5)/0.5
      expect(r.y).toBeCloseTo(0.4); // (0.7−0.5)/0.5
    });

    it("descends multiple levels", () => {
      // x=0.9, y=0.1.  Each step enters A (p=0.5).
      //   → (0.8, 0.2) → (0.6, 0.4) → (0.2, 0.8)  x<0.5 → stop
      const r = normalizeCursor(binary, { prefix: [], x: 0.9, y: 0.1 });
      expect(r.prefix).toEqual(["A", "A", "A"]);
      expect(r.x).toBeCloseTo(0.2);
      expect(r.y).toBeCloseTo(0.8);
    });

    it("descends into the correct child of a ternary model", () => {
      // Ternary: X=0.2 y∈[0,0.2), Y=0.5 y∈[0.2,0.7), Z=0.3 y∈[0.7,1)
      // Cursor at (0.6, 0.4) — should land in Y (x≥0.5, y∈[0.2,0.7)).
      const r = normalizeCursor(ternary, { prefix: [], x: 0.6, y: 0.4 });
      expect(r.prefix).toEqual(["Y"]);
      expect(r.x).toBeCloseTo((0.6 - 0.5) / 0.5); // 0.2
      expect(r.y).toBeCloseTo((0.4 - 0.2) / 0.5); // 0.4
    });

    it("descends with a context-sensitive model", () => {
      // Root: A=0.6 y∈[0,0.6), B=0.4 y∈[0.6,1).
      // (0.7, 0.2): x≥0.4 and y<0.6 → enter A.
      //   new: x=(0.7−0.4)/0.6=0.5, y=0.2/0.6=1/3
      // After A: A=0.3 y∈[0,0.3), B=0.7 y∈[0.3,1).
      //   x=0.5≥0.3, y=1/3≥0.3 → enter B.
      //   new: x=(0.5−0.3)/0.7=2/7, y=(1/3−0.3)/0.7=1/21
      const r = normalizeCursor(contextual, { prefix: [], x: 0.7, y: 0.2 });
      expect(r.prefix).toEqual(["A", "B"]);
      expect(r.x).toBeCloseTo(2 / 7);
      expect(r.y).toBeCloseTo(1 / 21);
    });

    it("descends with asymmetric probabilities", () => {
      // A=0.8 x≥0.2 y∈[0,0.8),  B=0.2 x≥0.8 y∈[0.8,1)
      // (0.5, 0.5): enter A →  x=(0.5−0.2)/0.8=0.375,  y=0.5/0.8=0.625
      // enter A again →         x=(0.375−0.2)/0.8=0.21875, y=0.625/0.8=0.78125
      // enter A again →         x=(0.21875−0.2)/0.8≈0.0234, y=0.78125/0.8≈0.977
      // x<0.2 → stop.
      const r = normalizeCursor(asym, { prefix: [], x: 0.5, y: 0.5 });
      expect(r.prefix).toEqual(["A", "A", "A"]);
      expect(r.x).toBeCloseTo(0.0234375);
      expect(r.y).toBeCloseTo(0.9765625);
    });
  });

  // -- Gap (no descent) --

  describe("gap", () => {
    it("stays at root when x is in the gap", () => {
      // Binary: children start at x=0.5.  Cursor at x=0.3 — gap.
      const r = normalizeCursor(binary, { prefix: [], x: 0.3, y: 0.3 });
      expect(r.prefix).toEqual([]);
      expect(r.x).toBe(0.3);
      expect(r.y).toBe(0.3);
    });

    it("stays in a non-root square when in its gap", () => {
      const r = normalizeCursor(binary, {
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
    it("ascends and enters the next sibling on y overflow", () => {
      // prefix=['A'], y=1.1 → leave A, enter B.
      // Ascend: x=0.5+0.3·0.5=0.65, y=0+1.1·0.5=0.55
      // B: x≥0.5, y∈[0.5,1) → enter B.
      //   x=(0.65−0.5)/0.5=0.3, y=(0.55−0.5)/0.5=0.1
      const r = normalizeCursor(binary, {
        prefix: ["A"],
        x: 0.3,
        y: 1.1,
      });
      expect(r.prefix).toEqual(["B"]);
      expect(r.x).toBeCloseTo(0.3);
      expect(r.y).toBeCloseTo(0.1);
    });

    it("ascends and enters the previous sibling on y underflow", () => {
      // prefix=['B'], y=−0.1 → leave B, enter A.
      // Ascend: x=0.5+0.3·0.5=0.65, y=0.5+(−0.1)·0.5=0.45
      // A: x≥0.5, y∈[0,0.5) → enter A.
      //   x=(0.65−0.5)/0.5=0.3, y=0.45/0.5=0.9
      const r = normalizeCursor(binary, {
        prefix: ["B"],
        x: 0.3,
        y: -0.1,
      });
      expect(r.prefix).toEqual(["A"]);
      expect(r.x).toBeCloseTo(0.3);
      expect(r.y).toBeCloseTo(0.9);
    });

    it("ascends to the gap on x underflow", () => {
      // prefix=['A'], x=−0.1 → leave A.
      // Ascend: x=0.5+(−0.1)·0.5=0.45, y=0+0.3·0.5=0.15
      // x=0.45 < 0.5 — in the root's gap.
      const r = normalizeCursor(binary, {
        prefix: ["A"],
        x: -0.1,
        y: 0.3,
      });
      expect(r.prefix).toEqual([]);
      expect(r.x).toBeCloseTo(0.45);
      expect(r.y).toBeCloseTo(0.15);
    });

    it("ascends multiple levels if needed", () => {
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
      const r = normalizeCursor(binary, {
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
      model: LanguageModel<string>;
      state: CursorState<string>;
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
      it(`preserves position: ${name}`, () => {
        const result = normalizeCursor(model, state);
        expectSameGlobal(model, state, result);
      });
    }
  });

  // -- Edge cases --

  describe("edge cases", () => {
    it("empty distribution — no descent possible", () => {
      const r = normalizeCursor(empty, { prefix: [], x: 0.5, y: 0.5 });
      expect(r.prefix).toEqual([]);
      expect(r.x).toBe(0.5);
      expect(r.y).toBe(0.5);
    });

    it("deterministic model stops at maxDepth", () => {
      const r = normalizeCursor(
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

    it("clamps at root on extreme out-of-bounds", () => {
      const r = normalizeCursor(binary, { prefix: [], x: -5, y: 3 });
      // Clamped to (0, 1−ε), then descend.
      expect(r.x).toBeGreaterThanOrEqual(0);
      expect(r.x).toBeLessThan(1);
      expect(r.y).toBeGreaterThanOrEqual(0);
      expect(r.y).toBeLessThan(1);
    });

    it("handles cursor exactly on the y boundary between tokens", () => {
      // y = 0.5 exactly: top boundary of B (inclusive), not A.
      const r = normalizeCursor(binary, { prefix: [], x: 0.7, y: 0.5 });
      expect(r.prefix).toEqual(["B"]);
    });
  });

  // -- Generics: non-string tokens --

  describe("generic token types", () => {
    it("works with numeric tokens", () => {
      const numModel: LanguageModel<number> = () => [
        { token: 1, probability: 0.4 },
        { token: 2, probability: 0.6 },
      ];
      // Token 1: x∈[0.6,1] y∈[0,0.4).  Token 2: x∈[0.4,1] y∈[0.4,1).
      // (0.5, 0.5): token 1 x≥0.6? no.  token 2 x≥0.4 and y∈[0.4,1)? yes.
      const r = normalizeCursor(numModel, { prefix: [], x: 0.5, y: 0.5 });
      expect(r.prefix).toEqual([2]);
      expect(r.x).toBeCloseTo(1 / 6);
      expect(r.y).toBeCloseTo(1 / 6);
    });

    it("ascends correctly with object tokens and custom equality", () => {
      type Tok = { id: number };
      const objModel: LanguageModel<Tok> = () => [
        { token: { id: 1 }, probability: 0.5 },
        { token: { id: 2 }, probability: 0.5 },
      ];
      const eq = (a: Tok, b: Tok) => a.id === b.id;

      // prefix=[{id:1}], y=1.1 → ascend, enter {id:2}.
      const r = normalizeCursor(
        objModel,
        { prefix: [{ id: 1 }], x: 0.3, y: 1.1 },
        { tokenEquals: eq },
      );
      expect(r.prefix).toHaveLength(1);
      expect(r.prefix[0]).toEqual({ id: 2 });
      expect(r.x).toBeCloseTo(0.3);
      expect(r.y).toBeCloseTo(0.1);
    });
  });

  // -- Exact precision under deep ascent/descent --

  describe("arbitrary-precision exactness", () => {
    // These tests exercise scenarios where float64 arithmetic would
    // accumulate visible error: deep ascent all the way to the root,
    // then re-descent down a different branch.

    it("deep round-trip: ascend 20 levels then re-descend", () => {
      // Start 20 levels deep in the "A" branch, then move y just past
      // the boundary so we ascend all the way to the root and come back
      // down the "B" branch.  With float math the repeated
      //   y = cum + y * prob   (ascent)
      //   y = (y - cum) / prob  (descent)
      // would accumulate error; with exact rationals it's exact.
      const depth = 20;
      const prefix = Array<string>(depth).fill("A");
      // y = 1.0 + tiny nudge → just past the bottom of the A square at
      // every ancestor, forcing a full climb to root.
      const state: CursorState<string> = {
        prefix,
        x: 0.3,
        y: 1.0 + 1e-10,
      };
      const result = normalizeCursor(binary, state);
      expectSameGlobal(binary, state, result);
    });

    it("deep round-trip: ascend 50 levels with asymmetric model", () => {
      const depth = 50;
      const prefix = Array<string>(depth).fill("A");
      const state: CursorState<string> = {
        prefix,
        x: 0.3,
        y: 1.0 + 0.001,
      };
      const result = normalizeCursor(asym, state);
      expectSameGlobal(asym, state, result);
    });

    it("deep descent lands exactly at zero", () => {
      // x = 1 − 2^(−5) = 0.96875, y = 0 (stays 0 throughout).
      // After 5 descents into A: x should be exactly 0, y stays 0.
      const x = 1 - 2 ** -5;
      const r = normalizeCursor(binary, { prefix: [], x, y: 0 });
      expect(r.prefix).toEqual(["A", "A", "A", "A", "A"]);
      expect(r.x).toBe(0); // exact zero, not ≈0
      expect(r.y).toBe(0);
    });

    it("cross-branch sibling transfer at depth 30", () => {
      // Deeply nested in B, nudge y upward to cross into A at depth 30.
      // This forces 30 ascents and 30 descents through a different branch.
      const depth = 30;
      const prefix = Array<string>(depth).fill("B");
      const state: CursorState<string> = {
        prefix,
        x: 0.3,
        y: -0.001, // just above B → enters A
      };
      const result = normalizeCursor(binary, state);
      expectSameGlobal(binary, state, result);
      // Should have landed in the A branch at the same depth.
      expect(result.prefix.length).toBe(depth);
      expect(result.prefix[result.prefix.length - 1]).toBe("A");
    });

    it("many-token distribution preserves position exactly", () => {
      // 10 tokens with awkward probabilities whose float sum isn't exact.
      const probs = [
        0.17, 0.13, 0.11, 0.09, 0.07, 0.06, 0.05, 0.03, 0.02, 0.27,
      ];
      const tokens = "abcdefghij".split("");
      const manyModel: LanguageModel<string> = () =>
        tokens.map((token, i) => ({ token, probability: probs[i] }));

      // Nest 15 deep in "j" (prob 0.27), then nudge into "a".
      const prefix = Array<string>(15).fill("j");
      const state: CursorState<string> = {
        prefix,
        x: 0.4,
        y: -0.01,
      };
      const result = normalizeCursor(manyModel, state);
      expectSameGlobal(manyModel, state, result);
    });
  });
});

// ---------------------------------------------------------------------------
// cursorToGlobal
// ---------------------------------------------------------------------------

describe("cursorToGlobal", () => {
  it("root with (0,0) maps to global (0,0)", () => {
    const g = cursorToGlobal(binary, { prefix: [], x: 0, y: 0 });
    expect(g.x).toBeCloseTo(0);
    expect(g.y).toBeCloseTo(0);
  });

  it("root with (1,1) maps to global (1,1)", () => {
    const g = cursorToGlobal(binary, { prefix: [], x: 1, y: 1 });
    expect(g.x).toBeCloseTo(1);
    expect(g.y).toBeCloseTo(1);
  });

  it("prefix=[A] with (0,0) maps to A's top-left corner", () => {
    // Binary: A has size 0.5, left edge at 0.5, top edge at 0.
    const g = cursorToGlobal(binary, { prefix: ["A"], x: 0, y: 0 });
    expect(g.x).toBeCloseTo(0.5);
    expect(g.y).toBeCloseTo(0);
  });

  it("prefix=[B] with (0,0) maps to B's top-left corner", () => {
    // Binary: B has size 0.5, left edge at 0.5, top edge at 0.5.
    const g = cursorToGlobal(binary, { prefix: ["B"], x: 0, y: 0 });
    expect(g.x).toBeCloseTo(0.5);
    expect(g.y).toBeCloseTo(0.5);
  });

  it("composes correctly for a two-token prefix", () => {
    // prefix=[A, B] with asymmetric model: A=0.8, B=0.2.
    // A square: size=0.8, top=0.
    // In A's dist, B: cum=0.8, prob=0.2 → AB square: size=0.16, top=0+0.8·0.8=0.64.
    // (0, 0) → global (1−0.16, 0.64) = (0.84, 0.64).
    const g = cursorToGlobal(asym, { prefix: ["A", "B"], x: 0, y: 0 });
    expect(g.x).toBeCloseTo(0.84);
    expect(g.y).toBeCloseTo(0.64);
  });
});
