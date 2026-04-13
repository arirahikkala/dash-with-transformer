import { describe, it, expect } from "vitest";
import { withCollation, type CollationRule } from "./collation";
import type { CDFView, TokenCDFExtent } from "./types";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/** Collect all items from an async iterable into an array. */
async function collect<T>(iter: AsyncIterable<T>): Promise<T[]> {
  const result: T[] = [];
  for await (const item of iter) result.push(item);
  return result;
}

/**
 * Build a trivial CDFView over string tokens.  Each prefix maps to an ordered
 * list of (token, prob) pairs; extents are computed by cumulative sum.
 */
function makeCDF(
  dists: Record<string, { token: string; prob: number }[]>,
): CDFView<string, string> {
  function extents(prefix: string): TokenCDFExtent<string>[] {
    const d = dists[prefix] ?? [];
    const out: TokenCDFExtent<string>[] = [];
    let cum = 0;
    for (const { token, prob } of d) {
      const start = cum;
      const end = cum + prob;
      cum = end;
      if (prob > 0) out.push({ token, start, end });
    }
    return out;
  }
  return {
    async *slice(prefix, rs, re, mp) {
      for (const ext of extents(prefix)) {
        if (ext.end <= rs || ext.start > re) continue;
        if (ext.end - ext.start < mp) continue;
        yield ext;
      }
    },
    async token(prefix, t) {
      for (const ext of extents(prefix)) {
        if (ext.token === t) return ext;
      }
      return null;
    },
  };
}

const keyOf = (s: string) => s;

/** Close-enough comparison for cumulative sums. */
function near(a: number, b: number, eps = 1e-9) {
  return Math.abs(a - b) < eps;
}

/** Assert extents are contiguous, non-overlapping, and sum to `expectedTotal`. */
function expectContiguous(
  extents: TokenCDFExtent<string>[],
  expectedTotal: number,
) {
  const sorted = [...extents].sort((a, b) => a.start - b.start);
  let cum = 0;
  for (const e of sorted) {
    expect(near(e.start, cum)).toBe(true);
    expect(e.end).toBeGreaterThan(e.start);
    cum = e.end;
  }
  expect(near(cum, expectedTotal)).toBe(true);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("withCollation", () => {
  it("is the identity when no rules are configured", async () => {
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.5 },
        { token: "b", prob: 0.3 },
        { token: "c", prob: 0.2 },
      ],
    });
    const view = withCollation(inner, [], keyOf);
    expect(view).toBe(inner);
  });

  it("leaves extents unchanged when rules exist but none fire", async () => {
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.5 },
        { token: "b", prob: 0.5 },
      ],
    });
    // 'å' doesn't exist in the distribution → rule is inactive.
    const view = withCollation(inner, [{ token: "å", after: "a" }], keyOf);
    const got = await collect(view.slice("", 0, 1, 0));
    expect(got).toHaveLength(2);
    expectContiguous(got, 1);
    const a = got.find((e) => e.token === "a")!;
    const b = got.find((e) => e.token === "b")!;
    expect(near(a.start, 0)).toBe(true);
    expect(near(a.end, 0.5)).toBe(true);
    expect(near(b.start, 0.5)).toBe(true);
    expect(near(b.end, 1)).toBe(true);
  });

  it("moves one token to just after its predecessor", async () => {
    // Natural order: a, b, c, å, d.  Rule: å after b.
    // Collated order: a, b, å, c, d.
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.1 },
        { token: "b", prob: 0.2 },
        { token: "c", prob: 0.3 },
        { token: "å", prob: 0.15 },
        { token: "d", prob: 0.25 },
      ],
    });
    const rules: CollationRule<string>[] = [{ token: "å", after: "b" }];
    const view = withCollation(inner, rules, keyOf);
    const got = await collect(view.slice("", 0, 1, 0));

    const ordered = [...got].sort((a, b) => a.start - b.start);
    expect(ordered.map((e) => e.token)).toEqual(["a", "b", "å", "c", "d"]);
    expectContiguous(ordered, 1);

    const find = (t: string) => ordered.find((e) => e.token === t)!;
    expect(near(find("a").start, 0)).toBe(true);
    expect(near(find("a").end, 0.1)).toBe(true);
    expect(near(find("b").start, 0.1)).toBe(true);
    expect(near(find("b").end, 0.3)).toBe(true);
    expect(near(find("å").start, 0.3)).toBe(true);
    expect(near(find("å").end, 0.45)).toBe(true);
    expect(near(find("c").start, 0.45)).toBe(true);
    expect(near(find("c").end, 0.75)).toBe(true);
    expect(near(find("d").start, 0.75)).toBe(true);
    expect(near(find("d").end, 1)).toBe(true);
  });

  it("preserves config order when multiple tokens share a predecessor", async () => {
    // Natural: a, b, c, ä, å, ö.  Rules (in this order): å, ä, ö all after b.
    // Collated: a, b, å, ä, ö, c.
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.1 },
        { token: "b", prob: 0.2 },
        { token: "c", prob: 0.3 },
        { token: "ä", prob: 0.1 },
        { token: "å", prob: 0.15 },
        { token: "ö", prob: 0.15 },
      ],
    });
    const rules: CollationRule<string>[] = [
      { token: "å", after: "b" },
      { token: "ä", after: "b" },
      { token: "ö", after: "b" },
    ];
    const view = withCollation(inner, rules, keyOf);
    const got = await collect(view.slice("", 0, 1, 0));
    const ordered = [...got].sort((a, b) => a.start - b.start);
    expect(ordered.map((e) => e.token)).toEqual(["a", "b", "å", "ä", "ö", "c"]);
    expectContiguous(ordered, 1);

    const find = (t: string) => ordered.find((e) => e.token === t)!;
    // b ends at 0.3; then å (0.15), ä (0.1), ö (0.15); then c (0.3).
    expect(near(find("å").start, 0.3)).toBe(true);
    expect(near(find("å").end, 0.45)).toBe(true);
    expect(near(find("ä").start, 0.45)).toBe(true);
    expect(near(find("ä").end, 0.55)).toBe(true);
    expect(near(find("ö").start, 0.55)).toBe(true);
    expect(near(find("ö").end, 0.7)).toBe(true);
    expect(near(find("c").start, 0.7)).toBe(true);
    expect(near(find("c").end, 1)).toBe(true);
  });

  it("handles rules with different predecessors", async () => {
    // Natural: a, b, c, å, Å, d.  Rules: å after a, Å after c.
    // Collated: a, å, b, c, Å, d.
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.1 },
        { token: "b", prob: 0.2 },
        { token: "c", prob: 0.2 },
        { token: "å", prob: 0.1 },
        { token: "Å", prob: 0.1 },
        { token: "d", prob: 0.3 },
      ],
    });
    const rules: CollationRule<string>[] = [
      { token: "å", after: "a" },
      { token: "Å", after: "c" },
    ];
    const view = withCollation(inner, rules, keyOf);
    const got = await collect(view.slice("", 0, 1, 0));
    const ordered = [...got].sort((a, b) => a.start - b.start);
    expect(ordered.map((e) => e.token)).toEqual(["a", "å", "b", "c", "Å", "d"]);
    expectContiguous(ordered, 1);
  });

  it("drops a moved token whose natural probability is zero", async () => {
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.5 },
        { token: "b", prob: 0.5 },
      ],
    });
    const view = withCollation(inner, [{ token: "å", after: "a" }], keyOf);
    const got = await collect(view.slice("", 0, 1, 0));
    expect(got.map((e) => e.token).sort()).toEqual(["a", "b"]);
    expect(await view.token("", "å")).toBeNull();
  });

  it("falls back to natural position when the predecessor has zero probability", async () => {
    // 'Z' is absent, so å's collation rule is inactive and å keeps its
    // natural position.
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.4 },
        { token: "b", prob: 0.4 },
        { token: "å", prob: 0.2 },
      ],
    });
    const view = withCollation(inner, [{ token: "å", after: "Z" }], keyOf);
    const got = await collect(view.slice("", 0, 1, 0));
    const ordered = [...got].sort((a, b) => a.start - b.start);
    expect(ordered.map((e) => e.token)).toEqual(["a", "b", "å"]);
    expectContiguous(ordered, 1);

    const å = await view.token("", "å");
    expect(å).not.toBeNull();
    expect(near(å!.start, 0.8)).toBe(true);
    expect(near(å!.end, 1)).toBe(true);
  });

  it("token() returns the collated extent for a moved token", async () => {
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.1 },
        { token: "b", prob: 0.2 },
        { token: "c", prob: 0.3 },
        { token: "å", prob: 0.15 },
        { token: "d", prob: 0.25 },
      ],
    });
    const view = withCollation(inner, [{ token: "å", after: "b" }], keyOf);
    const å = await view.token("", "å");
    expect(å).not.toBeNull();
    expect(near(å!.start, 0.3)).toBe(true);
    expect(near(å!.end, 0.45)).toBe(true);
  });

  it("token() returns the shifted extent for a stayed token", async () => {
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.1 },
        { token: "b", prob: 0.2 },
        { token: "c", prob: 0.3 },
        { token: "å", prob: 0.15 },
        { token: "d", prob: 0.25 },
      ],
    });
    const view = withCollation(inner, [{ token: "å", after: "b" }], keyOf);
    const c = await view.token("", "c");
    expect(c).not.toBeNull();
    // c naturally at [0.3, 0.6]; after å (+0.15) inserted before it, c → [0.45, 0.75].
    expect(near(c!.start, 0.45)).toBe(true);
    expect(near(c!.end, 0.75)).toBe(true);

    const a = await view.token("", "a");
    // a is before the insertion slot, so it is unshifted.
    expect(near(a!.start, 0)).toBe(true);
    expect(near(a!.end, 0.1)).toBe(true);

    const d = await view.token("", "d");
    // d is after å both naturally and collated; å moves from after-c to
    // after-b, both of which are before d.  Net shift: 0.
    expect(near(d!.start, 0.75)).toBe(true);
    expect(near(d!.end, 1)).toBe(true);
  });

  it("token() returns null for an absent token regardless of collation", async () => {
    const inner = makeCDF({
      "": [{ token: "a", prob: 1.0 }],
    });
    const view = withCollation(inner, [{ token: "å", after: "a" }], keyOf);
    expect(await view.token("", "å")).toBeNull();
    expect(await view.token("", "zzz")).toBeNull();
  });

  it("respects rangeStart/rangeEnd in slice (collated coordinates)", async () => {
    // Natural: a(0.1), b(0.2), c(0.3), å(0.15), d(0.25).
    // Collated: a[0,0.1), b[0.1,0.3), å[0.3,0.45), c[0.45,0.75), d[0.75,1).
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.1 },
        { token: "b", prob: 0.2 },
        { token: "c", prob: 0.3 },
        { token: "å", prob: 0.15 },
        { token: "d", prob: 0.25 },
      ],
    });
    const view = withCollation(inner, [{ token: "å", after: "b" }], keyOf);
    // Window [0.35, 0.5] should hit å and c only.
    const got = await collect(view.slice("", 0.35, 0.5, 0));
    const tokens = new Set(got.map((e) => e.token));
    expect(tokens).toEqual(new Set(["å", "c"]));
  });

  it("respects minProb in slice (applies to moved and stayed tokens)", async () => {
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.6 },
        { token: "b", prob: 0.35 },
        { token: "å", prob: 0.05 },
      ],
    });
    const view = withCollation(inner, [{ token: "å", after: "a" }], keyOf);
    // minProb = 0.1 excludes å (0.05).
    const got = await collect(view.slice("", 0, 1, 0.1));
    expect(got.map((e) => e.token).sort()).toEqual(["a", "b"]);
  });

  it("keeps extents non-overlapping under realistic-ish distributions", async () => {
    // A larger distribution, several rules with mixed predecessors.
    const tokens: { token: string; prob: number }[] = [];
    const n = 20;
    for (let i = 0; i < n; i++) {
      tokens.push({ token: `t${i}`, prob: 1 / n });
    }
    const inner = makeCDF({ "": tokens });
    const rules: CollationRule<string>[] = [
      { token: "t5", after: "t0" },
      { token: "t15", after: "t0" },
      { token: "t8", after: "t2" },
      { token: "t18", after: "t10" },
    ];
    const view = withCollation(inner, rules, keyOf);
    const got = await collect(view.slice("", 0, 1, 0));
    expect(got).toHaveLength(n);
    expectContiguous(got, 1);

    // Expected order: t0, t5, t15, t1, t2, t8, t3, t4, t6, t7, t9, t10,
    //                 t18, t11, t12, t13, t14, t16, t17, t19
    const ordered = [...got].sort((a, b) => a.start - b.start);
    expect(ordered.map((e) => e.token)).toEqual([
      "t0",
      "t5",
      "t15",
      "t1",
      "t2",
      "t8",
      "t3",
      "t4",
      "t6",
      "t7",
      "t9",
      "t10",
      "t18",
      "t11",
      "t12",
      "t13",
      "t14",
      "t16",
      "t17",
      "t19",
    ]);
  });

  it("agrees between slice and token on extents", async () => {
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.1 },
        { token: "b", prob: 0.2 },
        { token: "c", prob: 0.3 },
        { token: "å", prob: 0.15 },
        { token: "ä", prob: 0.1 },
        { token: "d", prob: 0.15 },
      ],
    });
    const rules: CollationRule<string>[] = [
      { token: "å", after: "b" },
      { token: "ä", after: "b" },
    ];
    const view = withCollation(inner, rules, keyOf);
    const sliceExtents = await collect(view.slice("", 0, 1, 0));
    for (const ext of sliceExtents) {
      const byToken = await view.token("", ext.token);
      expect(byToken).not.toBeNull();
      expect(near(byToken!.start, ext.start)).toBe(true);
      expect(near(byToken!.end, ext.end)).toBe(true);
    }
  });

  it("moves a token from an earlier natural position to a later collated slot", async () => {
    // Natural: a, å, b, c, d.  Rule: å after c.
    // Collated: a, b, c, å, d.
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.1 },
        { token: "å", prob: 0.2 },
        { token: "b", prob: 0.3 },
        { token: "c", prob: 0.2 },
        { token: "d", prob: 0.2 },
      ],
    });
    const view = withCollation(inner, [{ token: "å", after: "c" }], keyOf);
    const got = await collect(view.slice("", 0, 1, 0));
    const ordered = [...got].sort((a, b) => a.start - b.start);
    expect(ordered.map((e) => e.token)).toEqual(["a", "b", "c", "å", "d"]);
    expectContiguous(ordered, 1);

    const find = (t: string) => ordered.find((e) => e.token === t)!;
    // a is before å naturally and before c collated → no shift.
    expect(near(find("a").start, 0)).toBe(true);
    expect(near(find("a").end, 0.1)).toBe(true);
    // b and c have å removed from before them (−0.2) and no insertions
    // before them collated → shift = −0.2.
    expect(near(find("b").start, 0.1)).toBe(true);
    expect(near(find("b").end, 0.4)).toBe(true);
    expect(near(find("c").start, 0.4)).toBe(true);
    expect(near(find("c").end, 0.6)).toBe(true);
    // å is inserted right after c's collated end.
    expect(near(find("å").start, 0.6)).toBe(true);
    expect(near(find("å").end, 0.8)).toBe(true);
    // d had å removed from before (−0.2) and å re-inserted before (+0.2)
    // → net shift 0.
    expect(near(find("d").start, 0.8)).toBe(true);
    expect(near(find("d").end, 1)).toBe(true);

    // token() should agree on every position above.
    for (const t of ["a", "b", "c", "å", "d"]) {
      const byToken = await view.token("", t);
      expect(byToken).not.toBeNull();
      expect(near(byToken!.start, find(t).start)).toBe(true);
      expect(near(byToken!.end, find(t).end)).toBe(true);
    }
  });

  it("handles one earlier-move and one later-move in the same config", async () => {
    // Natural: a, ö, b, c, ä, d.
    // Rules: ö after c (later), ä after a (earlier).
    // Collated: a, ä, b, c, ö, d.
    const inner = makeCDF({
      "": [
        { token: "a", prob: 0.1 },
        { token: "ö", prob: 0.1 },
        { token: "b", prob: 0.2 },
        { token: "c", prob: 0.2 },
        { token: "ä", prob: 0.2 },
        { token: "d", prob: 0.2 },
      ],
    });
    const view = withCollation(
      inner,
      [
        { token: "ö", after: "c" },
        { token: "ä", after: "a" },
      ],
      keyOf,
    );
    const got = await collect(view.slice("", 0, 1, 0));
    const ordered = [...got].sort((a, b) => a.start - b.start);
    expect(ordered.map((e) => e.token)).toEqual(["a", "ä", "b", "c", "ö", "d"]);
    expectContiguous(ordered, 1);

    const find = (t: string) => ordered.find((e) => e.token === t)!;
    expect(near(find("a").end, 0.1)).toBe(true);
    expect(near(find("ä").start, 0.1)).toBe(true);
    expect(near(find("ä").end, 0.3)).toBe(true);
    expect(near(find("b").start, 0.3)).toBe(true);
    expect(near(find("b").end, 0.5)).toBe(true);
    expect(near(find("c").start, 0.5)).toBe(true);
    expect(near(find("c").end, 0.7)).toBe(true);
    expect(near(find("ö").start, 0.7)).toBe(true);
    expect(near(find("ö").end, 0.8)).toBe(true);
    expect(near(find("d").start, 0.8)).toBe(true);
    expect(near(find("d").end, 1)).toBe(true);
  });

  it("throws on duplicate rules for the same moved token", () => {
    const inner = makeCDF({ "": [{ token: "a", prob: 1 }] });
    expect(() =>
      withCollation(
        inner,
        [
          { token: "å", after: "a" },
          { token: "å", after: "a" },
        ],
        keyOf,
      ),
    ).toThrow(/duplicate/i);
  });
});
