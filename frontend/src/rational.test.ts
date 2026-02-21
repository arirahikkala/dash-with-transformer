import { describe, it, expect } from "vitest";
import {
  ZERO,
  ONE,
  add,
  sub,
  mul,
  div,
  lt,
  gte,
  fromFloat,
  toFloat,
  reduce,
} from "./rational";

describe("fromFloat", () => {
  it("converts 0", () => {
    const r = fromFloat(0);
    expect(r.n).toBe(0n);
    expect(r.d).toBe(1n);
  });

  it("converts 1", () => {
    const r = fromFloat(1);
    expect(r.n).toBe(1n);
    expect(r.d).toBe(1n);
  });

  it("converts 0.5 exactly", () => {
    const r = fromFloat(0.5);
    expect(r.n).toBe(1n);
    expect(r.d).toBe(2n);
  });

  it("converts 0.25 exactly", () => {
    const r = fromFloat(0.25);
    expect(r.n).toBe(1n);
    expect(r.d).toBe(4n);
  });

  it("converts negative numbers", () => {
    const r = fromFloat(-0.5);
    expect(r.n).toBe(-1n);
    expect(r.d).toBe(2n);
  });

  it("round-trips arbitrary floats", () => {
    const values = [0.1, 0.2, 0.3, 0.7, 0.9, 0.123456789, 1e-10, 0.999999];
    for (const v of values) {
      expect(toFloat(fromFloat(v))).toBe(v);
    }
  });

  it("throws on Infinity", () => {
    expect(() => fromFloat(Infinity)).toThrow();
  });

  it("throws on NaN", () => {
    expect(() => fromFloat(NaN)).toThrow();
  });
});

describe("arithmetic", () => {
  const HALF = fromFloat(0.5);
  const THIRD = div({ n: 1n, d: 1n }, { n: 3n, d: 1n });

  it("add", () => {
    const r = add(HALF, HALF);
    expect(r.n).toBe(1n);
    expect(r.d).toBe(1n);
  });

  it("sub", () => {
    const r = sub(ONE, HALF);
    expect(r.n).toBe(1n);
    expect(r.d).toBe(2n);
  });

  it("mul", () => {
    const r = mul(HALF, HALF);
    expect(r.n).toBe(1n);
    expect(r.d).toBe(4n);
  });

  it("div", () => {
    const r = div(ONE, HALF);
    expect(r.n).toBe(2n);
    expect(r.d).toBe(1n);
  });

  it("div by zero throws", () => {
    expect(() => div(ONE, ZERO)).toThrow("division by zero");
  });

  it("1/3 + 1/3 + 1/3 = 1 exactly", () => {
    const r = add(add(THIRD, THIRD), THIRD);
    expect(r.n).toBe(1n);
    expect(r.d).toBe(1n);
  });
});

describe("comparison", () => {
  it("lt", () => {
    expect(lt(ZERO, ONE)).toBe(true);
    expect(lt(ONE, ZERO)).toBe(false);
    expect(lt(ZERO, ZERO)).toBe(false);
  });

  it("gte", () => {
    expect(gte(ONE, ZERO)).toBe(true);
    expect(gte(ONE, ONE)).toBe(true);
    expect(gte(ZERO, ONE)).toBe(false);
  });
});

describe("reduce", () => {
  it("reduces to lowest terms", () => {
    const r = reduce({ n: 6n, d: 4n });
    expect(r.n).toBe(3n);
    expect(r.d).toBe(2n);
  });

  it("normalises negative denominator", () => {
    const r = reduce({ n: 1n, d: -2n });
    expect(r.n).toBe(-1n);
    expect(r.d).toBe(2n);
  });
});

describe("toFloat with large BigInts", () => {
  it("handles rationals with very large numerator and denominator", () => {
    // 0.8^50 computed via exact rationals.
    const p = fromFloat(0.8);
    let r = ONE;
    for (let i = 0; i < 50; i++) r = mul(r, p);

    const f = toFloat(r);
    expect(f).toBeCloseTo(0.8 ** 50, 5);
    expect(Number.isFinite(f)).toBe(true);
  });

  it("handles rationals with >1023-bit components", () => {
    // Construct n / d where both are > 2^1023.
    const big = 1n << 2000n;
    const r = { n: big + 1n, d: big * 2n };
    const f = toFloat(r);
    expect(f).toBeCloseTo(0.5, 5);
    expect(Number.isFinite(f)).toBe(true);
  });
});
