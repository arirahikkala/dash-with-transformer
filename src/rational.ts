/**
 * Exact rational arithmetic using native BigInt.
 *
 * Every IEEE-754 float64 is a dyadic rational (n / 2^k), so converting
 * model probabilities to Rats and doing all intermediate arithmetic
 * exactly means the ascent/descent coordinate transforms in the cursor
 * normaliser are perfect inverses — no error accumulates regardless of
 * tree depth.
 */

/** Exact rational: n / d, with d > 0. */
export interface Rat {
  readonly n: bigint;
  readonly d: bigint;
}

export const ZERO: Rat = { n: 0n, d: 1n };
export const ONE: Rat = { n: 1n, d: 1n };

// -- internal helpers -------------------------------------------------------

function bigAbs(x: bigint): bigint {
  return x < 0n ? -x : x;
}

function gcd(a: bigint, b: bigint): bigint {
  a = bigAbs(a);
  b = bigAbs(b);
  while (b !== 0n) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a;
}

/** Reduce to lowest terms, positive denominator. */
export function reduce({ n, d }: Rat): Rat {
  if (n === 0n) return ZERO;
  const g = gcd(n, d);
  const rn = n / g;
  const rd = d / g;
  return rd < 0n ? { n: -rn, d: -rd } : { n: rn, d: rd };
}

// -- arithmetic -------------------------------------------------------------

export function add(a: Rat, b: Rat): Rat {
  return reduce({ n: a.n * b.d + b.n * a.d, d: a.d * b.d });
}

export function sub(a: Rat, b: Rat): Rat {
  return reduce({ n: a.n * b.d - b.n * a.d, d: a.d * b.d });
}

export function mul(a: Rat, b: Rat): Rat {
  return reduce({ n: a.n * b.n, d: a.d * b.d });
}

export function div(a: Rat, b: Rat): Rat {
  if (b.n === 0n) throw new Error("division by zero");
  const n = a.n * b.d;
  const d = a.d * b.n;
  return d < 0n ? reduce({ n: -n, d: -d }) : reduce({ n, d });
}

// -- comparison -------------------------------------------------------------

/** a < b */
export function lt(a: Rat, b: Rat): boolean {
  // Both denominators are positive after reduce(), so cross-multiply is safe.
  return a.n * b.d < b.n * a.d;
}

/** a ≥ b */
export function gte(a: Rat, b: Rat): boolean {
  return a.n * b.d >= b.n * a.d;
}

// -- float conversion -------------------------------------------------------

// Shared buffer for IEEE-754 bit extraction.
const _buf = new ArrayBuffer(8);
const _f64 = new Float64Array(_buf);
const _u64 = new BigUint64Array(_buf);

/**
 * Convert an IEEE-754 float64 to the *exact* rational it represents.
 *
 * Every finite float64 is a dyadic rational m · 2^e, so this is lossless.
 */
export function fromFloat(x: number): Rat {
  if (x === 0) return ZERO;
  if (!Number.isFinite(x)) throw new Error(`not finite: ${x}`);

  _f64[0] = x;
  const bits = _u64[0];

  const sign = bits >> 63n;
  const expBiased = Number((bits >> 52n) & 0x7ffn);
  let mantissa = bits & ((1n << 52n) - 1n);
  let exp: number;

  if (expBiased === 0) {
    // Sub-normal: no implicit leading 1.
    exp = 1 - 1023 - 52; // −1074
  } else {
    mantissa |= 1n << 52n; // implicit leading 1
    exp = expBiased - 1023 - 52;
  }

  const n = sign !== 0n ? -mantissa : mantissa;

  return exp >= 0
    ? reduce({ n: n << BigInt(exp), d: 1n })
    : reduce({ n, d: 1n << BigInt(-exp) });
}

/**
 * Convert a rational to the nearest float64.
 *
 * When the numerator or denominator is too large for Number() (which
 * would overflow to Infinity), we shift both down to ~53 significant
 * bits and compensate with a power-of-two scale factor.
 */
export function toFloat(r: Rat): number {
  if (r.n === 0n) return 0;

  const absN = r.n < 0n ? -r.n : r.n;
  const sign = r.n < 0n ? -1 : 1;

  if (r.d === 1n) return sign * Number(absN);

  const nBits = bitLen(absN);
  const dBits = bitLen(r.d);

  // If both fit in float64 range (< 2^1023), just divide directly.
  if (nBits <= 1023 && dBits <= 1023) {
    return Number(r.n) / Number(r.d);
  }

  // Scale both to ~53 significant bits, preserving their ratio
  // up to float64 precision.  result = (sn / sd) · 2^(nShift − dShift).
  const nShift = Math.max(0, nBits - 53);
  const dShift = Math.max(0, dBits - 53);

  const sn = nShift > 0 ? absN >> BigInt(nShift) : absN;
  const sd = dShift > 0 ? r.d >> BigInt(dShift) : r.d;

  if (sd === 0n) return sign * Infinity;
  return sign * (Number(sn) / Number(sd)) * 2 ** (nShift - dShift);
}

/** Number of bits needed to represent n (0 → 0). */
function bitLen(n: bigint): number {
  if (n <= 0n) return 0;
  // toString(2) is simple and fast enough for the sizes we encounter
  // (a few thousand bits at most).
  return n.toString(2).length;
}
