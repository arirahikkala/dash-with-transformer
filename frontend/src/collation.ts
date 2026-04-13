/**
 * Collation adapter: reorder selected tokens in a CDFView by inserting them
 * at chosen positions, rather than their natural CDF order.
 *
 * Example (Finnish): move å, ä, ö so they sort after z (and likewise Å/Ä/Ö
 * after Z), instead of appearing wherever UTF-8 byte order would place them.
 *
 * The adapter is generic over the token type T.  The caller supplies a
 * `tokenKey` function for equality comparison (since tokens are typically
 * structural objects).
 */
import { mergeAsyncIterables } from "./async-iterables";
import type { CDFView, TokenCDFExtent } from "./types";

/**
 * A collation rule: move `token` so that it appears immediately after
 * `after` in the collated CDF.  Multiple rules may share the same `after`;
 * they are inserted in the order given in the config array.
 *
 * A rule is silently ignored (the token falls back to its natural position)
 * when the `after` token has zero probability under the current prefix.
 */
export interface CollationRule<T> {
  readonly token: T;
  readonly after: T;
}

/**
 * Wrap a CDFView to apply a list of collation rules.  The resulting view
 * behaves identically to `inner` except that each configured token is
 * removed from its natural position and re-inserted after its predecessor.
 *
 * Total probability mass is preserved.  Extents do not overlap.
 */
export function withCollation<P, T>(
  inner: CDFView<P, T>,
  rules: readonly CollationRule<T>[],
  tokenKey: (t: T) => string,
): CDFView<P, T> {
  if (rules.length === 0) return inner;

  // Pre-index the rules: check for duplicates and collect the unique set of
  // tokens we'll need to look up on every prefix (moved tokens + predecessors).
  const movedKeys = new Set<string>();
  const seen = new Set<string>();
  const tokensToQuery: T[] = [];
  for (const r of rules) {
    const mk = tokenKey(r.token);
    if (movedKeys.has(mk)) {
      throw new Error(`Duplicate collation rule for token key: ${mk}`);
    }
    movedKeys.add(mk);
    if (!seen.has(mk)) {
      seen.add(mk);
      tokensToQuery.push(r.token);
    }
    const pk = tokenKey(r.after);
    if (!seen.has(pk)) {
      seen.add(pk);
      tokensToQuery.push(r.after);
    }
  }

  /**
   * Resolve which moved tokens are "active" at this prefix (both the moved
   * token and its predecessor have nonzero probability) and return their
   * natural extents plus the predecessor extents we need for shift math.
   */
  async function resolve(prefix: P): Promise<{
    moved: { rule: CollationRule<T>; extent: TokenCDFExtent<T> }[];
    predExtent: Map<string, TokenCDFExtent<T>>;
    totalMovedMass: number;
  }> {
    const extents = await Promise.all(
      tokensToQuery.map((t) => inner.token(prefix, t)),
    );
    const extentByKey = new Map<string, TokenCDFExtent<T>>();
    for (let i = 0; i < tokensToQuery.length; i++) {
      const e = extents[i];
      if (e) extentByKey.set(tokenKey(tokensToQuery[i]), e);
    }

    const moved: { rule: CollationRule<T>; extent: TokenCDFExtent<T> }[] = [];
    let totalMovedMass = 0;
    for (const r of rules) {
      const me = extentByKey.get(tokenKey(r.token));
      const pe = extentByKey.get(tokenKey(r.after));
      if (!me || !pe) continue; // rule inactive: missing token or predecessor
      moved.push({ rule: r, extent: me });
      totalMovedMass += me.end - me.start;
    }

    return { moved, predExtent: extentByKey, totalMovedMass };
  }

  /**
   * Collated-minus-natural offset for a stayed token at natural start `s`.
   *
   *   shift(s) = preColl(s) − preNat(s)
   *
   * where
   *   preNat(s)  = Σ |m| for active m whose natural extent ends ≤ s
   *                (i.e. moved tokens that sat before this one naturally,
   *                 and whose removal shifts us left)
   *   preColl(s) = Σ |m| for active m whose predecessor sits strictly before
   *                s in the natural CDF (i.e. moved tokens now inserted
   *                before this one, shifting us right)
   */
  function shiftAtNatural(
    s: number,
    moved: { rule: CollationRule<T>; extent: TokenCDFExtent<T> }[],
    predExtent: Map<string, TokenCDFExtent<T>>,
  ): number {
    let preNat = 0;
    let preColl = 0;
    for (const m of moved) {
      const size = m.extent.end - m.extent.start;
      if (m.extent.end <= s) preNat += size;
      const pred = predExtent.get(tokenKey(m.rule.after))!;
      if (pred.start < s) preColl += size;
    }
    return preColl - preNat;
  }

  /**
   * Compute collated extents for all active moved tokens, grouped by
   * predecessor (so ties at the same insertion slot resolve in config order).
   */
  function collatedMoved(
    moved: { rule: CollationRule<T>; extent: TokenCDFExtent<T> }[],
    predExtent: Map<string, TokenCDFExtent<T>>,
  ): TokenCDFExtent<T>[] {
    // Bucket by predecessor key, preserving config order within each bucket.
    const buckets = new Map<
      string,
      { rule: CollationRule<T>; extent: TokenCDFExtent<T> }[]
    >();
    for (const m of moved) {
      const pk = tokenKey(m.rule.after);
      let b = buckets.get(pk);
      if (!b) {
        b = [];
        buckets.set(pk, b);
      }
      b.push(m);
    }

    const out: TokenCDFExtent<T>[] = [];
    for (const [pk, group] of buckets) {
      const pred = predExtent.get(pk)!;
      // Predecessor's collated end position.
      const predShift = shiftAtNatural(pred.start, moved, predExtent);
      let cursor = pred.end + predShift;
      for (const m of group) {
        const prob = m.extent.end - m.extent.start;
        out.push({ token: m.rule.token, start: cursor, end: cursor + prob });
        cursor += prob;
      }
    }
    return out;
  }

  return {
    async *slice(prefix, rangeStart, rangeEnd, minProb) {
      // Kick off the core natural-range slice in parallel with resolve().
      const coreIter = inner.slice(prefix, rangeStart, rangeEnd, minProb);
      const firstCore = coreIter.next();

      const { moved, predExtent, totalMovedMass } = await resolve(prefix);

      // Emit moved tokens at their collated positions.
      for (const mc of collatedMoved(moved, predExtent)) {
        if (mc.end <= rangeStart || mc.start > rangeEnd) continue;
        if (mc.end - mc.start < minProb) continue;
        yield mc;
      }

      const activeMovedKeys = new Set(moved.map((m) => tokenKey(m.rule.token)));

      // Widen by the total moved mass on each side.  A stayed token's
      // collated position differs from its natural position by at most
      // ±totalMovedMass, so the two wings cover every token that could
      // land in [rangeStart, rangeEnd] after shifting.
      const leftLo = Math.max(0, rangeStart - totalMovedMass);
      const rightHi = Math.min(1, rangeEnd + totalMovedMass);

      async function* primedCore() {
        const r = await firstCore;
        if (r.done) return;
        yield r.value;
        yield* coreIter;
      }

      const seen = new Set<string>();
      for await (const ext of mergeAsyncIterables([
        primedCore(),
        inner.slice(prefix, leftLo, rangeStart, minProb),
        inner.slice(prefix, rangeEnd, rightHi, minProb),
      ])) {
        const k = tokenKey(ext.token);
        if (seen.has(k)) continue;
        seen.add(k);
        if (activeMovedKeys.has(k)) continue;
        const shift = shiftAtNatural(ext.start, moved, predExtent);
        const s = ext.start + shift;
        const e = ext.end + shift;
        if (e <= rangeStart || s > rangeEnd) continue;
        yield { token: ext.token, start: s, end: e };
      }
    },

    async token(prefix, specificToken) {
      const key = tokenKey(specificToken);
      const { moved, predExtent } = await resolve(prefix);

      // Active moved token: return its collated extent.
      const asMoved = moved.find((m) => tokenKey(m.rule.token) === key);
      if (asMoved) {
        const predKey = tokenKey(asMoved.rule.after);
        const pred = predExtent.get(predKey)!;
        const predShift = shiftAtNatural(pred.start, moved, predExtent);
        let cursor = pred.end + predShift;
        // Step past earlier same-slot siblings (config order).
        for (const m of moved) {
          if (tokenKey(m.rule.after) !== predKey) continue;
          if (tokenKey(m.rule.token) === key) break;
          cursor += m.extent.end - m.extent.start;
        }
        const prob = asMoved.extent.end - asMoved.extent.start;
        return { token: specificToken, start: cursor, end: cursor + prob };
      }

      // Otherwise (stayed, or inactive-moved falling back to natural):
      // query inner and apply the stayed-token shift.
      const ext = await inner.token(prefix, specificToken);
      if (!ext) return null;
      const shift = shiftAtNatural(ext.start, moved, predExtent);
      return {
        token: specificToken,
        start: ext.start + shift,
        end: ext.end + shift,
      };
    },
  };
}
