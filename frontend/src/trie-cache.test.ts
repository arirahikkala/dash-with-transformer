import { describe, it, expect } from "vitest";
import { createTrieCache } from "./trie-cache";

function prefix(...bytes: number[]): Uint8Array {
  return new Uint8Array(bytes);
}

describe("createTrieCache", () => {
  it("returns undefined for a missing key", () => {
    const cache = createTrieCache<string>();
    expect(cache.get(prefix(1, 2, 3))).toBeUndefined();
  });

  it("round-trips a value through set/get", () => {
    const cache = createTrieCache<string>();
    cache.set(prefix(1, 2), "hello");
    expect(cache.get(prefix(1, 2))).toBe("hello");
  });

  it("distinguishes different prefixes", () => {
    const cache = createTrieCache<string>();
    cache.set(prefix(1, 2), "a");
    cache.set(prefix(1, 3), "b");
    expect(cache.get(prefix(1, 2))).toBe("a");
    expect(cache.get(prefix(1, 3))).toBe("b");
  });

  it("handles the empty prefix", () => {
    const cache = createTrieCache<number>();
    cache.set(prefix(), 42);
    expect(cache.get(prefix())).toBe(42);
  });

  it("delete removes a cached value", () => {
    const cache = createTrieCache<string>();
    cache.set(prefix(1), "x");
    cache.delete(prefix(1));
    expect(cache.get(prefix(1))).toBeUndefined();
  });

  it("delete is a no-op for a missing key", () => {
    const cache = createTrieCache<string>();
    cache.delete(prefix(1, 2, 3)); // should not throw
  });

  it("getOrSet returns cached value without calling compute", () => {
    const cache = createTrieCache<string>();
    cache.set(prefix(1), "cached");
    let called = false;
    const result = cache.getOrSet(prefix(1), () => {
      called = true;
      return "computed";
    });
    expect(result).toBe("cached");
    expect(called).toBe(false);
  });

  it("getOrSet computes and caches on a miss", () => {
    const cache = createTrieCache<string>();
    const result = cache.getOrSet(prefix(1), () => "computed");
    expect(result).toBe("computed");
    expect(cache.get(prefix(1))).toBe("computed");
  });
});

describe("generation-based pruning", () => {
  // Generation only advances on set/getOrSet-miss, so we use sets on a
  // throwaway key to burn through generations.
  function burnGenerations(
    cache: ReturnType<typeof createTrieCache<number>>,
    n: number,
  ) {
    for (let i = 0; i < n; i++) {
      cache.set(prefix(200, i & 0xff, (i >> 8) & 0xff), 0);
    }
  }

  it("evicts stale subtrees after pruneInterval ticks", () => {
    // pruneInterval=10, maxAge=5: after 10 ticks, anything not
    // touched in the last 5 generations gets clipped.
    const cache = createTrieCache<number>(10, 5);

    cache.set(prefix(1), 1); // generation ~1
    cache.set(prefix(2), 2); // generation ~2

    // Burn through generations with sets on a different key.
    burnGenerations(cache, 10);

    // Prune should have fired by now and clipped prefix(1) and prefix(2).
    expect(cache.get(prefix(1))).toBeUndefined();
    expect(cache.get(prefix(2))).toBeUndefined();
  });

  it("preserves entries that are re-set", () => {
    const cache = createTrieCache<number>(10, 5);

    cache.set(prefix(1), 1);
    cache.set(prefix(2), 2);

    // Keep prefix(1) alive by re-setting it among the burn writes.
    for (let i = 0; i < 12; i++) {
      cache.set(prefix(200, i), 0);
      if (i % 3 === 0) cache.set(prefix(1), 1);
    }

    // prefix(1) was re-set recently, prefix(2) was not.
    expect(cache.get(prefix(1))).toBe(1);
    expect(cache.get(prefix(2))).toBeUndefined();
  });

  it("clips entire subtrees, not just leaves", () => {
    const cache = createTrieCache<number>(10, 5);

    // Build a deep path.
    cache.set(prefix(1, 2, 3), 100);

    // Let it go stale.
    burnGenerations(cache, 10);

    // The whole subtree under prefix(1) should be gone.
    expect(cache.get(prefix(1, 2, 3))).toBeUndefined();

    // We can reuse the prefix without stale structural nodes.
    cache.set(prefix(1, 2, 3), 200);
    expect(cache.get(prefix(1, 2, 3))).toBe(200);
  });

  it("setting a deep path keeps ancestors alive", () => {
    const cache = createTrieCache<number>(20, 10);

    cache.set(prefix(1), 10);
    cache.set(prefix(1, 2), 20);
    cache.set(prefix(1, 2, 3), 30);

    // Re-set the deepest node among burn writes — ensure() stamps the whole path.
    for (let i = 0; i < 22; i++) {
      cache.set(prefix(200, i), 0);
      if (i % 3 === 0) cache.set(prefix(1, 2, 3), 30);
    }

    // All ancestors should still be alive because ensure stamps the path.
    expect(cache.get(prefix(1))).toBe(10);
    expect(cache.get(prefix(1, 2))).toBe(20);
    expect(cache.get(prefix(1, 2, 3))).toBe(30);
  });
});
