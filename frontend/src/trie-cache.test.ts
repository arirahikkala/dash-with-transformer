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

describe("generation-based eviction", () => {
  it("evicts oldest entries when capacity is exceeded", () => {
    const cache = createTrieCache<number>(4);

    // Fill to capacity.
    cache.set(prefix(1), 1);
    cache.set(prefix(2), 2);
    cache.set(prefix(3), 3);
    cache.set(prefix(4), 4);

    // All four present.
    expect(cache.get(prefix(1))).toBe(1);
    expect(cache.get(prefix(4))).toBe(4);

    // Insert a 5th — triggers prune down to capacity/2 = 2.
    // Oldest by insertion order are 1,2,3,4 but get() above touched
    // 1 and 4, making 2 and 3 the oldest.
    cache.set(prefix(5), 5);

    // The two most recently touched (4 and 5) survive, plus 1 was
    // touched by the get() above.  2 and 3 should be evicted.
    // After prune, we keep capacity/2 = 2 entries.  The 5th entry
    // was just inserted, so the 3 oldest of the 5 are pruned.
    expect(cache.get(prefix(2))).toBeUndefined();
    expect(cache.get(prefix(3))).toBeUndefined();
  });

  it("preserves recently-accessed entries over older ones", () => {
    const cache = createTrieCache<number>(4);

    cache.set(prefix(1), 1);
    cache.set(prefix(2), 2);
    cache.set(prefix(3), 3);
    cache.set(prefix(4), 4);

    // Touch the oldest entry to refresh it.
    cache.get(prefix(1));

    // Trigger eviction.
    cache.set(prefix(5), 5);

    // 1 was refreshed, so it should survive.  2 is the oldest.
    expect(cache.get(prefix(1))).toBe(1);
    expect(cache.get(prefix(2))).toBeUndefined();
  });

  it("can refill after eviction", () => {
    const cache = createTrieCache<number>(4);

    for (let i = 0; i < 10; i++) {
      cache.set(prefix(i), i);
    }

    // The most recent entries should be accessible.
    expect(cache.get(prefix(9))).toBe(9);

    // Old entries should have been evicted.
    expect(cache.get(prefix(0))).toBeUndefined();
  });

  it("cleans up empty structural nodes after eviction", () => {
    const cache = createTrieCache<number>(2);

    // Deep paths that share no structure.
    cache.set(prefix(1, 2, 3), 100);
    cache.set(prefix(4, 5, 6), 200);

    // This triggers eviction — the oldest entry's structural nodes
    // should be cleaned up, so a fresh set doesn't find stale nodes.
    cache.set(prefix(7, 8, 9), 300);

    expect(cache.get(prefix(7, 8, 9))).toBe(300);
    // One of the first two was evicted.
    const a = cache.get(prefix(1, 2, 3));
    const b = cache.get(prefix(4, 5, 6));
    expect([a, b]).toContain(undefined);
  });
});
