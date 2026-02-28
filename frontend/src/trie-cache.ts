/**
 * A generic trie cache keyed on numeric prefixes with generation-based eviction.
 *
 * The generation counter advances only when a value is actually written (set
 * or first computation in getOrSet).  Read-only accesses stamp the path with
 * the current generation but don't bump it, making this closer to FIFO than
 * LRU.  Every `pruneInterval` generations, a DFS clips any subtree whose root
 * is older than `generation - maxAge`.
 */

interface TrieNode<V> {
  children: Map<number, TrieNode<V>>;
  value?: V;
  generation: number;
}

export interface TrieCache<V> {
  /** Return the cached value for `prefix`, or undefined. Touches the path. */
  get(prefix: Iterable<number>): V | undefined;
  /** Cache `value` at `prefix`. Touches the path. May trigger pruning. */
  set(prefix: Iterable<number>, value: V): void;
  /** Return cached value or compute, cache, and return it. May trigger pruning. */
  getOrSet(prefix: Iterable<number>, compute: () => V): V;
  /** Remove the cached value for `prefix`. */
  delete(prefix: Iterable<number>): void;
  /** Walk the trie along `prefix` and return the deepest node that has a value. */
  findLongestPrefix(
    prefix: Iterable<number>,
  ): { value: V; length: number } | undefined;
}

export function createTrieCache<V>(
  pruneInterval = 20_000,
  maxAge = 40_000,
): TrieCache<V> {
  const root: TrieNode<V> = { children: new Map(), generation: 0 };
  let generation = 0;
  let nextPrune = pruneInterval;

  function touch(node: TrieNode<V>): void {
    node.generation = generation;
  }

  function walk(prefix: Iterable<number>): TrieNode<V> | undefined {
    let node = root;
    touch(node);
    for (const key of prefix) {
      const child = node.children.get(key);
      if (!child) return undefined;
      node = child;
      touch(node);
    }
    return node;
  }

  function ensure(prefix: Iterable<number>): TrieNode<V> {
    let node = root;
    touch(node);
    for (const key of prefix) {
      let child = node.children.get(key);
      if (!child) {
        child = { children: new Map(), generation };
        node.children.set(key, child);
      }
      node = child;
      touch(node);
    }
    return node;
  }

  function prune(): void {
    const threshold = generation - maxAge;
    (function sweep(node: TrieNode<V>): void {
      for (const [byte, child] of node.children) {
        if (child.generation < threshold) {
          node.children.delete(byte);
        } else {
          sweep(child);
        }
      }
    })(root);
  }

  function tick(): void {
    generation++;
    if (generation >= nextPrune) {
      nextPrune = generation + pruneInterval;
      prune();
    }
  }

  return {
    get(prefix) {
      const node = walk(prefix);
      if (node && "value" in node) {
        return node.value;
      }
      return undefined;
    },

    set(prefix, value) {
      tick();
      const node = ensure(prefix);
      node.value = value;
    },

    getOrSet(prefix, compute) {
      // Try a read-only walk first to avoid bumping generation on cache hits.
      const existing = walk(prefix);
      if (existing && "value" in existing) {
        return existing.value!;
      }
      // Cache miss — now tick and ensure the path exists.
      tick();
      const node = ensure(prefix);
      const value = compute();
      node.value = value;
      return value;
    },

    delete(prefix) {
      const node = walk(prefix);
      if (node && "value" in node) {
        delete node.value;
      }
    },

    findLongestPrefix(prefix) {
      let node = root;
      touch(node);
      let best: { value: V; length: number } | undefined;
      if ("value" in node) {
        best = { value: node.value!, length: 0 };
      }
      let depth = 0;
      for (const key of prefix) {
        const child = node.children.get(key);
        if (!child) break;
        node = child;
        depth++;
        touch(node);
        if ("value" in node) {
          best = { value: node.value!, length: depth };
        }
      }
      return best;
    },
  };
}
