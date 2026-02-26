/**
 * A generic trie cache keyed on byte prefixes with generation-based eviction.
 *
 * Every access stamps all nodes along the root-to-leaf path with the current
 * generation.  Every `pruneInterval` generations, a DFS clips any subtree
 * whose root is older than `generation - maxAge`.  This is cheap (no sorting,
 * no size tracking) and preserves the hot neighbourhood around recently-
 * accessed prefixes.
 */

interface TrieNode<V> {
  children: Map<number, TrieNode<V>>;
  value?: V;
  generation: number;
}

export interface TrieCache<V> {
  /** Return the cached value for `prefix`, or undefined. Touches the path. */
  get(prefix: Uint8Array): V | undefined;
  /** Cache `value` at `prefix`. Touches the path. May trigger pruning. */
  set(prefix: Uint8Array, value: V): void;
  /** Return cached value or compute, cache, and return it. May trigger pruning. */
  getOrSet(prefix: Uint8Array, compute: () => V): V;
  /** Remove the cached value for `prefix`. */
  delete(prefix: Uint8Array): void;
  /** Walk the trie along `prefix` and return the deepest node that has a value. */
  findLongestPrefix(
    prefix: Uint8Array,
  ): { value: V; length: number } | undefined;
}

export function createTrieCache<V>(
  pruneInterval = 200_000,
  maxAge = 400_000,
): TrieCache<V> {
  const root: TrieNode<V> = { children: new Map(), generation: 0 };
  let generation = 0;
  let nextPrune = pruneInterval;

  function touch(node: TrieNode<V>): void {
    node.generation = generation;
  }

  function walk(prefix: Uint8Array): TrieNode<V> | undefined {
    let node = root;
    touch(node);
    for (let i = 0; i < prefix.length; i++) {
      const child = node.children.get(prefix[i]);
      if (!child) return undefined;
      node = child;
      touch(node);
    }
    return node;
  }

  function ensure(prefix: Uint8Array): TrieNode<V> {
    let node = root;
    touch(node);
    for (let i = 0; i < prefix.length; i++) {
      let child = node.children.get(prefix[i]);
      if (!child) {
        child = { children: new Map(), generation };
        node.children.set(prefix[i], child);
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
      tick();
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
      tick();
      const node = ensure(prefix);
      if ("value" in node) {
        return node.value!;
      }
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
      tick();
      let node = root;
      touch(node);
      let best: { value: V; length: number } | undefined;
      if ("value" in node) {
        best = { value: node.value!, length: 0 };
      }
      for (let i = 0; i < prefix.length; i++) {
        const child = node.children.get(prefix[i]);
        if (!child) break;
        node = child;
        touch(node);
        if ("value" in node) {
          best = { value: node.value!, length: i + 1 };
        }
      }
      return best;
    },
  };
}
