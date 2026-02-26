/**
 * A generic trie cache keyed on byte prefixes with generation-based eviction.
 *
 * Each populated node tracks when it was last accessed. When the number of
 * cached entries exceeds `capacity`, the least-recently-accessed half is
 * pruned, and empty structural nodes are cleaned up.
 */

interface TrieNode<V> {
  children: Map<number, TrieNode<V>>;
  value?: V;
  generation: number;
}

export interface TrieCache<V> {
  /** Return the cached value for `prefix`, or undefined. Touches the node. */
  get(prefix: Uint8Array): V | undefined;
  /** Cache `value` at `prefix`. Touches the node. May trigger eviction. */
  set(prefix: Uint8Array, value: V): void;
  /** Return cached value or compute, cache, and return it. May trigger eviction. */
  getOrSet(prefix: Uint8Array, compute: () => V): V;
  /** Remove the cached value for `prefix`. */
  delete(prefix: Uint8Array): void;
}

export function createTrieCache<V>(capacity = 2048): TrieCache<V> {
  const root: TrieNode<V> = { children: new Map(), generation: 0 };
  let generation = 0;
  let size = 0;

  function walk(prefix: Uint8Array): TrieNode<V> | undefined {
    let node = root;
    for (let i = 0; i < prefix.length; i++) {
      const child = node.children.get(prefix[i]);
      if (!child) return undefined;
      node = child;
    }
    return node;
  }

  function ensure(prefix: Uint8Array): TrieNode<V> {
    let node = root;
    for (let i = 0; i < prefix.length; i++) {
      let child = node.children.get(prefix[i]);
      if (!child) {
        child = { children: new Map(), generation: 0 };
        node.children.set(prefix[i], child);
      }
      node = child;
    }
    return node;
  }

  function prune(): void {
    // Collect all populated nodes with their generations.
    const entries: { node: TrieNode<V>; generation: number }[] = [];
    const stack: TrieNode<V>[] = [root];
    while (stack.length > 0) {
      const node = stack.pop()!;
      if ("value" in node) {
        entries.push({ node, generation: node.generation });
      }
      for (const child of node.children.values()) stack.push(child);
    }

    // Sort oldest first, remove until we're at half capacity.
    entries.sort((a, b) => a.generation - b.generation);
    const target = capacity >>> 1;
    const toRemove = entries.length - target;
    for (let i = 0; i < toRemove; i++) {
      delete entries[i].node.value;
      size--;
    }

    // Remove childless structural nodes (post-order DFS).
    (function cleanup(node: TrieNode<V>): boolean {
      for (const [byte, child] of node.children) {
        if (cleanup(child)) node.children.delete(byte);
      }
      return !("value" in node) && node.children.size === 0;
    })(root);
  }

  return {
    get(prefix) {
      const node = walk(prefix);
      if (node && "value" in node) {
        node.generation = ++generation;
        return node.value;
      }
      return undefined;
    },

    set(prefix, value) {
      const node = ensure(prefix);
      if (!("value" in node)) size++;
      node.value = value;
      node.generation = ++generation;
      if (size > capacity) prune();
    },

    getOrSet(prefix, compute) {
      const node = ensure(prefix);
      if ("value" in node) {
        node.generation = ++generation;
        return node.value!;
      }
      const value = compute();
      node.value = value;
      size++;
      node.generation = ++generation;
      if (size > capacity) prune();
      return value;
    },

    delete(prefix) {
      const node = walk(prefix);
      if (node && "value" in node) {
        delete node.value;
        size--;
      }
    },
  };
}
