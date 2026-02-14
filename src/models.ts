import type { LanguageModel, TokenProb } from "./types";

/**
 * Trie node keyed by token values. Each node optionally holds
 * the cached distribution for that prefix.
 */
interface TrieNode<T> {
  children: Map<T, TrieNode<T>>;
  value?: readonly TokenProb<T>[];
}

function makeNode<T>(): TrieNode<T> {
  return { children: new Map() };
}

/** Walk the trie to the node for `prefix`, creating nodes along the way if `create` is true. */
function walk<T>(
  root: TrieNode<T>,
  prefix: readonly T[],
  create: boolean,
): TrieNode<T> | undefined {
  let node = root;
  for (const token of prefix) {
    let child = node.children.get(token);
    if (!child) {
      if (!create) return undefined;
      child = makeNode<T>();
      node.children.set(token, child);
    }
    node = child;
  }
  return node;
}

/** Wrap a LanguageModel with a trie-based memoization layer. */
export function memoize<T>(model: LanguageModel<T>): LanguageModel<T> {
  const root: TrieNode<T> = makeNode<T>();
  const inflight = new Map<string, Promise<readonly TokenProb<T>[]>>();

  return (prefix: readonly T[]) => {
    const cached = walk(root, prefix, false);
    if (cached?.value) return Promise.resolve(cached.value);

    // Deduplicate concurrent calls for the same prefix.
    // JSON.stringify is fine here — prefixes are short arrays of primitives.
    const key = JSON.stringify(prefix);
    let pending = inflight.get(key);
    if (pending) return pending;

    pending = model(prefix).then((result) => {
      walk(root, prefix, true)!.value = result;
      inflight.delete(key);
      return result;
    });
    inflight.set(key, pending);
    return pending;
  };
}
