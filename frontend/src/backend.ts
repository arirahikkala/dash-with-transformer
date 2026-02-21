/**
 * Backend client for the byte-level language model server.
 * - Trie-based cache (shared prefixes → shared trie paths)
 * - Automatic request batching via microtask scheduling
 */

const PREDICT_URL = "https://ari-rahikkala.net/predict";

// ---------------------------------------------------------------------------
// Trie cache
// ---------------------------------------------------------------------------

interface TrieNode {
  children: Map<number, TrieNode>;
  /** Cached or in-flight probability distribution (256 elements). */
  dist?: Promise<number[]>;
}

/** Walk (or create) the trie path for `prefix` and return the leaf node. */
function trieEnsure(root: TrieNode, prefix: Uint8Array): TrieNode {
  let node = root;
  for (const byte of prefix) {
    let child = node.children.get(byte);
    if (!child) {
      child = { children: new Map() };
      node.children.set(byte, child);
    }
    node = child;
  }
  return node;
}

// ---------------------------------------------------------------------------
// Trie response from backend
// ---------------------------------------------------------------------------

interface TrieResponse {
  dist: number[];
  children: Record<string, TrieResponse>;
}

/** Walk a trie response and populate the cache for all child prefixes. */
function populateTrieCache(prefix: Uint8Array, trie: TrieResponse): void {
  for (const [byteStr, childTrie] of Object.entries(trie.children)) {
    const byte = Number(byteStr);
    const childPrefix = new Uint8Array(prefix.length + 1);
    childPrefix.set(prefix);
    childPrefix[prefix.length] = byte;
    const childNode = trieEnsure(cache, childPrefix);
    if (!childNode.dist) {
      childNode.dist = Promise.resolve(childTrie.dist);
    }
    populateTrieCache(childPrefix, childTrie);
  }
}

// ---------------------------------------------------------------------------
// Batched requests
// ---------------------------------------------------------------------------

interface PendingRequest {
  prefix: Uint8Array;
  rangeStart: number;
  rangeEnd: number;
  minSize: number;
  node: TrieNode;
  resolve: (dist: number[]) => void;
  reject: (err: Error) => void;
}

const cache: TrieNode = { children: new Map() };
let pending: PendingRequest[] = [];
let flushScheduled = false;

async function flush(): Promise<void> {
  flushScheduled = false;
  const batch = pending;
  pending = [];
  if (batch.length === 0) return;

  const inputs = batch.map((req) => {
    let bin = "";
    for (const b of req.prefix) bin += String.fromCharCode(b);
    return {
      prefix: btoa(bin),
      range_start: req.rangeStart,
      range_end: req.rangeEnd,
      min_size: req.minSize,
    };
  });

  try {
    const resp = await fetch(PREDICT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs }),
    });
    if (!resp.ok) {
      throw new Error(`predict: ${resp.status} ${await resp.text()}`);
    }
    const { predictions } = (await resp.json()) as {
      predictions: TrieResponse[];
    };
    for (let i = 0; i < batch.length; i++) {
      const trie = predictions[i];
      batch[i].resolve(trie.dist);
      populateTrieCache(batch[i].prefix, trie);
    }
  } catch (err) {
    const error = err instanceof Error ? err : new Error(String(err));
    for (const req of batch) {
      req.node.dist = undefined; // clear so the prefix can be retried
      req.reject(error);
    }
  }
}

/**
 * Predict the next-byte distribution for a given byte prefix.
 * Returns a 256-element probability array.
 *
 * The backend returns a trie of pre-expanded distributions based on
 * rangeStart/rangeEnd/minSize, populating the cache for child prefixes.
 */
export function predictBytes(
  prefix: Uint8Array,
  rangeStart: number,
  rangeEnd: number,
  minSize: number,
): Promise<number[]> {
  const node = trieEnsure(cache, prefix);
  if (node.dist) return node.dist;
  const promise = new Promise<number[]>((resolve, reject) => {
    pending.push({
      prefix,
      rangeStart,
      rangeEnd,
      minSize,
      node,
      resolve,
      reject,
    });
    if (!flushScheduled) {
      flushScheduled = true;
      setTimeout(flush, 100);
    }
  });
  node.dist = promise;
  return promise;
}
