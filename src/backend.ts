/**
 * Backend client for the byte-level language model server.
 * - Trie-based cache (shared prefixes → shared trie paths)
 * - Automatic request batching via microtask scheduling
 */

const PREDICT_URL = "http://127.0.0.1:8000/predict";

// ---------------------------------------------------------------------------
// Trie cache
// ---------------------------------------------------------------------------

interface TrieNode {
  children: Map<number, TrieNode>;
  /** Cached 256-element probability distribution. */
  dist?: number[];
}

function trieLookup(root: TrieNode, prefix: Uint8Array): number[] | undefined {
  let node = root;
  for (const byte of prefix) {
    const child = node.children.get(byte);
    if (!child) return undefined;
    node = child;
  }
  return node.dist;
}

function trieInsert(root: TrieNode, prefix: Uint8Array, dist: number[]): void {
  let node = root;
  for (const byte of prefix) {
    let child = node.children.get(byte);
    if (!child) {
      child = { children: new Map() };
      node.children.set(byte, child);
    }
    node = child;
  }
  node.dist = dist;
}

// ---------------------------------------------------------------------------
// Batched requests
// ---------------------------------------------------------------------------

interface PendingRequest {
  prefix: Uint8Array;
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
    return btoa(bin);
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
      predictions: number[][];
    };
    for (let i = 0; i < batch.length; i++) {
      trieInsert(cache, batch[i].prefix, predictions[i]);
      batch[i].resolve(predictions[i]);
    }
  } catch (err) {
    const error = err instanceof Error ? err : new Error(String(err));
    for (const req of batch) req.reject(error);
  }
}

/**
 * Predict the next-byte distribution for a given byte prefix.
 * Returns a 256-element probability array.
 *
 * Results are cached in a trie. Cache misses are batched: all
 * requests enqueued within the same microtask checkpoint are sent
 * in a single HTTP request.
 */
export function predictBytes(prefix: Uint8Array): Promise<number[]> {
  const hit = trieLookup(cache, prefix);
  if (hit) return Promise.resolve(hit);

  return new Promise<number[]>((resolve, reject) => {
    pending.push({ prefix, resolve, reject });
    if (!flushScheduled) {
      flushScheduled = true;
      setTimeout(flush, 100);
    }
  });
}
