/**
 * Backend client for the byte-level language model server.
 * - Trie-based cache (shared prefixes → shared trie paths)
 * - Automatic request batching via microtask scheduling
 */

import { showErrorToast } from "../toast";
import { createTrieCache, type TrieCache } from "../trie-cache";

// ---------------------------------------------------------------------------
// Request compression
// ---------------------------------------------------------------------------

async function gzipCompress(text: string): Promise<Blob> {
  const stream = new Blob([text])
    .stream()
    .pipeThrough(new CompressionStream("gzip"));
  return new Response(stream).blob();
}

// ---------------------------------------------------------------------------
// Trie response from backend
// ---------------------------------------------------------------------------

interface TrieResponse {
  dist: number[];
  children: Record<string, TrieResponse>;
}

// ---------------------------------------------------------------------------
// Batched requests
// ---------------------------------------------------------------------------

interface PendingRequest {
  prefix: Uint8Array;
  minProb: number;
  resolve: (dist: number[]) => void;
  reject: (err: Error) => void;
}

export interface BackendClient {
  predictBytes(prefix: Uint8Array, minProb: number): Promise<number[]>;
}

export function createBackendClient(backendUrl: string): BackendClient {
  const predictUrl = `${backendUrl}/predict`;
  const cache: TrieCache<Promise<number[]>> = createTrieCache();
  let pending: PendingRequest[] = [];
  let flushScheduled = false;

  function populateTrieCache(prefix: Uint8Array, trie: TrieResponse): void {
    for (const [byteStr, childTrie] of Object.entries(trie.children)) {
      const byte = Number(byteStr);
      const childPrefix = new Uint8Array(prefix.length + 1);
      childPrefix.set(prefix);
      childPrefix[prefix.length] = byte;
      cache.getOrSet(childPrefix, () => Promise.resolve(childTrie.dist));
      populateTrieCache(childPrefix, childTrie);
    }
  }

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
        min_prob: req.minProb,
      };
    });

    try {
      const body = JSON.stringify({ inputs });
      const compressed = await gzipCompress(body);
      const resp = await fetch(predictUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Encoding": "gzip",
        },
        body: compressed,
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
      showErrorToast(error.message);
      for (const req of batch) {
        cache.delete(req.prefix);
        req.reject(error);
      }
    }
  }

  function predictBytes(
    prefix: Uint8Array,
    minProb: number,
  ): Promise<number[]> {
    const cached = cache.get(prefix);
    if (cached) return cached;
    const promise = new Promise<number[]>((resolve, reject) => {
      pending.push({
        prefix,
        minProb: minProb,
        resolve,
        reject,
      });
      if (!flushScheduled) {
        flushScheduled = true;
        setTimeout(flush, 500);
      }
    });
    cache.set(prefix, promise);
    return promise;
  }

  return { predictBytes };
}
