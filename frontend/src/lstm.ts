/**
 * Tiny byte-level LSTM language model — pure TypeScript, zero dependencies.
 * Runs entirely on the CPU via typed arrays.
 */

// ---------- Types ----------

interface WeightEntry {
  name: string;
  shape: number[];
  dtype: "float32" | "int8";
  offset: number;
  length: number;
  scale?: number; // only for int8
}

interface ModelManifest {
  config: {
    embed_dim: number;
    hidden_dim: number;
    num_layers: number;
    vocab_size: number;
  };
  weights: WeightEntry[];
  quantized: boolean;
}

export interface LSTMState {
  h: Float32Array[]; // one per layer
  c: Float32Array[]; // one per layer
}

// ---------- Matrix math ----------

/** Matrix-vector multiply: out = A @ x. A is (rows, cols), x is (cols,). */
function matvec(
  out: Float32Array,
  A: Float32Array,
  x: Float32Array,
  rows: number,
  cols: number,
): void {
  for (let i = 0; i < rows; i++) {
    let sum = 0;
    const base = i * cols;
    for (let j = 0; j < cols; j++) {
      sum += A[base + j] * x[j];
    }
    out[i] = sum;
  }
}

/** Dequantize int8 → float32 inline. */
function dequantize(int8Arr: Int8Array, scale: number): Float32Array {
  const out = new Float32Array(int8Arr.length);
  for (let i = 0; i < int8Arr.length; i++) {
    out[i] = int8Arr[i] * scale;
  }
  return out;
}

// ---------- Activations ----------

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// ---------- Weight loading ----------

function loadWeight(entry: WeightEntry, buffer: ArrayBuffer): Float32Array {
  if (entry.dtype === "float32") {
    return new Float32Array(buffer, entry.offset, entry.length / 4);
  } else {
    const int8 = new Int8Array(buffer, entry.offset, entry.length);
    return dequantize(int8, entry.scale!);
  }
}

function getWeight(
  weights: Map<string, Float32Array>,
  name: string,
): Float32Array {
  const w = weights.get(name);
  if (!w) throw new Error(`Weight not found: ${name}`);
  return w;
}

// ---------- Model ----------

export class ByteLSTM {
  private config: ModelManifest["config"];
  private embed: Float32Array;
  private layers: {
    weight_ih: Float32Array;
    weight_hh: Float32Array;
    bias: Float32Array;
  }[];
  private projWeight: Float32Array;
  private projBias: Float32Array;
  private state: LSTMState;

  // Scratch buffers (pre-allocated to avoid GC pressure)
  private gates_ih: Float32Array;
  private gates_hh: Float32Array;
  private gates: Float32Array;

  constructor(manifest: ModelManifest, buffer: ArrayBuffer) {
    this.config = manifest.config;
    const { hidden_dim, num_layers } = this.config;

    // Load all weights
    const weights = new Map<string, Float32Array>();
    for (const entry of manifest.weights) {
      weights.set(entry.name, loadWeight(entry, buffer));
    }

    this.embed = getWeight(weights, "embed.weight");
    this.layers = [];
    for (let l = 0; l < num_layers; l++) {
      this.layers.push({
        weight_ih: getWeight(weights, `lstm.weight_ih_l${l}`),
        weight_hh: getWeight(weights, `lstm.weight_hh_l${l}`),
        bias: getWeight(weights, `lstm.bias_l${l}`),
      });
    }
    this.projWeight = getWeight(weights, "proj.weight");
    this.projBias = getWeight(weights, "proj.bias");

    // Init state
    this.state = {
      h: Array.from({ length: num_layers }, () => new Float32Array(hidden_dim)),
      c: Array.from({ length: num_layers }, () => new Float32Array(hidden_dim)),
    };

    // Scratch buffers
    this.gates_ih = new Float32Array(4 * hidden_dim);
    this.gates_hh = new Float32Array(4 * hidden_dim);
    this.gates = new Float32Array(4 * hidden_dim);
  }

  /** Reset LSTM hidden state (call between unrelated sequences). */
  reset(): void {
    for (let l = 0; l < this.config.num_layers; l++) {
      this.state.h[l].fill(0);
      this.state.c[l].fill(0);
    }
  }

  /** Clone the current hidden state. */
  saveState(): LSTMState {
    return {
      h: this.state.h.map((a) => new Float32Array(a)),
      c: this.state.c.map((a) => new Float32Array(a)),
    };
  }

  /** Restore hidden state from a previous snapshot. */
  restoreState(state: LSTMState): void {
    for (let l = 0; l < this.config.num_layers; l++) {
      this.state.h[l].set(state.h[l]);
      this.state.c[l].set(state.c[l]);
    }
  }

  /**
   * Feed one byte, return logits (unnormalized log-probabilities) for the next byte.
   * The returned Float32Array has 256 entries.
   */
  next(byte: number): Float32Array {
    const { embed_dim, hidden_dim, num_layers, vocab_size } = this.config;
    const H = hidden_dim;

    // Embedding lookup
    let input: Float32Array = new Float32Array(embed_dim);
    const embOffset = byte * embed_dim;
    for (let i = 0; i < embed_dim; i++) {
      input[i] = this.embed[embOffset + i];
    }

    // LSTM layers
    for (let l = 0; l < num_layers; l++) {
      const { weight_ih, weight_hh, bias } = this.layers[l];
      const h_prev = this.state.h[l];
      const c_prev = this.state.c[l];
      const input_size = l === 0 ? embed_dim : hidden_dim;

      // gates = W_ih @ input + W_hh @ h_prev + bias
      matvec(this.gates_ih, weight_ih, input, 4 * H, input_size);
      matvec(this.gates_hh, weight_hh, h_prev, 4 * H, H);

      for (let i = 0; i < 4 * H; i++) {
        this.gates[i] = this.gates_ih[i] + this.gates_hh[i] + bias[i];
      }

      // Split gates: PyTorch order is [i, f, g, o]
      const h_new = this.state.h[l];
      const c_new = this.state.c[l];
      for (let i = 0; i < H; i++) {
        const ig = sigmoid(this.gates[i]); // input gate
        const fg = sigmoid(this.gates[H + i]); // forget gate
        const gg = Math.tanh(this.gates[2 * H + i]); // cell gate
        const og = sigmoid(this.gates[3 * H + i]); // output gate

        c_new[i] = fg * c_prev[i] + ig * gg;
        h_new[i] = og * Math.tanh(c_new[i]);
      }

      // Output of this layer is input to next
      input = h_new;
    }

    // Output projection: logits = W_proj @ h + b_proj
    const logits = new Float32Array(vocab_size);
    matvec(
      logits,
      this.projWeight,
      this.state.h[num_layers - 1],
      vocab_size,
      H,
    );
    for (let i = 0; i < vocab_size; i++) {
      logits[i] += this.projBias[i];
    }

    return logits;
  }

  /**
   * Feed a sequence of bytes, return logits after the last byte.
   * For an empty sequence, returns logits projected from the current hidden state.
   */
  forward(bytes: Uint8Array | number[]): Float32Array {
    for (const b of bytes) {
      this.next(b);
    }
    const { hidden_dim: H, num_layers, vocab_size } = this.config;
    const logits = new Float32Array(vocab_size);
    matvec(
      logits,
      this.projWeight,
      this.state.h[num_layers - 1],
      vocab_size,
      H,
    );
    for (let i = 0; i < vocab_size; i++) {
      logits[i] += this.projBias[i];
    }
    return logits;
  }
}

/** Convert logits to probabilities via softmax. */
export function softmax(logits: Float32Array): Float32Array {
  const probs = new Float32Array(logits.length);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    probs[i] = Math.exp(logits[i] - max);
    sum += probs[i];
  }
  for (let i = 0; i < logits.length; i++) {
    probs[i] /= sum;
  }
  return probs;
}

/** Sample a byte from a probability distribution with temperature. */
export function sample(probs: Float32Array, temperature: number = 1.0): number {
  if (temperature !== 1.0) {
    // Re-apply temperature to logits (convert probs back, scale, re-softmax)
    const logits = new Float32Array(probs.length);
    for (let i = 0; i < probs.length; i++) {
      logits[i] = Math.log(probs[i] + 1e-10) / temperature;
    }
    probs = softmax(logits);
  }
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) return i;
  }
  return probs.length - 1;
}

/**
 * Load a ByteLSTM model from a URL prefix.
 * Expects `${urlPrefix}/model.json` and `${urlPrefix}/model.bin`
 * (or model_q8.json / model_q8.bin for quantized).
 */
export async function loadModel(
  urlPrefix: string,
  quantized: boolean = false,
): Promise<ByteLSTM> {
  const suffix = quantized ? "_q8" : "";
  const [manifestResp, binResp] = await Promise.all([
    fetch(`${urlPrefix}/model${suffix}.json`),
    fetch(`${urlPrefix}/model${suffix}.bin`),
  ]);
  const manifest: ModelManifest = await manifestResp.json();
  const buffer = await binResp.arrayBuffer();
  return new ByteLSTM(manifest, buffer);
}

// ---------- LRU State Cache ----------

class LSTMStateCache {
  private cache = new Map<string, { state: LSTMState; lastAccess: number }>();
  private accessCounter = 0;

  constructor(private capacity: number = 1000) {}

  private prefixKey(prefix: Uint8Array): string {
    return String.fromCharCode(...prefix);
  }

  findLongestPrefix(
    prefix: Uint8Array,
  ): { state: LSTMState; length: number } | null {
    for (let len = prefix.length; len > 0; len--) {
      const key = this.prefixKey(prefix.subarray(0, len));
      const entry = this.cache.get(key);
      if (entry) {
        entry.lastAccess = ++this.accessCounter;
        return { state: entry.state, length: len };
      }
    }
    return null;
  }

  set(prefix: Uint8Array, state: LSTMState): void {
    const key = this.prefixKey(prefix);
    this.cache.set(key, { state, lastAccess: ++this.accessCounter });

    if (this.cache.size > this.capacity) {
      // Evict least recently accessed entry
      let oldestKey: string | null = null;
      let oldestAccess = Infinity;
      for (const [k, v] of this.cache) {
        if (v.lastAccess < oldestAccess) {
          oldestAccess = v.lastAccess;
          oldestKey = k;
        }
      }
      if (oldestKey !== null) this.cache.delete(oldestKey);
    }
  }
}

// ---------- Worker Pool ----------

import type { WorkerResponse } from "./lstm-worker";

interface PendingRequest {
  state: LSTMState | null;
  bytes: number[];
  resolve: (result: { states: LSTMState[]; logits: Float32Array }) => void;
  reject: (err: Error) => void;
}

class LSTMWorkerPool {
  private workers: Worker[] = [];
  private idleWorkers: number[] = [];
  private pendingRequests: PendingRequest[] = [];
  private nextId = 0;
  private inflightCallbacks = new Map<
    number,
    {
      resolve: (result: { states: LSTMState[]; logits: Float32Array }) => void;
      reject: (err: Error) => void;
    }
  >();
  private disposed = false;

  private constructor() {}

  static async create(
    modelUrl: string,
    quantized: boolean,
    numWorkers: number,
    onProgress?: (msg: string) => void,
  ): Promise<LSTMWorkerPool> {
    const pool = new LSTMWorkerPool();

    numWorkers = Math.max(1, numWorkers);
    onProgress?.(`Spawning ${numWorkers} LSTM worker(s)\u2026`);

    const readyPromises: Promise<void>[] = [];

    for (let i = 0; i < numWorkers; i++) {
      const worker = new Worker(new URL("./lstm-worker.ts", import.meta.url), {
        type: "module",
      });

      const readyPromise = new Promise<void>((resolve, reject) => {
        const onMessage = (e: MessageEvent<WorkerResponse>) => {
          if (e.data.type === "ready") {
            worker.removeEventListener("message", onMessage);
            worker.removeEventListener("error", onError);
            pool.setupWorker(worker, i);
            resolve();
          }
        };
        const onError = (e: ErrorEvent) => {
          worker.removeEventListener("message", onMessage);
          worker.removeEventListener("error", onError);
          reject(new Error(`Worker ${i} failed to init: ${e.message}`));
        };
        worker.addEventListener("message", onMessage);
        worker.addEventListener("error", onError);
      });

      worker.postMessage({ type: "init", modelUrl, quantized });
      pool.workers.push(worker);
      readyPromises.push(readyPromise);
    }

    await Promise.all(readyPromises);
    onProgress?.("LSTM workers ready");
    return pool;
  }

  private setupWorker(worker: Worker, index: number): void {
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === "result") {
        const cb = this.inflightCallbacks.get(msg.id);
        this.inflightCallbacks.delete(msg.id);
        cb?.resolve({ states: msg.states, logits: msg.logits });
      } else if (msg.type === "error") {
        const cb = this.inflightCallbacks.get(msg.id);
        this.inflightCallbacks.delete(msg.id);
        cb?.reject(new Error(msg.message));
      } else {
        return; // ignore unexpected messages
      }
      this.returnWorker(index);
    };
    this.idleWorkers.push(index);
  }

  private returnWorker(index: number): void {
    const next = this.pendingRequests.shift();
    if (next) {
      this.dispatch(index, next);
    } else {
      this.idleWorkers.push(index);
    }
  }

  private dispatch(workerIndex: number, req: PendingRequest): void {
    const id = this.nextId++;
    this.inflightCallbacks.set(id, {
      resolve: req.resolve,
      reject: req.reject,
    });
    this.workers[workerIndex].postMessage({
      type: "predict",
      id,
      state: req.state,
      bytes: req.bytes,
    });
  }

  predict(
    state: LSTMState | null,
    bytes: number[],
  ): Promise<{ states: LSTMState[]; logits: Float32Array }> {
    if (this.disposed) {
      return Promise.reject(new Error("Worker pool disposed"));
    }

    return new Promise((resolve, reject) => {
      const workerIndex = this.idleWorkers.shift();
      if (workerIndex !== undefined) {
        this.dispatch(workerIndex, { state, bytes, resolve, reject });
      } else {
        this.pendingRequests.push({ state, bytes, resolve, reject });
      }
    });
  }

  dispose(): void {
    this.disposed = true;
    for (const worker of this.workers) {
      worker.terminate();
    }
    for (const req of this.pendingRequests) {
      req.reject(new Error("Worker pool disposed"));
    }
    this.pendingRequests.length = 0;
    for (const cb of this.inflightCallbacks.values()) {
      cb.reject(new Error("Worker pool disposed"));
    }
    this.inflightCallbacks.clear();
  }
}

// ---------- Public API ----------

import type { PlainTokenProb } from "./types";

export async function createCachedLSTMPredictor(
  urlPrefix: string,
  quantized: boolean,
  onProgress?: (msg: string) => void,
): Promise<{
  predict: (prefix: Uint8Array) => Promise<readonly PlainTokenProb<number>[]>;
  dispose: () => void;
}> {
  onProgress?.("Loading LSTM model\u2026");

  const numWorkers = Math.min(
    typeof navigator !== "undefined" ? (navigator.hardwareConcurrency ?? 1) : 1,
    4,
  );

  const pool = await LSTMWorkerPool.create(
    urlPrefix,
    quantized,
    numWorkers,
    onProgress,
  );

  const cache = new LSTMStateCache();

  onProgress?.("Ready!");

  async function predict(
    prefix: Uint8Array,
  ): Promise<readonly PlainTokenProb<number>[]> {
    // Find longest cached prefix
    let ancestorState: LSTMState | null = null;
    let cachedLen = 0;

    if (prefix.length > 0) {
      const hit = cache.findLongestPrefix(prefix);
      if (hit) {
        ancestorState = hit.state;
        cachedLen = hit.length;
      }
    }

    const remainingBytes = Array.from(prefix.subarray(cachedLen));

    // If nothing to process, still need logits from the current state
    if (remainingBytes.length === 0 && prefix.length === 0) {
      // Empty prefix: reset state, get logits
      const { logits } = await pool.predict(null, []);
      const probs = softmax(logits);
      return toPlainProbs(probs);
    }

    if (remainingBytes.length === 0) {
      // Fully cached prefix — still need to get logits.
      // Send a predict with no new bytes.
      const { logits } = await pool.predict(ancestorState, []);
      const probs = softmax(logits);
      return toPlainProbs(probs);
    }

    const { states, logits } = await pool.predict(
      ancestorState,
      remainingBytes,
    );

    // Cache all intermediate states
    for (let i = 0; i < states.length; i++) {
      cache.set(prefix.subarray(0, cachedLen + i + 1), states[i]);
    }

    const probs = softmax(logits);
    return toPlainProbs(probs);
  }

  return {
    predict,
    dispose: () => pool.dispose(),
  };
}

function toPlainProbs(probs: Float32Array): PlainTokenProb<number>[] {
  const result: PlainTokenProb<number>[] = [];
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > 0) {
      result.push({ token: i, probability: probs[i] });
    }
  }
  return result;
}
