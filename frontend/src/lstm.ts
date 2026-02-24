/**
 * Byte-level LSTM language model — WASM SIMD accelerated, int8 quantized.
 * All matrix-vector products run in WebAssembly with SIMD128 intrinsics.
 * Weight matrices stay int8 in WASM linear memory (no dequantize-at-load).
 */

// ---------- Types ----------

interface WeightEntry {
  name: string;
  shape: number[];
  dtype: "float32" | "int8";
  offset: number;
  length: number;
  scale?: number;
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

// ---------- WASM ----------

interface WasmExports {
  matvec_fused_i8(
    out: number,
    w_ih: number,
    input: number,
    cols_ih: number,
    scale_ih: number,
    w_hh: number,
    h: number,
    cols_hh: number,
    scale_hh: number,
    bias: number,
    rows: number,
  ): void;
  matvec_i8(
    out: number,
    A: number,
    x: number,
    rows: number,
    cols: number,
    scale: number,
  ): void;
}

/** Byte offset where our data starts (the linker reserves [0, WASM_STACK) for stack). */
const WASM_STACK = 4096;

function align16(n: number): number {
  return (n + 15) & ~15;
}

// ---------- Activations ----------

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// ---------- Model ----------

export class ByteLSTM {
  private wasm: WasmExports;
  private config: ModelManifest["config"];

  // Embedding table (float32 in WASM memory, dequantized at load)
  private embedView: Float32Array;

  // Per-layer weight metadata (byte offsets into WASM memory + scales)
  private layers: {
    wIhPtr: number;
    scaleIh: number;
    inputSize: number;
    wHhPtr: number;
    scaleHh: number;
    biasPtr: number;
  }[];

  // Output projection
  private projPtr: number;
  private projScale: number;
  private projBiasView: Float32Array;

  // Scratch + state (all in WASM linear memory)
  private inputPtr: number;
  private inputView: Float32Array;
  private gatesPtr: number;
  private gatesView: Float32Array;
  private hPtrs: number[];
  private hViews: Float32Array[];
  private cViews: Float32Array[];
  private logitsPtr: number;
  private logitsView: Float32Array;

  constructor(
    manifest: ModelManifest,
    modelBuffer: ArrayBuffer,
    wasmModule: WebAssembly.Module,
  ) {
    this.config = manifest.config;
    const { embed_dim, hidden_dim: H, num_layers, vocab_size } = this.config;

    // ---- Compute memory layout (bump allocator, 16-byte aligned) ----

    let off = WASM_STACK;
    const alloc = (bytes: number) => {
      off = align16(off);
      const ptr = off;
      off += bytes;
      return ptr;
    };

    // Embedding table: float32
    const embedPtr = alloc(vocab_size * embed_dim * 4);

    // Per-layer weights
    const layerLayout: typeof this.layers = [];
    for (let l = 0; l < num_layers; l++) {
      const inputSize = l === 0 ? embed_dim : H;
      layerLayout.push({
        wIhPtr: alloc(4 * H * inputSize), // int8
        wHhPtr: alloc(4 * H * H), // int8
        biasPtr: alloc(4 * H * 4), // float32
        scaleIh: 0,
        scaleHh: 0,
        inputSize,
      });
    }

    // Output projection
    const projPtr = alloc(vocab_size * H); // int8
    const projBiasPtr = alloc(vocab_size * 4); // float32

    // Scratch vectors
    const inputVecPtr = alloc(Math.max(embed_dim, H) * 4);
    const gatesPtr = alloc(4 * H * 4);

    // State vectors
    const hPtrs: number[] = [];
    const cPtrs: number[] = [];
    for (let l = 0; l < num_layers; l++) {
      hPtrs.push(alloc(H * 4));
      cPtrs.push(alloc(H * 4));
    }
    const logitsPtr = alloc(vocab_size * 4);

    // ---- Create WASM memory + instantiate ----

    const pages = Math.ceil(off / 65536);
    const memory = new WebAssembly.Memory({ initial: pages });
    const instance = new WebAssembly.Instance(wasmModule, {
      env: { memory },
    });
    this.wasm = instance.exports as unknown as WasmExports;
    const buf = memory.buffer;

    // ---- Copy weights from model file into WASM memory ----

    const getEntry = (name: string): WeightEntry => {
      const e = manifest.weights.find((w) => w.name === name);
      if (!e) throw new Error(`Weight not found: ${name}`);
      return e;
    };

    // Copy raw int8 bytes into WASM memory, return scale
    const copyI8 = (ptr: number, entry: WeightEntry): number => {
      new Int8Array(buf, ptr, entry.length).set(
        new Int8Array(modelBuffer, entry.offset, entry.length),
      );
      return entry.scale!;
    };

    // Dequantize (int8→float32 or copy float32) into WASM memory
    const dequantInto = (
      ptr: number,
      entry: WeightEntry,
      count: number,
    ): Float32Array => {
      const dst = new Float32Array(buf, ptr, count);
      if (entry.dtype === "int8") {
        const src = new Int8Array(modelBuffer, entry.offset, entry.length);
        const s = entry.scale!;
        for (let i = 0; i < count; i++) dst[i] = src[i] * s;
      } else {
        dst.set(new Float32Array(modelBuffer, entry.offset, count));
      }
      return dst;
    };

    // Embedding (dequantize to float32 — small table, accessed per-byte)
    this.embedView = dequantInto(
      embedPtr,
      getEntry("embed.weight"),
      vocab_size * embed_dim,
    );

    // Layer weights: int8 matrices stay int8, biases dequantized to float32
    this.layers = layerLayout;
    for (let l = 0; l < num_layers; l++) {
      this.layers[l].scaleIh = copyI8(
        this.layers[l].wIhPtr,
        getEntry(`lstm.weight_ih_l${l}`),
      );
      this.layers[l].scaleHh = copyI8(
        this.layers[l].wHhPtr,
        getEntry(`lstm.weight_hh_l${l}`),
      );
      dequantInto(this.layers[l].biasPtr, getEntry(`lstm.bias_l${l}`), 4 * H);
    }

    // Projection
    this.projPtr = projPtr;
    this.projScale = copyI8(projPtr, getEntry("proj.weight"));
    this.projBiasView = dequantInto(
      projBiasPtr,
      getEntry("proj.bias"),
      vocab_size,
    );

    // ---- Create typed array views for JS-side access ----

    this.inputPtr = inputVecPtr;
    this.inputView = new Float32Array(buf, inputVecPtr, Math.max(embed_dim, H));
    this.gatesPtr = gatesPtr;
    this.gatesView = new Float32Array(buf, gatesPtr, 4 * H);
    this.hPtrs = hPtrs;
    this.hViews = hPtrs.map((p) => new Float32Array(buf, p, H));
    this.cViews = cPtrs.map((p) => new Float32Array(buf, p, H));
    this.logitsPtr = logitsPtr;
    this.logitsView = new Float32Array(buf, logitsPtr, vocab_size);
  }

  /** Reset LSTM hidden state (call between unrelated sequences). */
  reset(): void {
    for (const v of this.hViews) v.fill(0);
    for (const v of this.cViews) v.fill(0);
  }

  /** Clone the current hidden state. */
  saveState(): LSTMState {
    return {
      h: this.hViews.map((v) => new Float32Array(v)),
      c: this.cViews.map((v) => new Float32Array(v)),
    };
  }

  /** Restore hidden state from a previous snapshot. */
  restoreState(state: LSTMState): void {
    for (let l = 0; l < this.config.num_layers; l++) {
      this.hViews[l].set(state.h[l]);
      this.cViews[l].set(state.c[l]);
    }
  }

  /** Feed one byte through LSTM layers (no output projection). */
  step(byte: number): void {
    const { embed_dim, hidden_dim: H, num_layers } = this.config;

    // Embedding lookup → input vector in WASM memory
    const embOff = byte * embed_dim;
    for (let i = 0; i < embed_dim; i++)
      this.inputView[i] = this.embedView[embOff + i];

    // LSTM layers
    for (let l = 0; l < num_layers; l++) {
      const layer = this.layers[l];
      const inPtr = l === 0 ? this.inputPtr : this.hPtrs[l - 1];

      // Fused gate computation (WASM SIMD):
      //   gates = W_ih @ input * scale_ih + W_hh @ h * scale_hh + bias
      this.wasm.matvec_fused_i8(
        this.gatesPtr,
        layer.wIhPtr,
        inPtr,
        layer.inputSize,
        layer.scaleIh,
        layer.wHhPtr,
        this.hPtrs[l],
        H,
        layer.scaleHh,
        layer.biasPtr,
        4 * H,
      );

      // Apply gate activations and update cell/hidden state
      const g = this.gatesView;
      const h = this.hViews[l];
      const c = this.cViews[l];
      for (let i = 0; i < H; i++) {
        const ig = sigmoid(g[i]); // input gate
        const fg = sigmoid(g[H + i]); // forget gate
        const gg = Math.tanh(g[2 * H + i]); // cell gate
        const og = sigmoid(g[3 * H + i]); // output gate

        c[i] = fg * c[i] + ig * gg;
        h[i] = og * Math.tanh(c[i]);
      }
    }
  }

  /** Project current hidden state to logits (returns a new Float32Array). */
  project(): Float32Array {
    const { hidden_dim: H, num_layers, vocab_size } = this.config;

    // Output projection (WASM SIMD): logits_raw = W_proj @ h * scale
    this.wasm.matvec_i8(
      this.logitsPtr,
      this.projPtr,
      this.hPtrs[num_layers - 1],
      vocab_size,
      H,
      this.projScale,
    );

    // Add bias (small — 256 elements)
    const logits = new Float32Array(vocab_size);
    for (let i = 0; i < vocab_size; i++)
      logits[i] = this.logitsView[i] + this.projBiasView[i];
    return logits;
  }

  /**
   * Feed one byte, return logits (unnormalized log-probabilities) for the next byte.
   * The returned Float32Array has 256 entries.
   */
  next(byte: number): Float32Array {
    this.step(byte);
    return this.project();
  }

  /**
   * Feed a sequence of bytes, return logits after the last byte.
   * Uses step() internally to avoid redundant intermediate projections.
   */
  forward(bytes: Uint8Array | number[]): Float32Array {
    for (const b of bytes) this.step(b);
    return this.project();
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
 * Expects model_q8.json, model_q8.bin, and matvec.wasm at the prefix.
 */
export async function loadModel(urlPrefix: string): Promise<ByteLSTM> {
  const [manifest, buffer, wasmModule] = await Promise.all([
    fetch(`${urlPrefix}/model_q8.json`).then((r) =>
      r.json(),
    ) as Promise<ModelManifest>,
    fetch(`${urlPrefix}/model_q8.bin`).then((r) => r.arrayBuffer()),
    fetch(`${urlPrefix}/matvec.wasm`)
      .then((r) => r.arrayBuffer())
      .then((b) => WebAssembly.compile(b)),
  ]);
  return new ByteLSTM(manifest, buffer, wasmModule);
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

      worker.postMessage({ type: "init", modelUrl });
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

  const pool = await LSTMWorkerPool.create(urlPrefix, numWorkers, onProgress);

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
