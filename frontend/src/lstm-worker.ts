/**
 * Web worker entry point for LSTM inference.
 * Each worker loads its own copy of the model (with its own WASM instance)
 * and handles predict requests with state save/restore.
 */
import { ByteLSTM, loadModel, type LSTMState } from "./lstm";

let model: ByteLSTM | null = null;

export type WorkerRequest =
  | { type: "init"; modelUrl: string }
  | { type: "predict"; id: number; state: LSTMState | null; bytes: number[] };

export type WorkerResponse =
  | { type: "ready" }
  | { type: "result"; id: number; states: LSTMState[]; logits: Float32Array }
  | { type: "error"; id: number; message: string };

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;

  if (msg.type === "init") {
    try {
      model = await loadModel(msg.modelUrl);
      (self as unknown as Worker).postMessage({
        type: "ready",
      } satisfies WorkerResponse);
    } catch (err) {
      // No id for init errors — the pool handles init via the ready promise
      throw err;
    }
    return;
  }

  if (msg.type === "predict") {
    try {
      if (!model) throw new Error("Model not loaded");

      if (msg.state) {
        model.restoreState(msg.state);
      } else {
        model.reset();
      }

      const states: LSTMState[] = [];
      for (const b of msg.bytes) {
        model.step(b);
        states.push(model.saveState());
      }

      // Project final logits
      const logits = model.project();

      (self as unknown as Worker).postMessage({
        type: "result",
        id: msg.id,
        states,
        logits,
      } satisfies WorkerResponse);
    } catch (err) {
      (self as unknown as Worker).postMessage({
        type: "error",
        id: msg.id,
        message: err instanceof Error ? err.message : String(err),
      } satisfies WorkerResponse);
    }
  }
};
