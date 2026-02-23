import {
  AutoModelForCausalLM,
  AutoTokenizer,
  softmax,
  Tensor,
} from "@huggingface/transformers";
import type { LanguageModel, TokenProb } from "./types";
import { detokenize } from "./detokenize";

const MODEL_ID = "onnx-community/SmolLM2-135M-ONNX";

/**
 * Convert a raw SentencePiece vocabulary piece to the text it represents.
 */
function pieceToText(piece: string, specialTokens: Set<string>): string {
  if (specialTokens.has(piece)) return "";

  // Byte tokens: <0xHH>
  const m = piece.match(/^<0x([0-9A-Fa-f]{2})>$/);
  if (m) return String.fromCharCode(parseInt(m[1], 16));

  // SentencePiece: ▁ (U+2581) represents a space
  return piece.split("\u2581").join(" ");
}

export async function loadSmolLM(
  onProgress?: (message: string) => void,
): Promise<LanguageModel<number>> {
  onProgress?.("Loading tokenizer\u2026");
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);

  onProgress?.("Loading model\u2026");
  const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    progress_callback: (progress: {
      status: string;
      file?: string;
      progress?: number;
    }) => {
      if (progress.status === "progress" && progress.file) {
        const pct = Math.round(progress.progress ?? 0);
        onProgress?.(`Downloading ${progress.file}: ${pct}%`);
      }
    },
  });

  onProgress?.("Building vocabulary\u2026");

  const specialTokens = new Set<string>(
    (tokenizer as any).all_special_tokens ?? [],
  );

  // Build vocabulary: id → decoded text
  const rawVocab: Map<string, number> = tokenizer.get_vocab();
  const vocabSize = rawVocab.size;
  const vocab: string[] = new Array(vocabSize).fill("");
  const textToId = new Map<string, number>();

  for (const [piece, id] of rawVocab) {
    const text = pieceToText(piece, specialTokens);
    vocab[id] = text;
    if (text.length > 0) textToId.set(text, id);
  }

  const bosTokenId: number = (tokenizer as any).bos_token_id ?? 1;

  // Token-level language model: prefix is an array of decoded token strings
  const tokenModel: LanguageModel<string> = async (
    prefix: readonly string[],
  ): Promise<readonly TokenProb<string>[]> => {
    console.log("predicting on prefix", prefix);
    const preEverything = performance.now();
    // Convert token strings to IDs, always prepend BOS
    const ids = [bosTokenId];
    for (const tok of prefix) {
      const id = textToId.get(tok);
      if (id !== undefined) ids.push(id);
    }

    const seqLen = ids.length;
    const inputIds = new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [
      1,
      seqLen,
    ]);
    const attentionMask = new Tensor(
      "int64",
      BigInt64Array.from(new Array(seqLen).fill(1n)),
      [1, seqLen],
    );

    const preForward = performance.now();
    const output = await model.forward({
      input_ids: inputIds,
      attention_mask: attentionMask,
    });
    console.log(`forward: ${performance.now() - preForward}`);

    // Extract last-position logits: shape [1, seq_len, vocab_size]
    const logitsData = output.logits.data as Float32Array;
    const vSize = output.logits.dims[2];
    const offset = (seqLen - 1) * vSize;

    // Copy to a plain Array for softmax (avoid typed-array edge cases)
    const lastLogits = new Array<number>(vSize);
    for (let i = 0; i < vSize; i++) {
      lastLogits[i] = logitsData[offset + i];
    }

    const preSoftmax = performance.now();
    const probs = softmax(lastLogits) as number[];
    console.log(`softmax: ${performance.now() - preSoftmax}`);

    const result: TokenProb<string>[] = [];
    for (let i = 0; i < vSize; i++) {
      const p = probs[i];
      if (p > 0 && vocab[i].length > 0) {
        result.push({ token: vocab[i], probability: p });
      }
    }
    console.log(`the whole shebang: ${performance.now() - preEverything}`);
    return result;
  };

  onProgress?.("Ready!");
  return detokenize(tokenModel, vocab, 8, 1e-4);
}
