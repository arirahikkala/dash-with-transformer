import {
  AutoModelForCausalLM,
  AutoTokenizer,
  softmax,
  Tensor,
} from "@huggingface/transformers";
import type {
  LanguageModel,
  PlainLanguageModel,
  PlainTokenProb,
} from "./types";
import { detokenize } from "./detokenize";
import { fromCharCodeModel } from "./models";

const MODEL_ID = "onnx-community/SmolLM2-135M-ONNX";

// ---------------------------------------------------------------------------
// GPT-2 byte-level BPE decoding
//
// GPT-2's tokenizer maps every byte to a printable Unicode code point so the
// vocabulary contains no whitespace / control characters.  To recover the
// actual text a token represents we reverse the mapping: Unicode code point →
// original byte value, then decode the byte sequence as UTF-8.
// ---------------------------------------------------------------------------

/** Build the reverse lookup: Unicode code point → byte value. */
function buildUnicodeToByteMap(): Map<number, number> {
  // Bytes that map to themselves (printable ASCII + Latin-1 supplement subset)
  const direct: number[] = [];
  for (let b = 0x21; b <= 0x7e; b++) direct.push(b); // '!' .. '~'
  for (let b = 0xa1; b <= 0xac; b++) direct.push(b); // '¡' .. '¬'
  for (let b = 0xae; b <= 0xff; b++) direct.push(b); // '®' .. 'ÿ'

  const directSet = new Set(direct);
  const map = new Map<number, number>();

  for (const b of direct) map.set(b, b); // identity mapping

  // Remaining bytes (0x00-0x20, 0x7F-0xA0, 0xAD) are shifted to U+0100..
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!directSet.has(b)) {
      map.set(0x100 + n, b);
      n++;
    }
  }
  return map;
}

const UNICODE_TO_BYTE = buildUnicodeToByteMap();

/** Decode a GPT-2 BPE token string into the actual text it represents. */
function pieceToText(piece: string, specialTokens: Set<string>): string {
  if (specialTokens.has(piece)) return "";

  const bytes: number[] = [];
  for (const ch of piece) {
    const cp = ch.codePointAt(0)!;
    const b = UNICODE_TO_BYTE.get(cp);
    if (b === undefined) {
      // Not part of the GPT-2 byte mapping — keep the character as-is.
      // (Shouldn't happen for a well-formed GPT-2 vocabulary.)
      const encoded = new TextEncoder().encode(ch);
      for (const byte of encoded) bytes.push(byte);
    } else {
      bytes.push(b);
    }
  }
  return new TextDecoder("utf-8", { fatal: false }).decode(
    Uint8Array.from(bytes),
  );
}

export async function loadSmolLM(
  onProgress?: (message: string) => void,
): Promise<LanguageModel<readonly number[], number>> {
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
  const tokenModel: PlainLanguageModel<readonly string[], string> = async (
    prefix: readonly string[],
  ): Promise<readonly PlainTokenProb<string>[]> => {
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

    const result: PlainTokenProb<string>[] = [];
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
  return fromCharCodeModel(detokenize(tokenModel, vocab, 8, 1e-4));
}
