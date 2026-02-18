import type { LanguageModel, TokenDisplay, TokenProb } from "./types";

const N = 128;

export class TrigramModel {
  private data: Uint32Array;

  /** Expects the raw 8MB model.bin ArrayBuffer. */
  constructor(buffer: ArrayBuffer) {
    if (buffer.byteLength !== N * N * N * 4) {
      throw new Error(
        `Expected buffer of ${N * N * N * 4} bytes, got ${buffer.byteLength}`,
      );
    }
    this.data = new Uint32Array(buffer);
  }

  /**
   * Given exactly 2 ASCII characters, return the frequency counts
   * for every possible next character (number[128]). The counts are
   * unsmoothed and not normalized, but always non-negative.
   */
  predict(lastChars: string): number[] {
    if (lastChars.length !== 2) {
      throw new Error("lastChars must be exactly 2 characters");
    }
    const a = lastChars.charCodeAt(0);
    const b = lastChars.charCodeAt(1);
    if (a >= N || b >= N) {
      throw new Error("Both characters must be 7-bit ASCII (< 128)");
    }
    const offset = a * N * N + b * N;
    const slice = this.data.subarray(offset, offset + N);
    return Array.from(slice);
  }
}

// ---------------------------------------------------------------------------
// Trigram adapter
// ---------------------------------------------------------------------------

/** Printable ASCII (32..126) plus newline (10). */
function isPrintableOrNewline(code: number): boolean {
  return code === 10 || (code >= 32 && code <= 126);
}

/** Wrap a TrigramModel as a generic LanguageModel<readonly number[], number>. */
function wrapTrigramModel(
  trigram: TrigramModel,
): LanguageModel<readonly number[], number> {
  return async (
    prefix: readonly number[],
  ): Promise<readonly TokenProb<number>[]> => {
    let context: string;
    if (prefix.length === 0) {
      context = "  ";
    } else if (prefix.length === 1) {
      context = " " + String.fromCharCode(prefix[0]);
    } else {
      const a = prefix[prefix.length - 2];
      const b = prefix[prefix.length - 1];
      context = String.fromCharCode(a) + String.fromCharCode(b);
    }

    const counts = trigram.predict(context);

    const entries: { token: number; count: number }[] = [];
    let total = 0;
    for (let c = 0; c < counts.length; c++) {
      if (!isPrintableOrNewline(c)) continue;
      if (counts[c] <= 0) continue;
      entries.push({ token: c, count: counts[c] });
      total += counts[c];
    }
    if (total === 0) return [];

    entries.sort((a, b) => a.token - b.token);

    return entries.map((e) => ({
      token: e.token,
      probability: e.count / total,
    }));
  };
}

// ---------------------------------------------------------------------------
// Render helpers for char-code tokens
// ---------------------------------------------------------------------------

function labelFor(code: number): string {
  if (code === 32) return "\u25A1"; // □
  if (code === 10) return "\u23CE"; // ⏎
  return String.fromCharCode(code);
}

function colorFor(code: number): string {
  const hue = (code * 137.508) % 360;
  return `hsl(${hue}, 45%, 35%)`;
}

function prefixToDisplayString(prefix: readonly number[]): string {
  return prefix
    .map((c) => {
      if (c === 10) return "\u23CE";
      return String.fromCharCode(c);
    })
    .join("");
}

/** Fetch and load the trigram model. */
export async function loadTrigramModel() {
  const resp = await fetch("/model.bin");
  const buffer = await resp.arrayBuffer();
  const trigram = new TrigramModel(buffer);
  const model: LanguageModel<readonly number[], number> =
    wrapTrigramModel(trigram);
  const display: TokenDisplay<number> = {
    label: labelFor,
    color: colorFor,
    prefixToString: prefixToDisplayString,
  };
  return { model, display };
}
