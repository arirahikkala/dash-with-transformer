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
