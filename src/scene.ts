import { TrigramModel } from "./trigram";

/** A single character band in the prediction column. */
export interface Band {
  /** The predicted character code. */
  charCode: number;
  /** Display label (□ for space, the character itself otherwise). */
  label: string;
  /** Top edge, 0..1 fraction of the widget height. */
  y0: number;
  /** Bottom edge, 0..1 fraction of the widget height. */
  y1: number;
}

/** Everything needed to render one frame of the widget. */
export interface Scene {
  /** The current sentence prefix. */
  prefix: string;
  /** Predicted-character bands, sorted by char code ascending (top to bottom). */
  bands: Band[];
  /** Crosshairs position, 0..1 in both axes. */
  crosshairs: { x: number; y: number };
}

/** Printable ASCII (32..126) plus newline (10). */
function isPrintableOrNewline(code: number): boolean {
  return code === 10 || (code >= 32 && code <= 126);
}

function labelFor(code: number): string {
  if (code === 32) return "\u25A1"; // □
  if (code === 10) return "\u23CE"; // ⏎
  return String.fromCharCode(code);
}

/**
 * Given a prefix and a trigram model, produce a Scene describing the
 * next-character prediction bands.
 *
 * The last two characters of the prefix are fed to the model. If the
 * prefix is shorter than 2 characters it is left-padded with spaces.
 */
export function buildScene(prefix: string, model: TrigramModel): Scene {
  const padded = prefix.padStart(2, " ");
  const context = padded.slice(-2);
  const counts = model.predict(context);

  // Filter to printable + newline, collect (charCode, count) pairs.
  const entries: { charCode: number; count: number }[] = [];
  let total = 0;
  for (let c = 0; c < counts.length; c++) {
    if (!isPrintableOrNewline(c)) continue;
    if (counts[c] <= 0) continue;
    entries.push({ charCode: c, count: counts[c] });
    total += counts[c];
  }

  // Sort by char code (alphabetical top-to-bottom).
  entries.sort((a, b) => a.charCode - b.charCode);

  // Convert counts to bands with normalized y extents.
  let y = 0;
  const bands: Band[] = entries.map((e) => {
    const height = e.count / total;
    const band: Band = {
      charCode: e.charCode,
      label: labelFor(e.charCode),
      y0: y,
      y1: y + height,
    };
    y += height;
    return band;
  });

  return {
    prefix,
    bands,
    crosshairs: { x: 0.5, y: 0.5 },
  };
}
