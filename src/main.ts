import { TrigramModel } from "./trigram";
import { type LanguageModel, type TokenProb, type Cursor } from "./cursor";
import { buildScene } from "./scene";
import { renderScene, type RenderOptions } from "./render";

/** Printable ASCII (32..126) plus newline (10). */
function isPrintableOrNewline(code: number): boolean {
  return code === 10 || (code >= 32 && code <= 126);
}

/** Wrap a TrigramModel as a generic LanguageModel<number>. */
function wrapTrigramModel(trigram: TrigramModel): LanguageModel<number> {
  return (prefix: readonly number[]): readonly TokenProb<number>[] => {
    // Build the 2-char context string from the last two tokens.
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

    // Collect printable + newline entries with positive counts.
    const entries: { token: number; count: number }[] = [];
    let total = 0;
    for (let c = 0; c < counts.length; c++) {
      if (!isPrintableOrNewline(c)) continue;
      if (counts[c] <= 0) continue;
      entries.push({ token: c, count: counts[c] });
      total += counts[c];
    }
    if (total === 0) return [];

    // Sort by char code (alphabetical top-to-bottom).
    entries.sort((a, b) => a.token - b.token);

    return entries.map((e) => ({
      token: e.token,
      probability: e.count / total,
    }));
  };
}

function labelFor(code: number): string {
  if (code === 32) return "\u25A1"; // □
  if (code === 10) return "\u23CE"; // ⏎
  return String.fromCharCode(code);
}

function colorFor(code: number): string {
  const hue = (code * 137.508) % 360;
  return `hsl(${hue}, 45%, 35%)`;
}

async function main() {
  const resp = await fetch("/model.bin");
  const buffer = await resp.arrayBuffer();
  const trigram = new TrigramModel(buffer);
  const model = wrapTrigramModel(trigram);

  const prefix = "The cat";
  const tokenPrefix = Array.from(prefix).map((ch) => ch.charCodeAt(0));

  const prefixEl = document.getElementById("prefix-display")!;
  prefixEl.textContent = prefix;

  const canvas = document.getElementById("dasher-canvas") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;

  // Cursor at the origin of the prefix's square = fully zoomed out
  const cursor: Cursor<number> = { prefix: tokenPrefix, x: 0, y: 0 };

  const renderOpts: RenderOptions<number> = {
    label: labelFor,
    color: colorFor,
  };

  const scene = buildScene(model, cursor, 0.01);
  renderScene(ctx, scene, canvas.width, canvas.height, renderOpts);
}

main();
