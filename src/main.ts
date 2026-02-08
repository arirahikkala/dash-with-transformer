import { TrigramModel } from "./trigram";
import { buildScene } from "./scene";
import { renderScene } from "./render";

async function main() {
  const resp = await fetch("/model.bin");
  const buffer = await resp.arrayBuffer();
  const model = new TrigramModel(buffer);

  const prefix = "The cat";

  const prefixEl = document.getElementById("prefix-display")!;
  prefixEl.textContent = prefix;

  const canvas = document.getElementById("dasher-canvas") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;

  const scene = buildScene(prefix, model);
  renderScene(ctx, scene, canvas.width, canvas.height);
}

main();
